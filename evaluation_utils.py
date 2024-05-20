import re
import json
import torch
import Levenshtein
from tqdm.notebook import tqdm
from collections import defaultdict
from torchmetrics import Accuracy, F1, Precision, Recall
from event_models_utils import get_event_names_dict, load_rams_data, load_ontology


def evaluate_event_results(
    res_dir: str = "results/events",
    test_filename: str = "datasets/rams/raw/test.jsonlines",
    average: str = "weighted"
):

    results = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "accuracy": 0,
        "precision_edit": 0,
        "recall_edit": 0,
        "f1_edit": 0,
        "accuracy_edit": 0,
        "details": []
    }

    evt2sent, sent2evt, _, _ = get_event_names_dict()
    docs, dicts = load_rams_data()
    
    # Specifica il nome e il numero degli eventi.
    evt2idx = dicts["evt2idx"]

    # Quando si usa la media pesata, l'F1-scoe potrebbe non essere compreso
    # fra precision e recall:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    precision = Precision(num_classes=140, average=average)
    recall = Recall(num_classes=140, average=average)
    f1 = F1(num_classes=140, average=average)
    accuracy = Accuracy(num_classes=140, average=average)

    predictions = []
    test_data = []

    y_preds = []
    y_preds_edit = []
    y_true = []

    # Recupero le predizioni del modello.
    with open(
        f"{res_dir}/event_gen_predictions.jsonl",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            predictions.append(obj)

    # Recupero la ground truth.
    with open(
        test_filename,
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            test_data.append(obj)

    for prediction in tqdm(
        predictions,
        desc="Evaluating prediction",
        leave=False
    ):

        predicted_evt, predicted_evt_edit, gold_evt, predicted_evt_sent, gold_evt_sent = find_event_matches(
            prediction, evt2sent, sent2evt, evt2idx
        )

        y_preds.append(predicted_evt)
        y_preds_edit.append(predicted_evt_edit)
        y_true.append(gold_evt)

        d_res = {
            "prediction": predicted_evt_sent,
            "gold": gold_evt_sent,
            "predicted_evt": predicted_evt,
            "predicted_evt_edit": predicted_evt_edit,
            "gold_evt": gold_evt 
        }

        # Esempio di dettaglio:
        # {'prediction': 'This broadcast of a prevarication',
        # 'gold': 'a broadcast about a prevarication',
        # 'predicted_evt': 0,
        # 'predicted_evt': 42,
        # 'gold_evt': 42}
        results["details"].append(d_res)

    y_preds = torch.tensor(y_preds)
    y_true = torch.tensor(y_true)

    precision(y_preds, y_true)
    recall(y_preds, y_true)
    f1(y_preds, y_true)
    accuracy(y_preds, y_true)

    results["precision"] = precision.compute().item()
    results["recall"] = recall.compute().item()
    results["f1"] = f1.compute().item()
    results["accuracy"] = accuracy.compute().item()
    
    y_preds_edit = torch.tensor(y_preds_edit)

    precision(y_preds_edit, y_true)
    recall(y_preds_edit, y_true)
    f1(y_preds_edit, y_true)
    accuracy(y_preds_edit, y_true)
    
    results["precision_edit"] = precision.compute().item()
    results["recall_edit"] = recall.compute().item()
    results["f1_edit"] = f1.compute().item()
    results["accuracy_edit"] = accuracy.compute().item()

    with open(
        f"{res_dir}/event_gen_results.json", 
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)
        
        
def find_event_matches(
    prediction, evt2sent, sent2evt, evt2idx
):
    re_combine_whitespace = re.compile(r"\s+")

    # Questa è semplicemente la frase predetta (come evento) senza spazi all'inizio e alla fine.
    # Esempio: 'This broadcast of a prevarication'.
    predicted_words = re_combine_whitespace.sub(" ", prediction["predicted"]).strip()
    # Questa è semplicemente la frase vera (come evento) senza spazi all'inizio e alla fine.
    # Esempio: 'This document is about a broadcast about a prevarication'.
    gold_words = re_combine_whitespace.sub(" ", prediction["gold"]).strip()

    # Qui vengono rimosse le parole "This document is about".
    predicted_evt_sent = re.sub("This document is about ", "", predicted_words)
    gold_evt_sent = re.sub("This document is about ", "", gold_words)
    
    # Qui vengono rimosso altre parole iniziali se diverse da "This document is about".
    predicted_evt_sent = re.sub("This is about ", "", predicted_evt_sent)
    predicted_evt_sent = re.sub("This is ", "", predicted_evt_sent)
    predicted_evt_sent = re.sub("This ", "", predicted_evt_sent)
    
    # Qui viene individuato il numero dell'evento predetto. Se l'evento predetto
    # non corrisponde ad alcuno degli eventi del dataset, allora viene restituito
    # 0 che corrisponde a un non-evento.
    predicted_evt = 0
    if predicted_evt_sent in sent2evt.keys():
        predicted_evt = sent2evt[predicted_evt_sent].replace("unspecified", "n/a")
        predicted_evt = evt2idx[predicted_evt]
        
    # Qui viene calcolato l'evento predetto come l'evento con la più alta similarità
    # per la distanza di Levenshtein rispetto agli eventi della ground truth.
    # Questo però solo se la similarità è maggiore di 0.5.
    predicted_evt_edit = 0
    similarities = []
    for i in range(len(sent2evt.keys())):
        evt_sent = list(sent2evt.keys())[i]
        distance = Levenshtein.distance(predicted_evt_sent, evt_sent)
        similarity = 1 - (distance / max(len(predicted_evt_sent), len(evt_sent)))
        similarities.append(similarity)
    max_similarity = max(similarities)
    if max_similarity >= 0.5:
        best_match = similarities.index(max_similarity)
        best_match_sent = list(sent2evt.keys())[best_match]
        predicted_evt_edit = sent2evt[best_match_sent].replace("unspecified", "n/a")
        predicted_evt_edit = evt2idx[predicted_evt_edit]

    # Qui viene individuato il numero dell'evento della ground truth.
    gold_evt = sent2evt[gold_evt_sent].replace("unspecified", "n/a")
    gold_evt = evt2idx[gold_evt]

    return predicted_evt, predicted_evt_edit, gold_evt, predicted_evt_sent, gold_evt_sent


def evaluate_arguments_results(
    res_dir: str = "results/arguments",
    test_filename: str = "datasets/rams/raw/test.jsonlines"
):
    """
    Il cacolo di precision, recall e F1-score si basa sullo scorer
    ufficiale del dataset RAMS.
    """
    total_correct = 0
    # Qui consideriamo corretto anche quando l'argomento predett è <arg> e 
    # anche nella ground truth è <arg>.
    total_correct_arg = 0
    total_missing = 0
    total_overpred = 0

    results = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "accuracy": 0,
        "precision_arg": 0,
        "recall_arg": 0,
        "f1_arg": 0,
        "accuracy_arg": 0,
        "details": []
    }

    predictions = []
    test_data = []

    # Recupero le predizioni del modello.
    with open(
        f"{res_dir}/predictions.jsonl",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            predictions.append(obj)

    # Recupero la ground truth.
    with open(
        test_filename,
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            test_data.append(obj)
    
    ontology_dict = load_ontology()

    for prediction in predictions:
        # Questo è l'identificatore del documento.
        doc_key = prediction["doc_key"]
        doc = None
        # Qui viene recuperato tutto il documento sulla base dell'identificatore.
        for d in test_data:
            if d["doc_key"] == doc_key[0]:
                doc = d
        if doc is None:
            raise Exception("Document not found.")
        
        # Qui recupero il documento, inteso come testo.
        doc_sentences = doc["sentences"]
        document = ' '.join([' '.join(word) for word in doc_sentences])
        document = re.sub(r'\s+\.', '.', document)
        document = re.sub(r'\s+\,', ',', document)

        # Questo è il trigger dell'evento. Contiene diverse informazioni tra cui il tipo di evento.
        doc_trigger = doc["evt_triggers"]
        # Questo è lo specifico tipo di evento. 
        event_type = doc_trigger[0][2][0][0]
        event_type = event_type.replace("n/a", "unspecified")
        # Qui viene recuperato il template per quel tipo di evento.
        template = ontology_dict[event_type]["template"]

        # Qui vengono restituiti i risultati sulla predizione che vengono specificati
        # nell'esempio successivo.
        match_dict = find_matches(
            prediction, template, event_type, ontology_dict
        )

        total_correct += match_dict["correct_num"]
        total_correct_arg += match_dict["correct_num_arg"]
        total_missing += match_dict["missing_num"]
        total_overpred += match_dict["overpred_num"]

        '''
        Esempio:
        {'template': '<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)',
        'prediction': 'athletes communicated to us about <arg> topic at <arg> place (one-way communication)',
        'gold': 'athletes communicated to us, and the world about <arg> topic at Russia place (one-way communication)',
        'correct': 1,
        'missing': 1,
        'overpred': 1,
        'predicted_args': defaultdict(list,
                    {'evt043arg01communicator': ['athletes'],
                    'evt043arg02recipient': ['us'],
                    'evt043arg04topic': [None],
                    'evt043arg03place': [None]}),
        'gold_args': defaultdict(list,
                    {'evt043arg01communicator': ['athletes'],
                    'evt043arg02recipient': ['us, and the world'],
                    'evt043arg04topic': [None],
                    'evt043arg03place': ['Russia']})}
        '''
        d_res = {
            "document": document,
            "template": match_dict["template"],
            "prediction": match_dict["predicted"],
            "gold": match_dict["gold"],
            "correct": match_dict["correct_num"],
            "missing": match_dict["missing_num"],
            "overpred": match_dict["overpred_num"],
            "predicted_args": match_dict["predicted_args"],
            "gold_args": match_dict["gold_args"] 
        }

        results["details"].append(d_res)

    # Calcolo precision, recall e F1-score.
    p = float(total_correct) / float(total_correct + total_missing) if (total_correct + total_missing) else 0.0
    r = float(total_correct) / float(total_correct + total_overpred) if (total_correct + total_overpred) else 0.0
    f1 = 2.0 * p * r  / (p + r) if (p + r) else 0.0
    acc = float(total_correct) / float(total_correct + total_missing + total_overpred)
    
    p_arg = float(total_correct_arg) / float(total_correct_arg + total_missing) if (total_correct_arg + total_missing) else 0.0
    r_arg = float(total_correct_arg) / float(total_correct_arg + total_overpred) if (total_correct_arg + total_overpred) else 0.0
    f1_arg = 2.0 * p_arg * r_arg  / (p_arg + r_arg) if (p_arg + r_arg) else 0.0
    acc_arg = float(total_correct_arg) / float(total_correct_arg + total_missing + total_overpred)

    results["precision"] = p
    results["recall"] = r
    results["f1"] = f1
    results["accuracy"] = acc
    results["precision_arg"] = p_arg
    results["recall_arg"] = r_arg
    results["f1_arg"] = f1_arg
    results["accuracy_arg"] = acc_arg

    with open(
        f"{res_dir}/argument_results.json", 
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)
        
        
def find_matches(pred, template, evt_type, ontology_dict):
    """
    Codice basato sullo scorer ufficiale del dataset RAMS:
    https://github.com/raspberryice/gen-arg/blob/1f547018f078aeb6fbcdf7a7a11366a77a53fc7e/src/genie/scorer.py
    """

    re_combine_whitespace = re.compile(r"\s+")

    # Esempio:
    # ['<arg1>', 'communicated', 'to', '<arg2>', 'about', '<arg4>', 'topic', 'at', '<arg3>', 'place', '(one-way', 'communication)'].
    template_words = re_combine_whitespace.sub(" ", template).strip().split()
    # Esempio:
    # ['athletes', 'communicated', 'to', 'us', 'about', '<arg>', 'topic', 'at', '<arg>', 'place', '(one-way', 'communication)'].
    predicted_words = re_combine_whitespace.sub(" ", pred["predicted"]).strip().split()
    # Esempio:
    # ['athletes', 'communicated', 'to', 'us,', 'and', 'the', 'world', 'about', '<arg>', 'topic', 'at', 'Russia', 'place', '(one-way', 'communication)'].
    gold_words = re_combine_whitespace.sub(" ", pred["gold"]).strip().split()  
    
    predicted_args = defaultdict(list) # ciascun argomento può avere più parole 
    gold_args = defaultdict(list)
    t_ptr= 0
    p_ptr= 0
    g_ptr = 0
    correct_num = 0
    missing_num = 0
    overpred_num = 0
    correct_num_arg = 0
  
    while t_ptr < len(template_words) and p_ptr < len(predicted_words) and g_ptr < len(gold_words):
        # Si verifica se il template inizia con un <arg>.
        if re.match(r"<(arg\d+)>", template_words[t_ptr]):
            # Qui si trova il numero dell'argomento. Esempio: 'arg1'.
            m = re.match(r"<(arg\d+)>", template_words[t_ptr])
            arg_num = m.group(1)
            try:
                # Qui si trova il nome dell'argomento. 
                # Esempio: 'evt043arg01communicator'.
                arg_name = ontology_dict[evt_type][arg_num]
            except KeyError:
                print(evt_type)
                exit() 

            # Nessuna predizione per l'argomento.
            if predicted_words[p_ptr] == "<arg>":
                # Nessun argomento anche nella ground truth.
                if gold_words[g_ptr] == "<arg>":
                    gold_args[arg_name].append(None)
                    correct_num_arg += 1
                    g_ptr+=1
                else:
                    # Il template gold ha un argomento, ma la predizione
                    # lo ha mancato.
                    missing_num += 1

                    gold_arg_start = g_ptr
                    while (g_ptr < len(gold_words)) and ((t_ptr == len(template_words)-1)
                            or (gold_words[g_ptr] != template_words[t_ptr+1])
                        ):
                        g_ptr+=1 
                    # gold_arg_text è il testo dell'argomento nella ground truth.
                    # Esempio: ['athletes'].
                    gold_arg_text = gold_words[gold_arg_start:g_ptr]
                    gold_arg_text = remove_det_prefix_str(" ".join(gold_arg_text))
                    gold_args[arg_name].append(gold_arg_text)

                predicted_args[arg_name].append(None)
                p_ptr += 1 
                t_ptr += 1  
            else:
                # La predizione ha trovato un argomento.
                pred_arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and ((t_ptr == len(template_words)-1)
                        or (predicted_words[p_ptr] != template_words[t_ptr+1])
                    ):
                    p_ptr += 1 
                # pred_arg_text è il testo dell'argomento predetto.
                # Esempio: ['athletes'].
                pred_arg_text = predicted_words[pred_arg_start:p_ptr]
                pred_arg_text = remove_det_prefix_str(" ".join(pred_arg_text))
                predicted_args[arg_name].append(pred_arg_text)

                # Il modello ha overpredetto l'argomento, cioè ha predetto
                # un argomento che non era specificato nella ground truth.
                if gold_words[g_ptr] == "<arg>":
                    overpred_num += 1
                    gold_args[arg_name].append(None)
                    g_ptr+=1

                else:
                    gold_arg_start = g_ptr
                    while (g_ptr < len(gold_words)) and ((t_ptr == len(template_words)-1)
                            or (gold_words[g_ptr] != template_words[t_ptr+1])
                        ):
                        g_ptr+=1 
                    gold_arg_text = gold_words[gold_arg_start:g_ptr]
                    gold_arg_text = remove_det_prefix_str(" ".join(gold_arg_text))
                    gold_args[arg_name].append(gold_arg_text)

                    if gold_arg_text == pred_arg_text:
                        correct_num += 1
                        correct_num_arg += 1
                    else:
                        overpred_num += 1

                t_ptr += 1

        else:
            t_ptr += 1 
            p_ptr += 1
            g_ptr += 1 

    res = {
        "correct_num": correct_num,
        "correct_num_arg": correct_num_arg,
        "missing_num": missing_num,
        "overpred_num": overpred_num,
        "predicted_args": predicted_args, 
        "gold_args": gold_args,
        "template": " ".join(template_words),
        "predicted": " ".join(predicted_words),
        "gold": " ".join(gold_words)
    }

    return res


def remove_det_prefix_str(
    text: str
):
    prefixes = ["the ", "The ", "an ", "An ", "a ", "A "]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text 