import json
import os
import pandas as pd
import re
import torch
import spacy
import traceback
import transformers
from tqdm.notebook import tqdm
from torchmetrics import Accuracy, F1, Precision, Recall
from transformers import (
    BartTokenizer, BertTokenizer
)
from typing import List

spacy.prefer_gpu()


def load_rams_data(
    base_path: str = "datasets/rams",
    #base_path: str = "/content/drive/MyDrive/history/datasets/rams",
    split: str = "train",
):
    docs = []
    events = []
    arguments = []
    evt2idx = {}
    idx2evt = {}
    arg2idx = {}
    idx2arg = {}

    with open(
        f"{base_path}/raw/{split}.jsonlines",
        encoding="utf-8"
    ) as reader:
        for line in reader:
            obj = json.loads(line)
            docs.append(obj)
            # I tipi di eventi sono specificati con riferimento ai relativi trigger.
            events.append(obj["evt_triggers"][0][2][0][0])
            # ent_spans specifica i tipi di argomenti per un dato evento e i relativi
            # token in uno specifico documento: 
            # "ent_spans": [[27, 27, [["evt043arg01communicator", 1.0]]], [48, 48, [["evt043arg03place", 1.0]]], [32, 36, [["evt043arg02recipient", 1.0]]]]
            for ent in obj["ent_spans"]:
                arguments.append(ent[2][0][0])

    events = sorted(set(events))
    arguments = sorted (set(arguments))

    # Viene aggiunto un indice anche per la mancanza di eventi e di argomenti.
    evt2idx = {evt: idx for idx, evt in enumerate(["no_evt"] + events)}
    idx2evt = {idx: evt for evt, idx in evt2idx.items()}
    arg2idx = {arg: idx for idx, arg in enumerate(["no_arg"] + arguments)}
    idx2arg = {idx: arg for arg, idx in arg2idx.items()}

    dicts = {
        "evt2idx": evt2idx,
        "idx2evt": idx2evt,
        "arg2idx": arg2idx,
        "idx2arg": idx2arg
    }

    return docs, dicts


def get_rams_data_dict(
    docs: List[dict],
    tokenizer,
    split: str,
    span_max_length: int = 3,
    #base_path: str = "datasets/rams/preprocessed",
    base_path: str = "/content/drive/MyDrive/history/datasets/rams/preprocessed",
    write: bool = True,
    map_dicts: dict = None
):
    evt2idx = map_dicts["evt2idx"]
    arg2idx = map_dicts["arg2idx"]

    spans = []
    spans_trg_labels = []
    spans_arg_labels = []
    tokens = []
    tokens_ids = []
    attention_masks = []
    triggers = []
    arguments = []
    doc_keys = []
    span_mappers = []

    args_ids = []
    args_masks = []
    args_dec_ids = []
    args_dec_masks = []

    evt_ids = []
    evt_masks = []
    evt_dec_ids = []
    evt_dec_masks = []

    # Per i template degli eventi viene utilizzato BART.
    argument_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    argument_tokenizer.add_tokens([" <arg>", " <trg>", " <evt>"])

    for doc in tqdm(
        docs,
        desc="Processing document",
        leave=False
    ):
        # Questa è la chiave (identificatore) del singolo documento.
        doc_keys.append(doc["doc_key"])
        # Questi sono i token del documento, cioè le parole del testo.
        doc_tokens = sum(doc["sentences"], [])
        
        # Qui vengono corretti alcuni token.
        fixed_doc_tokens = []
        for tok in doc_tokens:
            if "\u200b" in tok or "\u200e" in tok:
                if len(tok) == 1:
                    tok = tok.replace("\u200b", "-")
                    tok = tok.replace("\u200e", "-")
                else:
                    tok = tok.replace("\u200b", "")
                    tok = tok.replace("\u200e", "")
            tok = tok.replace("\u201c", "\"").replace("\u201d", "\"") \
                    .replace("\u2019", "'").replace("\u2014", "⁠—").replace("\u2060", "") \
                    .replace("\xad", "").replace("\x9a", "").replace("\x7f", "-") \
                    .replace("\x93", "\"").replace("\x94", "\"").replace("\x96", "–") \
                    .replace("\x92", "'").replace("☰", "-").replace("📸", "-") \
                    .replace("▪", "-").replace("👎🏻", "-").replace("�", "-") \
                    .replace("\ufffd", "-").replace("\x9d", "")
            fixed_doc_tokens.append(tok)

        doc_tokens = fixed_doc_tokens
        tokens.append(doc_tokens)

        # ========================
        # EVENT/TRIGGER EXTRACTION
        # ========================

        # Il tokenizer output è dato dagli encoding: Encoding(num_tokens=512, 
        # attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]).
        # Qui viene usato il padding per avere sempre 512 token.
        tokenizer_output = tokenizer(
            doc_tokens,
            is_split_into_words=True,
            padding="max_length", 
            truncation=True,
            return_offsets_mapping=True
        )
        tokens_ids.append(tokenizer_output.input_ids)
        attention_masks.append(tokenizer_output.attention_mask)

        token2idx, idx2token, last_token_idx = get_mapper(
            tokenizer_output.offset_mapping,
            tokenizer_output.input_ids,
            doc_tokens,
            tokenizer,
            doc["evt_triggers"],
            doc["ent_spans"]
        )

        # Dato che stiamo usando i doc_tokens e non l'output del tokenizer,
        # prendiamo gli span solo fino alla lunghezza della frase originale.
        # Non prendiamo gli span dei token di padding.
        doc_spans = get_spans(
            doc_tokens, 
            last_token_idx,
            span_max_length=span_max_length
        )
        span_mappers.append(doc_spans)

        # doc_spans contiene gli indici di inizio e di fine degli span
        # nei token originali.  
        doc_spans = [
            [token2idx[span[0]][0], token2idx[span[1]][1], span[2]]
            for span in doc_spans
        ]
        spans.append(doc_spans)

        doc_trigger = doc["evt_triggers"]
        # Recuperiamo gli indici del trigger nella frase originale e li
        # trasformiamo negli indici restituiti dal tokenizer di BERT.
        # Per l'inizio del trigger prendiamo il primo indice, che è l'indice
        # della prima sotto-parola se il token è stato diviso. Per la fine
        # del trigger prendiamo il secondo indice, che è l'indice dell'ultima
        # sotto-parola.
        trigger_start = token2idx[doc_trigger[0][0]][0]
        trigger_end = token2idx[doc_trigger[0][1]][1]
        trigger_name = doc_trigger[0][2][0][0]
        # trigger_label è il numero dell'evento nell'elenco degli eventi.
        trigger_label = evt2idx[trigger_name]
            
        triggers.append([trigger_start, trigger_end, trigger_label])

        # Assegniamo una label a ogni span. Se lo span contiene un trigger,
        # allora gli assegniamo la label del trigger, altrimenti gli assegniamo
        # la label 0 corrispondente a "no_evt".
        doc_spans_trg_labels = []
        for span in doc_spans:
            if (span[0] == trigger_start) and (span[1] == trigger_end):
                doc_spans_trg_labels.append(trigger_label)
            else:
                doc_spans_trg_labels.append(evt2idx["no_evt"])
        spans_trg_labels.append(doc_spans_trg_labels)

        # Qui viene fatto per gli argomenti quello che prima è stato fatto per
        # il trigger. Quindi per ogni argomento vengono inidicati gli indici di
        # inizio e di fine rispetto ai token di BERT. E a ogni argomento viene
        # attribuita la relativa label estratta dal dizionario.
        # Esempio:
        # doc["ent_spans"]:  [[27, 27, [['evt043arg01communicator', 1.0]]], [48, 48, [['evt043arg03place', 1.0]]], [32, 36, [['evt043arg02recipient', 1.0]]]]
        # doc_arguments:  [[[31, 31], [31, 31], 118], [[53, 53], [53, 53], 120], [[37, 37], [41, 41], 119]]
        doc_arguments = []
        for argument in doc["ent_spans"]:
            arg_start = token2idx[argument[0]]
            arg_end = token2idx[argument[1]]
            arg_label = arg2idx[argument[2][0][0]]
            doc_arguments.append([arg_start, arg_end, arg_label])
        arguments.append(doc_arguments)

        # Qui assegniamo una label a ogni argomento, come è stato fatto anche
        # per il trigger.
        doc_spans_arg_labels = []
        '''
        for span in doc_spans:
            if (span[0] == trigger_start) and (span[1] == trigger_end):
                doc_spans_arg_labels.append(trigger_label)
            else:
                doc_spans_arg_labels.append(evt2idx["no_evt"])
        '''
        for span in doc_spans:
            label = 0
            for argument in doc_arguments:
                if (span[0] >= argument[0][0]) and (span[1] <= argument[1][1]):
                    label = argument[2]
            doc_spans_arg_labels.append(label)
                    
        spans_arg_labels.append(doc_spans_arg_labels)

        # ===================
        # ARGUMENT EXTRACTION
        # ===================

        doc_trigger = doc["evt_triggers"]
        trigger_start = doc_trigger[0][0]
        trigger_end =doc_trigger[0][1]
        trigger_name = doc_trigger[0][2][0][0]

        ontology_dict = load_ontology()
        # Qui viene recuperato il template dello specifico evento.
        template = ontology_dict[trigger_name.replace("n/a", "unspecified")]["template"]

        # Qui viene trasformato il template in un template di token ottenuti con BART.
        template_in = template2tokens(template, argument_tokenizer)

        # Qui viene riempito il template dell'evento con gli argomenti corretti.
        for argument in doc["ent_spans"]:
            arg_start = argument[0]
            arg_end = argument[1]
            arg_name = argument[2][0][0]
            arg_num = ontology_dict[
                trigger_name.replace("n/a", "unspecified")
            ][arg_name]
            arg_text = " ".join(doc_tokens[arg_start : arg_end + 1])
            template = re.sub(f"<{arg_num}>", arg_text, template)

        # Qui viene trasformato il template riempito in un template di token ottenuti con BART.
        template_out = template2tokens(template, argument_tokenizer)
        
        # Qui viene aggiunto un prefisso a ogni token della frase (non più del template)
        # fino al trigger.
        prefix = argument_tokenizer.tokenize(
            " ".join(doc_tokens[:trigger_start]),
            add_prefix_space=True
        )
        # Qui viene aggiunto un prefisso al trigger della frase (non più del template).
        trg = argument_tokenizer.tokenize(
            " ".join(doc_tokens[trigger_start:trigger_end+1]),
            add_prefix_space=True
        )
        # Qui viene aggiunto un prefisso a ogni token della frase (non più del template)
        # successivo al trigger.
        suffix = argument_tokenizer.tokenize(
            " ".join(doc_tokens[trigger_end+1:]),
            add_prefix_space=True
        )
        # Questa è la concatenazione dei token con prefisso. Prima e dopo il trigger
        # vengono aggiunti dei tag che specificano il trigger.
        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix
        
        # Qui vengono ottenuti gli encodings forniti da BART per il template 
        # non riempito.
        arg_in = argument_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424, # max_length specifica la lunghezza del testo tokenizzato
            truncation="only_second", # questo tronca solo la seconda frase di una coppia, se è fornita una coppia
            padding="max_length"
        )

        # Qui vengono ottenuti gli encodings forniti da BART per il template 
        # riempito.
        arg_out = argument_tokenizer.encode_plus(
            template_out, 
            add_special_tokens=True,
            add_prefix_space=True, 
            max_length=72,
            truncation=True, # questo tronca token per token
            padding="max_length"
        )
        
        args_ids.append(arg_in["input_ids"])
        args_masks.append(arg_in["attention_mask"])
        args_dec_ids.append(arg_out["input_ids"])
        args_dec_masks.append(arg_out["attention_mask"])

        #=================
        # EVENT GENERATION
        #=================

        evt_template_in = "This document is about <evt>"
    
        evt2sent, sent2evt, _, _ = get_event_names_dict()

        # Trigger dell'evento. Esempio:
        # [[31, 31, [['contact.prevarication.broadcast', 1.0]]]]
        doc_trigger = doc["evt_triggers"]
        
        # Nome del trigger dell'evento. Esempio:
        # 'contact.prevarication.broadcast'
        trigger_name = doc_trigger[0][2][0][0].replace("n/a", "unspecified")

        # Descrizione dell'evento ricavata dal dizionario evt2sent. Esempio:
        # 'a broadcast about a prevarication'
        evt_sent = evt2sent[trigger_name]
        
        evt_sent = "This document is about " + evt_sent

        # Viene tokenizzata con BART la frase evt_sent relativa all'evento. Esempio:
        # ['ĠThis', 'Ġdocument', 'Ġis', 'Ġabout', 'Ġa', 'Ġbroadcast', 'Ġabout', 'Ġa', 'Ġpre', 'var', 'ication']
        template_tokens = []
        for word in evt_sent.split(" "):
            template_tokens.extend(
                argument_tokenizer.tokenize(
                    word,
                    add_prefix_space=True 
                )
            )
        evt_template_out = template_tokens

        # Qui viene nuovamente tokenizzata con BART la frase originale del documento.
        evt_context = argument_tokenizer.tokenize(
            " ".join(doc_tokens),
            add_prefix_space=True
        )

        # Qui vengono ottenuti tramite BART gli encoding di evt_template_in ("This document is about <evt>")
        # e del testo originale. I due testi vengono separati con [SEP].
        evt_in = argument_tokenizer.encode_plus(
            evt_template_in, 
            evt_context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        # Qui vengono ottenuti tramite BART gli encoding di evt_template_out ("This document is about <evt>").
        # Risultato: ['<s>','ĠThis','Ġdocument','Ġis','Ġabout','Ġa','Ġbroadcast','Ġabout','Ġa','Ġpre','var','ication','</s>','<pad>',...]
        evt_out = argument_tokenizer.encode_plus(
            evt_template_out, 
            add_special_tokens=True,
            add_prefix_space=True, 
            max_length=72,
            truncation=True,
            padding="max_length"
        )

        evt_ids.append(evt_in["input_ids"])
        evt_masks.append(evt_in["attention_mask"])
        evt_dec_ids.append(evt_out["input_ids"])
        evt_dec_masks.append(evt_out["attention_mask"])

    result = {
        "spans": spans,
        "spans_trg_labels": spans_trg_labels, 
        "spans_arg_labels": spans_arg_labels, 
        "tokens": tokens,
        "tokens_ids": tokens_ids,
        "attention_masks": attention_masks, 
        "triggers": triggers,
        "arguments": arguments, 
        "map_dicts": map_dicts,
        "args_ids": args_ids,
        "args_masks": args_masks,
        "args_dec_ids": args_dec_ids,
        "args_dec_masks": args_dec_masks,
        "doc_keys": doc_keys,
        "span_mappers": span_mappers,
        "evt_ids": evt_ids,
        "evt_masks": evt_masks,
        "evt_dec_ids": evt_dec_ids,
        "evt_dec_masks": evt_dec_masks
    }

    if write:
        os.makedirs(base_path, exist_ok=True)

        with open(
            f"{base_path}/{split}.json", "w"
        ) as f_out:
            json.dump(result, f_out)

    return result


def get_mapper(
    offset_mapping, 
    input_ids, 
    tokens: List[str],
    tokenizer,
    doc_evts,
    doc_args
):
    """
    Restituisce un dizionario che mappa ogni token della frase originale
    agli indici corrispondenti nella frase tokenizzata dal BERT tokenizer,
    che può dividere una parola in sotto-parole.
    Nel dizionario per ogni token si ottiene una tupla che corrisponde
    agli inidici di inizio e di fine del token della frase tokenizzata.
    Restituisce anche un dizionario per l'operazione inversa.
    """
    token2idx = {}
    idx2token = {}
    idx = -1

    # off_num è un indice da 0 al numero di token.
    # offset è composto da coppie di valori. Tipicamente gli offset [[(0,0),(0,3),(3,4)...] ] 
    # contengono gli indici di inizio e di fine per ogni parola nella frase di input.
    # Tuttavia qui stiamo passando in input già la lista di token. Quindi gli offset
    # indicano il numero di caratteri di ciascun token. Se il primo valore della tupla
    # è diverso da 0, allora il token è stato suddiviso in sotto-parole. 
    # L'offset (0,0) indica token speciali come CLS. 
    for off_num, offsets in enumerate(offset_mapping):
        # Saltiamo i token speciali come CLS, SEP e PAD in quanto
        # non presenti nella frase originale.
        if offsets[1] == 0:
            continue

        # Il token è stato suddiviso in sotto-parole. L'indice di inizio
        # del token originale rimane lo stesso, mentre occorre aggiornare
        # l'indice di fine.
        if offsets[0] != 0:
            token2idx[idx][1] = off_num
            
        # Token normali.
        else:
            idx = idx + 1
            token2idx[idx] = [off_num, off_num]

    start2token = {start: tok for tok, [start, end] in token2idx.items()}
    end2token = {end: tok for tok, [start, end] in token2idx.items()}
    idx2token = {"start": start2token,"end": end2token}

    last_token_idx = None

    '''
    for i, token in enumerate(tokens):
        if i >= len(token2idx):
            last_token_idx = i - 1
    '''
    
    # Sanity check.
    for i, token in enumerate(tokens):
        if i >= len(token2idx):
            print(
                "The original sentence was", len(tokens),
                "tokens long and was tokenized by BERT into",
                f"{len(input_ids)} tokens.", "The sentence was",
                f"truncated at index {i}. The token2idx dict",
                f"had length {len(token2idx)}." 
            )
            
            last_token_idx = i - 1

            # Controlla che tutti gli eventi e gli argomenti siano
            # nella frase troncata.
            for evt in doc_evts:
                evt_start = evt[0]
                evt_end = evt[1]
                if evt_start >= i or evt_end >= i:
                    raise Exception(
                        f"Event {evt} was in a truncated"
                        "part of a sentence."
                    )

            for arg in doc_args:
                arg_start = arg[0]
                arg_end = arg[1]
                if arg_start >= i or arg_end >= i:
                    raise Exception(
                        f"Argument {arg} was in a truncated"
                        "part of a sentence."
                    )

            print("doc_evts", doc_evts)
            print("doc_args", doc_args)
            break
        try:
            ids = input_ids[token2idx[i][0]:token2idx[i][1] + 1]
        except Exception as ex:
            traceback.print_exc()
            print("token number", i)
            print("input ids", input_ids)
            print("token2idx", token2idx)
            print("tokens", tokens)
            print("tokens len", len(tokens))
            print("input ids len", len(input_ids))
            print("token2idx len", len(token2idx))
            print("decoded ids", tokenizer.decode(input_ids))
            print("offset_mapping", offset_mapping)
            print("len offset_mapping", len(offset_mapping))

        decoded_token = tokenizer.decode(ids)

        if (token != decoded_token) and (decoded_token in token):
            # E' possibile che l'ultimo token sia quello suddiviso
            # in sotto-parole e alcune sotto-parole possono aver
            # superato il limite di 512 token.
            print("Last token was splitted: ", token, decoded_token)
            continue
        try:
            assert token == decoded_token.replace(" ", ""),(token, decoded_token)
        except AssertionError as ae:
            print("token number", i)
            print("input ids", input_ids)
            print("token2idx", token2idx)
            print("tokens", tokens)
            print("tokens len", len(tokens))
            print("input ids len", len(input_ids))
            print("token2idx len", len(token2idx))
            print("decoded ids", tokenizer.decode(input_ids))
            print("offset_mapping", offset_mapping)
            print("len offset_mapping", len(offset_mapping))
            raise ae

    return token2idx, idx2token, last_token_idx


def get_spans(
    tokens: List[str], 
    last_token_idx: int,
    span_max_length: int = 3
):
    # Usiamo last_token_idx per il caso in cui la frase originale
    # è stata troncata dal tokenizer di BERT. In questo caso occorre
    # fermare la creazione degli span a un indice precedente rispetto
    # alla lunghezza della frase originale.
    if last_token_idx is None:
        last_token_idx = len(tokens)

    # Questo è l'indice dell'ultimo token (parola) nella frase originale.
    last_index = min(len(tokens), last_token_idx)

    # Suddividiamo i token in span. Dato che lo span_max_length è 3, vengono
    # creati blocchi di 3 token in successione.
    spans = []
    for i in range(last_index):
        for j in range(i, min(last_index, i + span_max_length)):
            spans.append((i, j, j - i + 1))
    return spans


def load_ontology(
    #base_path: str = "datasets"
    base_path: str = "/content/drive/MyDrive/history/datasets"
):
    # L'ontologia specifica: il tipo di evento, il template e gli argomenti.
    # L'ontologia contiene 149 eventi.
    ontology = pd.read_csv(f"/content/drive/MyDrive/history/datasets/aida_ontology_cleaned.csv")
    
    # ontology_dict è semplicemente l'ontologia tradotta in un dizionario.
    # Le chiavi del dizionario sono le tipologie di eventi. I valori sono il 
    # template e gli argomenti. Anche i valori sono espressi come dizionario.
    ontology_dict = dict()
    for event_type in ontology["event_type"]:
        ontology_dict[event_type] = dict()
        
        # Qui viene specificata la chiave, ovvero il tipo di evento.
        row = ontology[ontology["event_type"] == event_type]
        # Qui viene aggiunto il template come valore dell'evento.
        ontology_dict[event_type]["template"] = row["template"].values[0]
        # Qui vengono aggiunti anche gli argomenti. Per gli argomenti viene
        # specificato sia il tipo che il numero dell'argomento nel template:
        # 'evt152arg01mechanicalartifact ': 'arg1',
        # 'arg1': 'evt152arg01mechanicalartifact ',
        for arg_num, arg in zip(
            row.iloc[0,2:].index, 
            row.iloc[0,2:]
        ):
            if isinstance(arg, str):
                ontology_dict[event_type][arg] = arg_num
                ontology_dict[event_type][arg_num] = arg

    return ontology_dict


def template2tokens(
    template: str,
    tokenizer: transformers.BartTokenizerFast
):
    # Qui il template viene ricostruito come sequenza di parole/token:
    # ['<arg>', 'communicated', 'to', '<arg>', 'about', '<arg>', 'topic', 'at', '<arg>', 'place', '(one-way', 'communication)']
    template = re.sub(r"<arg\d>", "<arg>", template).split(" ")

    # Qui le parole del template vengono tokenizzate tramite BART. Inoltre
    # viene aggiunto un prefisso che per gli argomenti è uno spazio.
    template_tokens = []
    for word in template:
        template_tokens.extend(
            tokenizer.tokenize(
                word,
                add_prefix_space=True 
            )
        )

    return template_tokens


def get_event_names_dict():

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_tokenizer.add_tokens([" <arg>", " <trg>", " <evt>"])

    evt2sent = {
        'artifactexistence.artifactfailure.mechanicalfailure': "artifact mechanical failure",
        'artifactexistence.damagedestroy.unspecified': "artifact damage or destruction",
        'artifactexistence.damagedestroy.damage': "artifact damage",
        'artifactexistence.damagedestroy.destroy': "artifact destruction",
        'artifactexistence.shortage.shortage': "artifact shortage",
        'conflict.attack.unspecified': "attack in a conflict",
        'conflict.attack.airstrikemissilestrike': "air strike or missile strike in a conflict",
        'conflict.attack.biologicalchemicalpoisonattack': "biological or chemical or poison attack in a conflict",
        'conflict.attack.bombing': "bombing in a conflict",
        'conflict.attack.firearmattack': "firearm attack in a conflict",
        'conflict.attack.hanging': "hanging in a conflict",
        'conflict.attack.invade': "invasion in a conflict",
        'conflict.attack.selfdirectedbattle': "self directed battle in a conflict",
        'conflict.attack.setfire': "setting fire in a conflict",
        'conflict.attack.stabbing': "stabbing in a conflict",
        'conflict.attack.stealrobhijack': "steal or rob or hijack in a conflict",
        'conflict.attack.strangling': "strangling in a conflict",
        'conflict.coup.coup': "coup in a conflict",
        'conflict.demonstrate.unspecified': "demonstration in a conflict",
        'conflict.demonstrate.marchprotestpoliticalgathering': "march or protest or political gathering in a conflict",
        'conflict.yield.unspecified': "yielding in a conflict",
        'conflict.yield.retreat': "retreat in a conflict",
        'conflict.yield.surrender': "surrender in a conflict",
        'contact.collaborate.unspecified': "collaboration with a contact",
        'contact.collaborate.correspondence': "correspondence in a collaboration",
        'contact.collaborate.meet': "meeting in a collaboration",
        'contact.commandorder.unspecified': "command or order",
        'contact.commandorder.broadcast': "a broadcast about a command or order",
        'contact.commandorder.correspondence': "correspondence about a command or order",
        'contact.commandorder.meet': "a meeting about a command or order",
        'contact.commitmentpromiseexpressintent.unspecified': "commitment or promise or intent expression",
        'contact.commitmentpromiseexpressintent.broadcast': "a broadcast about commitment or promise or intent expression",
        'contact.commitmentpromiseexpressintent.correspondence': "correspondence about commitment or promise or intent expression",
        'contact.commitmentpromiseexpressintent.meet': "meeting for commitment or promise or intent expression",
        'contact.discussion.unspecified': "discussion with a contact",
        'contact.discussion.correspondence': "correspondence for a discussion",
        'contact.discussion.meet': "meeting for a discussion",
        'contact.funeralvigil.unspecified': "a funeral or vigil",
        'contact.funeralvigil.meet': "meeting for a funeral or vigil",
        'contact.mediastatement.unspecified': "a media statement",
        'contact.mediastatement.broadcast': "the broadcast of a media statement",
        'contact.negotiate.unspecified': "a negotiation",
        'contact.negotiate.correspondence': "correspondence for a negotiation",
        'contact.negotiate.meet': "meeting for a negotiation",
        'contact.prevarication.unspecified': "prevarication",
        'contact.prevarication.broadcast': "a broadcast about a prevarication",
        'contact.prevarication.correspondence': "correspondence about a prevarication",
        'contact.prevarication.meet': "a meeting about a prevarication",
        'contact.publicstatementinperson.unspecified': "a public statement in person",
        'contact.publicstatementinperson.broadcast': "the broadcast of a public statement in person",
        'contact.requestadvise.unspecified': "a request or advise",
        'contact.requestadvise.broadcast': "the broadcast of a request or advise",
        'contact.requestadvise.correspondence': "correspondence about a request or advise",
        'contact.requestadvise.meet': "a request or advise in a meeting",
        'contact.threatencoerce.unspecified': "a threat or coercion",
        'contact.threatencoerce.broadcast': "the broadact of a threat or coercion",
        'contact.threatencoerce.correspondence': "correspondence about a threat or coercion",
        'contact.threatencoerce.meet': "threat or coercion in a meeting",
        'disaster.accidentcrash.accidentcrash': "an accident or crash",
        'disaster.diseaseoutbreak.diseaseoutbreak': "a disease outbreak",
        'disaster.fireexplosion.fireexplosion': "a fire explosion",
        'genericcrime.genericcrime.genericcrime': "a generic crime",
        'government.agreements.unspecified': "government agreements",
        'government.agreements.acceptagreementcontractceasefire': "the acceptance of an agreement or a cease fire",
        'government.agreements.rejectnullifyagreementcontractceasefire': "the rejection of an agreement or cease fire",
        'government.agreements.violateagreement': "the violation of an agreement",
        'government.convene.convene': "a government convention",
        'government.formation.unspecified': "a government formation",
        'government.formation.mergegpe': "a government merge",
        'government.formation.startgpe': "a government start",
        'government.legislate.legislate': "legislation",
        'government.spy.spy': "spionage",
        'government.vote.unspecified': "a government vote",
        'government.vote.castvote': "a government vote casted",
        'government.vote.violationspreventvote': "a government vote violation or prevention",
        'inspection.sensoryobserve.unspecified': "a sensory inspection",
        'inspection.sensoryobserve.inspectpeopleorganization': "an inspection of people",
        'inspection.sensoryobserve.monitorelection': "the monitoring of an election",
        'inspection.sensoryobserve.physicalinvestigateinspect': "a physical investigation or inspection",
        'inspection.targetaimat.targetaimat': "a target aim in an inspection",
        'justice.arrestjaildetain.arrestjaildetain': "arrest or jail or detention",
        'justice.initiatejudicialprocess.unspecified': "a judicial process",
        'justice.initiatejudicialprocess.chargeindict': "charge indiction in a judicial process",
        'justice.initiatejudicialprocess.trialhearing': "a trial hearing in a judicial process",
        'justice.investigate.unspecified': "an investigation",
        'justice.investigate.investigatecrime': "a crime investigation",
        'justice.judicialconsequences.unspecified': "judicial consequences",
        'justice.judicialconsequences.convict': "conviction as a judicial consequence",
        'justice.judicialconsequences.execute': "execution as a judicial consequence",
        'justice.judicialconsequences.extradite': "extradition as a judicial consequence",
        'life.die.unspecified': "death",
        'life.die.deathcausedbyviolentevents': "death caused by violent events",
        'life.die.nonviolentdeath': "non violent death",
        'life.injure.unspecified': "an injury",
        'life.injure.illnessdegradationhungerthirst': "an injury caused hunger or thirst",
        'life.injure.illnessdegradationphysical': "an injury caused by physical degradation",
        'life.injure.illnessdegredationsickness': "an injury caused by sickness",
        'life.injure.injurycausedbyviolentevents': "an injury caused by violent events",
        'manufacture.artifact.unspecified': "the manufacturing of an artifact",
        'manufacture.artifact.build': "the building of an artifact",
        'manufacture.artifact.createintellectualproperty': "the creation of an intellectual property",
        'manufacture.artifact.createmanufacture': "the creation of a manufacture",
        'medical.intervention.intervention': "a medical intervention",
        'movement.transportartifact.unspecified': "the transportation of an artifact",
        'movement.transportartifact.bringcarryunload': "carrying or unloading an artifact",
        'movement.transportartifact.disperseseparate': "an artifact dispersed or separated during transportation",
        'movement.transportartifact.fall': "an artifact fallen during transportation",
        'movement.transportartifact.grantentry': "an entry grant for the transportation of an artifact",
        'movement.transportartifact.hide': "an artifact hidden during transportation",
        'movement.transportartifact.lossofcontrol': "the loss of control of an artifact during transportation",
        'movement.transportartifact.nonviolentthrowlaunch': "a non violent throw or launch of an artifact",
        'movement.transportartifact.prevententry': "an entry prevention for the transportation of an artifact",
        'movement.transportartifact.preventexit': "an exit prevention for the transportation of an artifact",
        'movement.transportartifact.receiveimport': "an artifact received or imported",
        'movement.transportartifact.sendsupplyexport': "sending supply or exporting an artifact",
        'movement.transportartifact.smuggleextract': "smuggling or extracting an artifact",
        'movement.transportperson.unspecified': "the transportation of a person",
        'movement.transportperson.bringcarryunload': "carrying or unloading a person",
        'movement.transportperson.disperseseparate': "a person dispersed or separated during transportation",
        'movement.transportperson.evacuationrescue': "the evacuation or rescue of a person",
        'movement.transportperson.fall': "a person fallen during transportation",
        'movement.transportperson.grantentryasylum': "a grant of entry or asylum to a person",
        'movement.transportperson.hide': "a person hidden during transportation",
        'movement.transportperson.prevententry': "an entry prevention for the transportation of a person",
        'movement.transportperson.preventexit': "an exit prevention for the transportation of a person",
        'movement.transportperson.selfmotion': "a person in self motion",
        'movement.transportperson.smuggleextract': "smuggling or extracting a person",
        'personnel.elect.unspecified': "election of a person",
        'personnel.elect.winelection': "an election won by a person",
        'personnel.endposition.unspecified': "the end of a position",
        'personnel.endposition.firinglayoff': "a person who was fired or laid off",
        'personnel.endposition.quitretire': "a person who quit a position or retired",
        'personnel.startposition.unspecified': "the start of a position",
        'personnel.startposition.hiring': "a person being hired",
        'transaction.transaction.unspecified': "a transaction",
        'transaction.transaction.embargosanction': "an embargo or sanction in a transaction",
        'transaction.transaction.giftgrantprovideaid': "a gift granted or aid provided in a transaction",
        'transaction.transfermoney.unspecified': "a transfer of money",
        'transaction.transfermoney.borrowlend': "money borrowed or lent",
        'transaction.transfermoney.embargosanction': "an embargo or sanction in a transfer of money",
        'transaction.transfermoney.giftgrantprovideaid': "a gift granted or aid provided in a transfer of money",
        'transaction.transfermoney.payforservice': "payment for a service",
        'transaction.transfermoney.purchase': "a purchase with a transfer of money",
        'transaction.transferownership.unspecified': "a transfer of ownership",
        'transaction.transferownership.borrowlend': "ownership borrowed or lent",
        'transaction.transferownership.embargosanction': "an embargo or sanction in a transfer of ownership",
        'transaction.transferownership.giftgrantprovideaid': "a gift granted or aid provided in a transfer of ownership",
        'transaction.transferownership.purchase': "a purchase with a transfer of ownership",
        'transaction.transaction.transfercontrol': "a transfer of control in a transaction"
    }

    # Questo è semplicemente il reciproco di evt2sent.
    sent2evt = {v: k for k, v in evt2sent.items()}
    
    # Qui viene costruito il vocabolario delle parole che compaiono nella descrizione
    # (sotto forma di frase) di un evento.
    vocab = set()
    for sent in sent2evt.keys():
        for word in sent.split(" "):
            vocab.add(word)

    # Qui vengono aggiunte le parole: "this", "document", "is" e "about".
    prompt = "This document is about"
    for word in prompt.split(" "):
        vocab.add(word)

    # Trasforma il vocabolario da insieme a lista.
    vocab = list(vocab)

    # Viene creata una singola stringa con le parole del vocabolario separate
    # da uno spazio.
    v = " ".join(vocab)

    out = bart_tokenizer.encode_plus(
        v, 
        add_special_tokens=True,
        add_prefix_space=True,
        max_length=424,
        padding="max_length"
    )

    return evt2sent, sent2evt, vocab, torch.tensor(out["input_ids"])

