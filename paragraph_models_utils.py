import re
import spacy
import transformers
import numpy as np
from cache_decorator import Cache
from tqdm.notebook import tqdm
from typing import List, Tuple, Union

spacy.prefer_gpu()

@Cache(
    cache_path=[
        "cache/h_events/{function_name}/{limit}/texts_{_hash}.pkl",
        "cache/h_events/{function_name}/{limit}/tags_{_hash}.pkl",
        "cache/h_events/{function_name}/{limit}/labels_{_hash}.pkl"
    ],
    args_to_ignore=["dataset", "nlp"]
)
def load_data(
    dataset: dict,
    nlp: spacy.language.Language,
    limit: int = None
):
    texts = []
    tags = []
    labels = []

    if limit == None:
        limit = len(dataset["paragraphs"])

    # Qui vengono usate le entità in ogni paragrafo per creare
    # dei tag per il paragrafo.
    for paragraph in tqdm(
        dataset["paragraphs"][:limit],
        desc="Creating tags",
        leave=False
    ):
        text_tokens, text_tokens_tags = tag_paragraph(
            paragraph, nlp
        )
        texts.extend(text_tokens)
        tags.extend(text_tokens_tags)
        for _ in range(len(text_tokens)):
            labels.append(paragraph["historical"])

    # Texts è la sequenza di token di ogni paragrafo.
    texts = np.array(texts, dtype=object)
    # Tags è la sequenza di tag (nel formato BIO) di ogni paragrafo.
    tags = np.array(tags, dtype=object)
    # Labels specifica se i paragrafi sono storici o meno in base alle proprietà
    # Wikidata della pagina a cui appartengono i paragrafi.
    # Label 0: not historical
    # Label 1: historical
    labels = np.array(labels, dtype=object)

    return texts, tags, labels


def tag_paragraph(
    paragraph_dict: dict,
    nlp: spacy.language.Language,
):
    # Il testo è la sequenza di token del paragrafo.
    text = paragraph_dict["clean_content"].strip()
    # I token sono stati ottenuti tramite SpaCy. Ogni token viene memorizzato insieme
    # alla sua parte del discorso (PoS). Esempio: [('Dark', 'PROPN'),('Ages', 'PROPN'),('is', 'AUX')].
    text_tokens = [(token.text, token.pos_) for token in nlp(text)]
    
    # Ottengo i tag.
    # Il tag "O" significa che il token corrispondente non è ancora stato etichettato.
    tags = ["O"] * len(text_tokens)
    entities = paragraph_dict["entities"]
    # Per ogni entità vengono specificati i tag dei token corrispondenti.
    # In particolare il tag del primo token è "B-hist" se l'entità è storica e "B-not-hist" se non lo è.
    # I tag dei token successivi della stessa entità vengono poi impostati come "I-hist" e "I-not-hist".
    for entity in entities:
        token_offset = entity["token_offset"]
        tokens_length = entity["tokens_length"]
        historical = entity["historical"]
        suffix = "hist" if historical else "not-hist"
        for i in range(tokens_length):
            if i == 0:
                tags[token_offset + i] = "B-" + suffix
            else:
                tags[token_offset + i] = "I-" + suffix

    # Suddivido il paragrafo in frasi più corte di un certo numero di token
    # (256) in modo che siano compatibili con la lunghezza massima dell'input
    # di BERT pre-addestrato (512). 
    offsets = get_sentences_offsets(text_tokens)

    tokens_results = []
    tags_results = []
    # Gli offset sono rappresentati da coppie (x,y) dove x è l'inizio e y è la
    # fine di una sequenza di token.
    for min_off, max_off in offsets:
        tokens_results.append(
            [token for token, pos in text_tokens[min_off:max_off]]
        )
        tags_results.append(tags[min_off:max_off])

    return tokens_results, tags_results


def get_sentences_offsets(
    text_tokens: List[Tuple],
    threshold: int = 256
):
    """
    BERT consente un massimo di 512 token in input. Quindi occorre suddividere
    i paragrafi più lunghi di 512 token in paragrafi più piccoli. Usiamo il
    punto (".") per ottenere i confini delle frasi. Poi prendiamo le sequenze
    più lunghe che contengano un numero di token più piccolo di una soglia 
    prefissata. Questa soglia è più bassa di 512 in quanto il tokenizer di 
    BERT divide alcune parole in sotto-parole, incrementando così il numero
    totale di token.
    """
    
    offset = 0
    # Inizialmente il max_offset è pari a 256 o alla lunghezza del paragrafo
    # se questa è più corta di 256.
    max_offset = min(threshold, len(text_tokens))
    length_left = len(text_tokens)

    offsets = []

    while length_left > 0:
        for i in reversed(range(max_offset)):
            token, pos = text_tokens[i]

            # Salviamo una sequenza quando troviamo un punto di punteggiatura,
            # o la sequenza è lunga quanto il resto del paragrafo o non abbiamo
            # trovato un punto in tutta la sezione del paragrafo.
            if ((token == ".") and (pos == "PUNCT")) or (i == (len(text_tokens) - 1)) \
                or (i == offset):
                new_offset = i + 1
                offsets.append((offset, new_offset))
                max_offset = min(new_offset + threshold, len(text_tokens))
                offset = new_offset

                length_left = len(text_tokens) - offset

                break
    # Si ottengono offset come (0,42), dove il primo valore rappresenta l'inizio
    # di una sequenza e il secondo ne rappresenta la fine.
    return offsets


def preprocess_data(
    texts: List[str],
    tags: List[str],
    labels: List[int],
    tokenizer: transformers.BertTokenizerFast,
    ignore_other: bool = True,
    padding: Union[str, bool] = True
):
    # Creiamo i dizionari tag2idx e idx2tag per convertire i tag in indici e
    # viceversa.
    # unique_tags:  ['-', 'B', 'I', 'O', 'h', 'i', 'n', 'o', 's', 't']
    unique_tags = sorted(set(tag for text_tags in tags for tag in text_tags))
    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    # Qui viene aggiunto il tag MASK con valore -100.
    tag2idx["MASK"] = -100
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    if ignore_other:
        # Qui anche il tag O viene impostato a -100.
        del idx2tag[tag2idx["O"]]
        tag2idx["O"] = -100
    
    
    # Rimuoviamo i paragrafi che possono essere problematici. 
    new_texts = texts
    indices = []
    for i in range(len(texts)):
        if correct_text(texts[i]) == False or all(tag == "O" for tag in tags[i]):
            indices.append(i)
            
    new_texts = np.delete(texts, indices, 0)
    new_tags = np.delete(tags, indices, 0)
    new_labels = np.delete(labels, indices, 0)
    texts = new_texts.tolist()
    tags = new_tags.tolist()
    labels = new_labels.tolist() 
   
    # Ora dobbiamo usare il tokenizer di BERT per ottenere i veri token, non
    # quelli di SpaCy. BERT può suddividere una parola in più sotto-parole,
    # quindi occorre ripetere il tag della parola per ciascuna sotto-parola.
    # Il tokenizer di BERT aggiunge dei token speciali [CLS] e [SEP] ed effettua
    # il padding del testo fino a 512 token. Se il padding viene impostato a
    # "max_length" allora il padding sarà sempre di 512 token.
    # Qui gli encodings sono oggetti restituiti da BERT con specifici attributi.
    # L'attributo "tokens" restituisce i token (parole) di un paragrafo.
    # L'attributo "ids" restituisce degli indici (numeri) associati a ciascun token.
    # L'attributo "attention_mask" attribuisce un 1 in corrispondenza di ogni token
    # del paragrafo originale e 0 in corrispondenza dei token aggiunti con il padding.
    # In questo modo il modello non si concentra sui token aggiunti.
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        padding=padding, 
        truncation=True
    )

    tokens_labels = [] 

    for i, (text, text_tags) in tqdm(
        enumerate(zip(texts, tags)),
        desc="Adjusting tags to encodings",
        leave=False
    ):
        #print("text: ", text)
        #print("tags: ", tags)
        text_tokens_tags = get_text_tokens_tags(text, text_tags, tokenizer)
        text_enc_tokens_labels = np.ones(len(encodings[i]), dtype=int) * -100
        
        # Vengono ricreati i tag dei token tenendo conto delle parole suddivise
        # in sotto-parole.
        text_enc_tokens_labels[1:len(text_tokens_tags) + 1] = [
                tag2idx[tag] for tag in text_tokens_tags
            ]
        tokens_labels.append(text_enc_tokens_labels.tolist())

    if ignore_other:
        # Abbiamo già convertito tutte le "O" in "MASK". Quindi
        # possiamo rimuovere le chiavi corrispondenti dal dizionario.
        del tag2idx["O"]

    #print("tokens_labels: ", tokens_labels)
    # tokens_labels sono in realtà i tag dei token.
    # -100: O
    # 0: B-hist
    # 1: B-not-hist
    # 2: I-hist
    # 3: I-not-hist
    return encodings, tokens_labels, labels, tag2idx, idx2tag


def correct_text(text):
    result = True
    
    # Il paragrafo è molto corto o molto lungo.
    if len(text) < 20 or len(text) > 480:
        result = False
    
    # Il paragrafo non termina con un punto.
    if text[len(text)-1] != ".":
        result = False
        
    # Il paragrafo contiene una parola con un punto alternato ad altre lettere.
    # Esempio: "a.b", "abc." e ".xyz".
    regex = r"\.\w+"
    for word in text:
        if re.search(regex, word):
            result = False
    
    special_symbols = [':', '|', '\'', '\\']
    for word in text:
        for symbol in special_symbols:
            if symbol in word:
                result = False
        
    # Il paragrafo contiene parole problematiche.
    for word in text:
        if word == "--" or word == "\xa0" or word == "[" or word == "]" \
            or word == "\n" or word == "{" or word == "}" or word == "=" \
            or word == "\ufeff" or word == "...":
                result = False
    
    return result


def get_text_tokens_tags(text, tags, tokenizer):
    text_tokens_tags = []

    # Se una parola viene suddivisa in più sotto-parole dal tokenizer, allora
    # ripetiamo il tag della parola per ciascuna sotto-parola.
    for word, tag in zip(text, tags):
        tokenized_word = tokenizer.tokenize(word)
        text_tokens_tags.extend([tag] * len(tokenized_word))

    return text_tokens_tags

