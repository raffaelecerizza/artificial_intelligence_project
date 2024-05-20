import json
import os
import torch

from event_models_utils import (
    load_rams_data, load_ontology, template2tokens, get_event_names_dict
)
from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn
from torch.nn import (
    CrossEntropyLoss
)
from torch.nn import functional as F 
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1, Precision, Recall
from transformers import (
    BartConfig, BartModel, BartTokenizer, BertModel, BertTokenizer
)
from transformers.file_utils import ModelOutput
from transformers.generation_utils import top_k_top_p_filtering
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from typing import Iterable, Optional

###############################################################################
# RAMS Dataset
###############################################################################

class RAMSDataset(Dataset):
    def __init__(
        self, input_ids, attention_masks, spans, 
        spans_trg_true, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys, doc_tokens,
        span_mappers
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.spans = spans
        self.spans_trg_true = spans_trg_true
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys
        self.doc_tokens = doc_tokens
        self.span_mappers = span_mappers

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], \
            self.spans[idx], self.spans_trg_true[idx], \
            self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx], \
            self.doc_tokens[idx], self.span_mappers[idx]


class RAMSArgumentDataset(Dataset):
    def __init__(
        self, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys
    ):
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        return self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx]


class RAMSEventGenDataset(Dataset):
    def __init__(
        self, encoder_input_ids,
        encoder_attention_mask, dec_input_ids,
        dec_attention_mask, doc_keys
    ):
        self.encoder_input_ids = encoder_input_ids
        self.encoder_attention_mask = encoder_attention_mask 
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.doc_keys = doc_keys

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        return self.encoder_input_ids[idx], \
            self.encoder_attention_mask[idx], self.dec_input_ids[idx], \
            self.dec_attention_mask[idx], self.doc_keys[idx]


def collate_RAMS(batch):
    tokens_ids = torch.tensor([item[0] for item in batch])
    attention_masks = torch.tensor([item[1] for item in batch])
    spans = [item[2] for item in batch]
    # Per ogni frase del batch specifica quale span contiene il trigger dell'evento (e quale evento rappresenta).
    spans_trg_labels = [item[3] for item in batch]
    encoder_input_ids = torch.tensor([item[4] for item in batch])
    encoder_attention_mask = torch.tensor([item[5] for item in batch])
    # dec = decoder
    dec_input_ids = torch.tensor([item[6] for item in batch])
    dec_attention_mask = torch.tensor([item[7] for item in batch])
    doc_keys = [item[8] for item in batch]
    doc_tokens = [item[9] for item in batch]
    span_mappers = [item[10] for item in batch]
        
    return {
        "input_ids": tokens_ids, 
        "attention_masks": attention_masks,
        "spans": spans, 
        "spans_trg_true": list2tensor(spans_trg_labels),
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "doc_keys": doc_keys,
        "doc_tokens": doc_tokens,
        "span_mappers": span_mappers
    }


def collate_argument_RAMS(batch):
    encoder_input_ids = torch.tensor([item[0] for item in batch])
    encoder_attention_mask = torch.tensor([item[1] for item in batch])
    dec_input_ids = torch.tensor([item[2] for item in batch])
    dec_attention_mask = torch.tensor([item[3] for item in batch])
    doc_keys = [item[4] for item in batch]

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "doc_keys": doc_keys
    }


class RAMSDataModule(LightningDataModule):
    def __init__(
        self, 
        #data_dir: str = "datasets/rams",
        data_dir: str = "/content/drive/MyDrive/history/datasets/rams",
        batch_size: int = 16, 
        num_workers: int = 0,
        pin_memory: bool = True # se pin_memory è True, l'utilizzo della GPU è più efficiente
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assegna i dataset di train/val per essere usati nei DataLoader.
        if stage == "fit" or stage is None:
            self.rams_train = RAMSDataset(
                input_ids=self.train_dict["tokens_ids"], 
                attention_masks=self.train_dict["attention_masks"], 
                spans=self.train_dict["spans"], 
                spans_trg_true=self.train_dict["spans_trg_labels"],
                encoder_input_ids=self.train_dict["args_ids"], 
                encoder_attention_mask=self.train_dict["args_masks"],
                dec_input_ids=self.train_dict["args_dec_ids"],
                dec_attention_mask=self.train_dict["args_dec_masks"],
                doc_keys=self.train_dict["doc_keys"],
                doc_tokens=self.train_dict["tokens"],
                span_mappers=self.train_dict["span_mappers"]
            )

            self.rams_valid = RAMSDataset(
                input_ids=self.valid_dict["tokens_ids"], 
                attention_masks=self.valid_dict["attention_masks"], 
                spans=self.valid_dict["spans"], 
                spans_trg_true=self.valid_dict["spans_trg_labels"],
                encoder_input_ids=self.valid_dict["args_ids"], 
                encoder_attention_mask=self.valid_dict["args_masks"],
                dec_input_ids=self.valid_dict["args_dec_ids"],
                dec_attention_mask=self.valid_dict["args_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"],
                doc_tokens=self.valid_dict["tokens"],
                span_mappers=self.valid_dict["span_mappers"]
            )

        # Assegna il dataset di test per essere usato nei DataLoader.
        if stage == "test" or stage is None:
            self.rams_test = RAMSDataset(
                input_ids=self.test_dict["tokens_ids"], 
                attention_masks=self.test_dict["attention_masks"], 
                spans=self.test_dict["spans"], 
                spans_trg_true=self.test_dict["spans_trg_labels"],
                encoder_input_ids=self.test_dict["args_ids"], 
                encoder_attention_mask=self.test_dict["args_masks"],
                dec_input_ids=self.test_dict["args_dec_ids"],
                dec_attention_mask=self.test_dict["args_dec_masks"],
                doc_keys=self.test_dict["doc_keys"],
                doc_tokens=self.test_dict["tokens"],
                span_mappers=self.test_dict["span_mappers"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_RAMS
        ) 


class RAMSArgumentDataModule(LightningDataModule):
    def __init__(
        self, 
        #data_dir: str = "datasets/rams",
        data_dir: str = "/content/drive/MyDrive/history/datasets/rams",
        batch_size: int = 16, 
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assegna i dataset di train/val per essere usati nei DataLoader.
        if stage == "fit" or stage is None:
            self.rams_train = RAMSArgumentDataset(
                encoder_input_ids=self.train_dict["args_ids"], 
                encoder_attention_mask=self.train_dict["args_masks"],
                dec_input_ids=self.train_dict["args_dec_ids"],
                dec_attention_mask=self.train_dict["args_dec_masks"],
                doc_keys=self.train_dict["doc_keys"]
            )

            self.rams_valid = RAMSArgumentDataset(
                encoder_input_ids=self.valid_dict["args_ids"], 
                encoder_attention_mask=self.valid_dict["args_masks"],
                dec_input_ids=self.valid_dict["args_dec_ids"],
                dec_attention_mask=self.valid_dict["args_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"]
            )

        # Assegna il dataset di test per essere usato nei DataLoader.
        if stage == "test" or stage is None:
            self.rams_test = RAMSArgumentDataset(
                encoder_input_ids=self.test_dict["args_ids"], 
                encoder_attention_mask=self.test_dict["args_masks"],
                dec_input_ids=self.test_dict["args_dec_ids"],
                dec_attention_mask=self.test_dict["args_dec_masks"],
                doc_keys=self.test_dict["doc_keys"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_argument_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        ) 


class RAMSEventGenDataModule(LightningDataModule):
    def __init__(
        self, 
        #data_dir: str = "datasets/rams",
        data_dir: str = "/content/drive/MyDrive/history/datasets/rams",
        batch_size: int = 16, 
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, split: str):
        with open(
            f"{self.data_dir}/preprocessed/{split}.json", "r"
        ) as f_in:
            return json.load(f_in)

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("dev")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):
        # Assegna i dataset di train/val per essere usati nei DataLoader.
        if stage == "fit" or stage is None:
            self.rams_train = RAMSEventGenDataset(
                encoder_input_ids=self.train_dict["evt_ids"], 
                encoder_attention_mask=self.train_dict["evt_masks"],
                dec_input_ids=self.train_dict["evt_dec_ids"],
                dec_attention_mask=self.train_dict["evt_dec_masks"],
                doc_keys=self.train_dict["doc_keys"]
            )

            self.rams_valid = RAMSEventGenDataset(
                encoder_input_ids=self.valid_dict["evt_ids"], 
                encoder_attention_mask=self.valid_dict["evt_masks"],
                dec_input_ids=self.valid_dict["evt_dec_ids"],
                dec_attention_mask=self.valid_dict["evt_dec_masks"],
                doc_keys=self.valid_dict["doc_keys"]
            )

        # Assegna il dataset di test per essere usato nei DataLoader.
        if stage == "test" or stage is None:
            self.rams_test = RAMSEventGenDataset(
                encoder_input_ids=self.test_dict["evt_ids"], 
                encoder_attention_mask=self.test_dict["evt_masks"],
                dec_input_ids=self.test_dict["evt_dec_ids"],
                dec_attention_mask=self.test_dict["evt_dec_masks"],
                doc_keys=self.test_dict["doc_keys"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.rams_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_argument_RAMS
        )

    def val_dataloader(self):
        return DataLoader(
            self.rams_valid, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        )

    def test_dataloader(self):
        return DataLoader(
            self.rams_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=collate_argument_RAMS
        ) 


def list2tensor(lss):
    res = torch.tensor([])
    for ls in lss:
        res = torch.cat([res, torch.tensor(ls)])
    return res



###############################################################################
# Modello per gli eventi
###############################################################################

class EventModel(LightningModule):
    def __init__(
        self,
        bert,
        tokenizer,
        bart_tokenizer,
        #data_dir: str = "datasets/rams",
        data_dir: str = "/content/drive/MyDrive/history/datasets/rams",
        #ontology_dir: str = "datasets",
        ontology_dir: str = "/content/drive/MyDrive/history/datasets",
        num_events: int = 140, # 139 eventi + no_evt 
        dropout_rate: float = 0.2,
        bert_hidden_dims: int = 768, 
        hidden_dims: int = 150,
        max_span_length: int = 3,
        width_embedding_dim: int = 768,
        average: str = "weighted"
    ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.bart_tokenizer.add_tokens([" <arg>", " <trg>"])
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(bert_hidden_dims * 3, hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_events)
        )

        self.idx2evt, self.evt2idx = self.load_event_dicts()

        self.average = average

        self.accuracy = Accuracy(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.f1 = F1(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.prec = Precision(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.recall = Recall(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )

        # Con full non ignoriamo i non-eventi.
        self.accuracy_full = Accuracy(
            num_classes=num_events,
            average=average
        )
        self.f1_full = F1(
            num_classes=num_events,
            average=average
        )
        self.prec_full = Precision(
            num_classes=num_events,
            average=average
        )
        self.recall_full = Recall(
            num_classes=num_events,
            average=average
        )
        
        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def load_event_dicts(self):
        _, data_dicts = load_rams_data(self.data_dir)

        return data_dicts["idx2evt"], data_dicts["evt2idx"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def score(self, span_logits, span_true):
        span_preds = torch.argmax(span_logits, -1)
        acc = self.accuracy(span_preds.view(-1), span_true.view(-1).int())
        f1 = self.f1(span_preds.view(-1), span_true.view(-1).int())
        prec = self.prec(span_preds.view(-1), span_true.view(-1).int())
        recall = self.recall(span_preds.view(-1), span_true.view(-1).int())

        acc_f = self.accuracy_full(span_preds.view(-1), span_true.view(-1).int())
        f1_f = self.f1_full(span_preds.view(-1), span_true.view(-1).int())
        prec_f = self.prec_full(span_preds.view(-1), span_true.view(-1).int())
        recall_f = self.recall_full(span_preds.view(-1), span_true.view(-1).int())

        scores = {
            "acc": acc,
            "f1": f1,
            "prec": prec, 
            "recall": recall, 
            "acc_f": acc_f, 
            "f1_f": f1_f, 
            "prec_f": prec_f, 
            "recall_f": recall_f
        }

        return scores

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        span_batch,
        span_mappers,
        doc_tokens,
        training: bool = False
    ):
        # output è un oggetto di tipo BaseModelOutputWithPoolingAndCrossAttentions
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
        # Esempio last_hidden_state.shape:  torch.Size([2, 512, 768]):
        # 2 è il numero di frasi, che corrisponde al batch_size;
        # 512 è il numero di token per frase;
        # 768 è la dimensione latente di BERT.
        last_hidden_state = output.last_hidden_state
        span_embeddings = []
        span_lists_lengths = []

        # Se ci sono 2 frasi, allora span_batch ha lunghezza 2.
        # span_list è la lista degli span di ciascuna frase.
        for batch_num, span_list in enumerate(span_batch):
            span_lists_lengths.append(len(span_list))
            batch_hidden_state = last_hidden_state[batch_num]
        
            batch_span_embs = []
            for span in span_list:
                span_start, span_end, span_width = span
                # embedding della prima parola di uno span
                first_word_emb = batch_hidden_state[span_start]
                # embedding dell'ultima parola di uno span
                last_word_emb = batch_hidden_state[span_end]
                # lunghezza dello span
                span_width = torch.tensor([span_width], device=self.device)
                # media dei valori degli embedding delle parole dello span
                mean_span = torch.mean(
                    batch_hidden_state[span_start:span_end + 1],
                    dim=0
                )

                span_emb = torch.cat(
                    [
                        first_word_emb,
                        last_word_emb,
                        mean_span,
                    ],
                    dim = -1
                )
                batch_span_embs.append(span_emb)

            span_embeddings.append(torch.stack(batch_span_embs))

        span_embeddings = torch.cat(span_embeddings)
        
        # I logits sono ottenuti come output dell'ultimo layer del modello
        # applicato agli embedding degli span.
        # Di fatto solo i logits vengono poi usati per l'apprendimento e la 
        # classificazione.
        logits = self.clf(span_embeddings)

        # Recupera il nome dell'evento per ogni batch. 
        evt_names = []
        spans_with_evts = []
        start = 0
        
        # span_lists_lengths contiene il numero di span di ogni frase.
        # Esempio: span_lists_lengths:  [234, 405].
        for spans_len in span_lists_lengths:
            
            end = start + spans_len
            
            # logits contiene i logits degli span di tutte le frasi del batch.
            # Esempio: logits.shape:  torch.Size([639, 140]). 
            # Qui si recuperano i logits di una specifica frase.
            # Esempio: span_logits.shape:  torch.Size([235, 140]).
            span_logits = logits[start:end + 1]

            # In realtà probs non sono probabilità, dato che non abbiamo
            # applicato la softmax. Tuttavia qui assolvono allo stesso ruolo.
            # probs specifica il più alto valore di probabilità per ogni span della frase.
            # Esempio: probs.shape:  torch.Size([235]).
            # evt_ids è l'id dell'evento corrispondente al valore di probabilità più alto per ogni span.
            # Esempio: evt_ids.shape:  torch.Size([235]).
            probs, evt_ids = torch.max(span_logits, -1)
            # range_ids è semplicemente una lista di indici: range_ids.shape:  torch.Size([235]).
            range_ids = torch.tensor(list(range(len(probs)))).to("cuda")
            probs = probs.clone().detach().to("cuda")
            evt_ids = evt_ids.clone().detach().to("cuda")
   
            # Controlliamo solo gli eventi che non sono non-eventi.
            probs = probs[evt_ids > 0]
            range_ids = range_ids[evt_ids > 0]
            evt_ids = evt_ids[evt_ids > 0]

            # Qui la lunghezza di probs è 0 se la probabilità più alta era per
            # i non-eventi (cioè l'evento con indice 0).
            if len(probs) == 0:
                # Nessun evento trovato. Questo non è corretto nel training,
                # dato che ogni istanza ha un evento. Prendiamo un evento
                # random per uno span random.
                # TODO: fare in modo che quando non si è in training, se non
                # ci sono eventi, ci si ferma. Non si potrebbe comunque produrre
                # un template per il modello degli argomenti.
                evt_names.append(
                    self.idx2evt[1 + torch.randint(139, (1,)).item()] # + 1 per evitare no_evt
                )
                spans_with_evts.append(
                    torch.randint(spans_len - 1, (1,)).item()
                )
            else:
                max_prob, max_prob_id, max_range_id = -1000, -1, -1
                # Qui viene individuato l'evento più probabile (max_prob_id).
                for prob, prob_id, range_id in zip(probs, evt_ids, range_ids):
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_id = prob_id
                        max_range_id = range_id
                
                # Qui viene recuperato il nome dell'evento più probabile.
                evt_names.append(self.idx2evt[max_prob_id.item()])
                if max_range_id == spans_len:
                    max_range_id = max_range_id - 1

                # Questi sono gli indici degli span che corrispondono agli eventi.
                spans_with_evts.append(max_range_id)

            start = end

        enc_ids = None
        enc_attn_masks = None
        enc_sentences = None

        # Questa parte di fatto non viene più usata.
        if training:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_train(
                input_ids=input_ids,
                span_batch=span_batch,
                spans_with_evts=spans_with_evts,
                evt_names=evt_names,
                doc_tokens=doc_tokens,
                span_mappers=span_mappers,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )
        else:
            enc_ids, enc_attn_masks, enc_sentences = get_bart_sentences_not_train(
                input_ids=input_ids,
                span_batch=span_batch,
                spans_with_evts=spans_with_evts,
                evt_names=evt_names,
                bert_tokenizer=self.tokenizer,
                bart_tokenizer=self.bart_tokenizer,
                ontology_base_path=self.ontology_dir
            )

        return logits, evt_names, enc_ids, enc_attn_masks

    def training_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )
        
        # I logits specificano uno valore per ogni tipo di evento (compreso il
        # non-evento). Lo spans_trg_true specifica l'evento (o non-evento) corretto
        # per ogni span. Per ogni frase ci sarà un solo span con un evento, e questo
        # evento sarà di un tipo compreso nei 139 tipi del dataset. La maggior parte
        # degli span sarà di tipo non-evento. E questo rende il problema fortemente
        # sbilanciato.
        # Esempio:
        # logits.shape:  torch.Size([249, 140])
        # spans_trg_true.shape:  torch.Size([249])
        # len(batch[spans][0]):  249
        spans_trg_true = batch["spans_trg_true"]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        scores = self.score(logits, spans_trg_true)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", scores["acc"], prog_bar=True)
        self.log("train_f1", scores["f1"], prog_bar=True)
        self.log("train_prec", scores["prec"], prog_bar=True)
        self.log("train_recall", scores["recall"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=True
        )

        spans_trg_true = batch["spans_trg_true"]

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            spans_trg_true.view(-1).long()
        )

        scores = self.score(logits, spans_trg_true)

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", scores["acc"], prog_bar=True)
        self.log("valid_f1", scores["f1"], prog_bar=True)
        self.log("valid_prec", scores["prec"], prog_bar=True)
        self.log("valid_recall", scores["recall"], prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, evt_names, enc_ids, enc_attn_masks = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            batch["spans"],
            batch["span_mappers"],
            batch["doc_tokens"],
            training=False
        )

        spans_trg_true = batch["spans_trg_true"]

        scores = self.score(logits, spans_trg_true)
        
        return list(scores.values())

    def test_epoch_end(self, outputs):
        save_dir = "/content/drive/MyDrive/history/results/events"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "final_acc_f": self.accuracy_full.compute().item(),
            "final_f1_f": self.f1_full.compute().item(),
            "final_prec_f": self.prec_full.compute().item(),
            "final_recall_f": self.recall_full.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item(),
                "acc_f": tup[4].item(),  
                "f1_f": tup[5].item(),
                "prec_f": tup[6].item(),
                "recall_f": tup[7].item()
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/event_extraction_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)


def get_bart_sentences_train(
    input_ids, span_batch, 
    spans_with_evts, evt_names,
    doc_tokens, span_mappers,
    bart_tokenizer,
    #ontology_base_path: str = "datasets"
    ontology_base_path: str = "/content/drive/MyDrive/history/datasets"
):
    enc_ids = []
    enc_attn_masks = []
    enc_sentences = []

    for idx, (span_list, evt_span_idx, batch_ids, evt_name) in enumerate(
        zip(span_batch, spans_with_evts, input_ids, evt_names)
    ):
        # Queste sono le parole (token) di una frase.
        tokens = doc_tokens[idx]
        
        span_mapper = span_mappers[idx]
        #print("span_mapper: ", span_mapper)

        # L'original_evt_span è il trigger dell'evento.
        # Esempio: original_evt_span:  [39, 39, 1].
        original_evt_span = span_mapper[evt_span_idx]
        original_span_start, original_span_end, _ = original_evt_span

        ontology_dict = load_ontology(ontology_base_path)
        # Qui viene recuperato il template dello specifico evento.
        # Esempio: template:  <arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place.
        template = ontology_dict[evt_name.replace("n/a", "unspecified")]["template"]

        # Qui viene trasformato il template in un template di token ottenuti con BART.
        # Esempio: template_in:  [' <arg>', 'Ġtransported', ' <arg>', 'Ġin', ' <arg>', 'Ġfrom', ' <arg>', 'Ġplace', 'Ġto', ' <arg>', 'Ġplace'].
        template_in = template2tokens(template, bart_tokenizer)

        # Qui viene aggiunto un prefisso a ogni token della frase (non più del template)
        # fino al trigger.
        prefix = bart_tokenizer.tokenize(
            " ".join(tokens[:original_span_start]),
            add_prefix_space=True
        )

        # Qui viene aggiunto un prefisso al trigger della frase (non più del template).
        trg = bart_tokenizer.tokenize(
            " ".join(tokens[original_span_start:original_span_end+1]),
            add_prefix_space=True
        )

        # Qui viene aggiunto un prefisso a ogni token della frase (non più del template)
        # successivo al trigger.
        suffix = bart_tokenizer.tokenize(
            " ".join(tokens[original_span_end+1:]),
            add_prefix_space=True
        )

        # Questa è la concatenazione dei token con prefisso. Prima e dopo il trigger
        # vengono aggiunti dei tag che specificano il trigger.
        '''
        Esempio:
        context:  ['ĠThree', 'Ġspecific', 'Ġpoints', 'Ġillustrate', 'Ġwhy', 'ĠAmericans', 
        'Ġsee', 'ĠTrump', 'Ġas', 'Ġthe', 'Ġproblem', 'Ġ:', 'Ġ1', 'Ġ)', 'ĠTrump', 'Ġhas', 
        'Ġtrouble', 'Ġworking', 'Ġwith', 'Ġpeople', 'Ġbeyond', 'Ġhis', 'Ġbase', 'Ġ.', 'ĠIn', 
        'ĠSaddam', 'ĠHussein', "Ġ'", 's', 'ĠIraq', 'Ġthat', 'Ġmight', 'Ġwork', 'Ġwhen', 'Ġopponents', 
        'Ġcan', 'Ġbe', 'Ġthrown', 'Ġin', 'Ġjail', ' <trg>', 'Ġor', ' <trg>', 'Ġextermin', 'ated', 
        'Ġ.', 'ĠIn', 'Ġthe', 'ĠUnited', 'ĠStates', 'Ġthat', 'Ġwo', 'Ġn', "'t", 'Ġfly', 'Ġ:', 
        'Ġpresidents', 'Ġmust', 'Ġbuild', 'Ġbridges', 'Ġwithin', 'Ġand', 'Ġbeyond', 'Ġtheir', 
        'Ġcore', 'Ġsupport', 'Ġto', 'Ġresolve', 'Ġchallenges', 'Ġ.', 'ĠWithout', 'Ġalliances', 
        'Ġ,', 'Ġa', 'Ġpresident', 'Ġca', 'Ġn', "'t", 'Ġget', 'Ġapproval', 'Ġto', 'Ġget', 'Ġthings', 'Ġdone', 'Ġ.']
        '''
        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix

        arg_in = bart_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        enc_id = arg_in["input_ids"]
        '''
        Esempio:
        enc_sentece:  <s>  <arg>  transported  <arg>  in  <arg>  from  <arg>  place to  <arg>  place</s></s> 
        Three specific points illustrate why Americans see Trump as the problem : 1 ) Trump has trouble working with 
        people beyond his base. In Saddam Hussein's Iraq that might work when opponents can be thrown in jail  <trg>  or  <trg>  exterminated. 
        In the United States that won't fly : presidents must build bridges within and beyond their core support to resolve challenges. 
        Without alliances, a president can't get approval to get things done.</s><pad>...<pad>
        '''
        enc_sentence = bart_tokenizer.decode(enc_id)
        
        enc_ids.append(enc_id)
        enc_attn_masks.append(arg_in["attention_mask"])
        enc_sentences.append(enc_sentence)

    return enc_ids, enc_attn_masks, enc_sentences


def get_bart_sentences_not_train(
    input_ids, span_batch, 
    spans_with_evts, evt_names,
    bert_tokenizer, bart_tokenizer,
    #ontology_base_path: str = "datasets"
    ontology_base_path: str = "/content/drive/MyDrive/history/datasets"
):
    enc_ids = []
    enc_attn_masks = []
    enc_sentences = []
    ontology_dict = load_ontology(ontology_base_path)

    for span_list, evt_span_idx, batch_ids, evt_name in zip(
        span_batch, spans_with_evts, input_ids, evt_names
    ):
        # Esempio:
        # template:  <arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place
        template = ontology_dict[evt_name.replace("n/a", "unspecified")]["template"]
        
        # Esempio:
        # template_in:  [' <arg>', 'Ġmet', 'Ġface', '-', 'to', '-', 'face', 'Ġwith', ' <arg>', 'Ġabout', ' <arg>', 'Ġtopic', 'Ġat', ' <arg>', 'Ġplace']
        template_in = template2tokens(template, bart_tokenizer)

        # Esempio:
        # evt_span:  [37, 39, 3]
        evt_span = span_list[evt_span_idx]
        span_start, span_end, _ = evt_span

        # Esempio:
        # prefix_text:  [CLS] We are ashamed of them. " However, Mutko stopped short of admitting the doping scandal 
        # was state sponsored. " We are very sorry that athletes who tried to deceive
        prefix_text = bert_tokenizer.decode(batch_ids[:span_start])
        
        # Esempio:
        # prefix_text:  We are ashamed of them. " However, Mutko stopped short of admitting the doping scandal 
        # was state sponsored. " We are very sorry that athletes who tried to deceive
        prefix_text = remove_special_tokens(prefix_text).strip()
        
        # Esempio:
        # prefix:  ['ĠWe', 'Ġare', 'Ġashamed', 'Ġof', 'Ġthem', '.', 'Ġ"', 'ĠHowever', ',', 'ĠMut', 'ko', 'Ġstopped', 
        # 'Ġshort', 'Ġof', 'Ġadmitting', 'Ġthe', 'Ġdoping', 'Ġscandal', 'Ġwas', 'Ġstate', 'Ġsponsored', '.', 'Ġ"', 'ĠWe', 
        # 'Ġare', 'Ġvery', 'Ġsorry', 'Ġthat', 'Ġathletes', 'Ġwho', 'Ġtried', 'Ġto', 'Ġdeceive']
        prefix = bart_tokenizer.tokenize(
            prefix_text,
            add_prefix_space=True
        )
 
        # Esempio:
        # trg_text:  us, and
        trg_text = bert_tokenizer.decode(batch_ids[span_start : span_end + 1])
        trg_text = remove_special_tokens(trg_text).strip()
        
        # Esempio:
        # trg:  ['Ġus', ',', 'Ġand']
        trg = bart_tokenizer.tokenize(
            trg_text,
            add_prefix_space=True
        )
        
        # Esempio:
        # suffix_text:  the world, were not caught sooner. We are very sorry because Russia is committed to upholding the highest 
        # standards in sport and is opposed to anything that threatens the Olympic values, " he said. English former heptathlete and Athens 2004 
        # bronze medallist Kelly Sotherton was unhappy with Mutko's plea for Russia's ban to be lifted for Rio [SEP] [PAD] [PAD] ...
        suffix_text = bert_tokenizer.decode(batch_ids[span_end + 1:])
        
        # Esempio:
        # suffix_text:  the world, were not caught sooner. We are very sorry because Russia is committed 
        # to upholding the highest standards in sport and is opposed to anything that threatens 
        # the Olympic values, " he said. English former heptathlete and Athens 2004 bronze medallist Kelly Sotherton 
        # was unhappy with Mutko's plea for Russia's ban to be lifted for Rio
        suffix_text = remove_special_tokens(suffix_text).strip()
        
        # Esempio:
        # suffix:  ['Ġthe', 'Ġworld', ',', 'Ġwere', 'Ġnot', 'Ġcaught', 'Ġsooner', '.', 'ĠWe', 
        # 'Ġare', 'Ġvery', 'Ġsorry', 'Ġbecause', 'ĠRussia', 'Ġis', 'Ġcommitted', 'Ġto', 
        # 'Ġupholding', 'Ġthe', 'Ġhighest', 'Ġstandards', 'Ġin', 'Ġsport', 'Ġand', 'Ġis', 
        # 'Ġopposed', 'Ġto', 'Ġanything', 'Ġthat', 'Ġthreatens', 'Ġthe', 'ĠOlympic', 'Ġvalues', ',', 
        # 'Ġ"', 'Ġhe', 'Ġsaid', '.', 'ĠEnglish', 'Ġformer', 'Ġhe', 'pt', 'ath', 'lete', 'Ġand', 'ĠAthens', 
        # 'Ġ2004', 'Ġbronze', 'Ġmed', 'all', 'ist', 'ĠKelly', 'ĠS', 'other', 'ton', 'Ġwas', 'Ġunhappy', 'Ġwith', 
        # 'ĠMut', 'ko', "'s", 'Ġplea', 'Ġfor', 'ĠRussia', "'s", 'Ġban', 'Ġto', 'Ġbe', 'Ġlifted', 'Ġfor', 'ĠRio']
        suffix = bart_tokenizer.tokenize(
            suffix_text,
            add_prefix_space=True
        )
        
        # Esempio:
        # context:  ['ĠWe', 'Ġare', 'Ġashamed', 'Ġof', 'Ġthem', '.', 'Ġ"', 'ĠHowever', ',', 
        # 'ĠMut', 'ko', 'Ġstopped', 'Ġshort', 'Ġof', 'Ġadmitting', 'Ġthe', 'Ġdoping', 'Ġscandal', 
        # 'Ġwas', 'Ġstate', 'Ġsponsored', '.', 'Ġ"', 'ĠWe', 'Ġare', 'Ġvery', 'Ġsorry', 'Ġthat', 'Ġathletes', 
        # 'Ġwho', 'Ġtried', 'Ġto', 'Ġdeceive', ' <trg>', 'Ġus', ',', 'Ġand', ' <trg>', 'Ġthe', 'Ġworld', ',', 'Ġwere', 
        # 'Ġnot', 'Ġcaught', 'Ġsooner', '.', 'ĠWe', 'Ġare', 'Ġvery', 'Ġsorry', 'Ġbecause', 'ĠRussia', 'Ġis', 
        # 'Ġcommitted', 'Ġto', 'Ġupholding', 'Ġthe', 'Ġhighest', 'Ġstandards', 'Ġin', 'Ġsport', 'Ġand', 'Ġis', 
        # 'Ġopposed', 'Ġto', 'Ġanything', 'Ġthat', 'Ġthreatens', 'Ġthe', 'ĠOlympic', 'Ġvalues', ',', 'Ġ"', 'Ġhe', 
        # 'Ġsaid', '.', 'ĠEnglish', 'Ġformer', 'Ġhe', 'pt', 'ath', 'lete', 'Ġand', 'ĠAthens', 'Ġ2004', 'Ġbronze', 
        # 'Ġmed', 'all', 'ist', 'ĠKelly', 'ĠS', 'other', 'ton', 'Ġwas', 'Ġunhappy', 'Ġwith', 'ĠMut', 'ko', "'s", 
        # 'Ġplea', 'Ġfor', 'ĠRussia', "'s", 'Ġban', 'Ġto', 'Ġbe', 'Ġlifted', 'Ġfor', 'ĠRio']
        context = prefix + [" <trg>", ] + trg + [" <trg>", ] + suffix
        
        arg_in = bart_tokenizer.encode_plus(
            template_in, 
            context, 
            add_special_tokens=True,
            add_prefix_space=True,
            max_length=424,
            truncation="only_second",
            padding="max_length"
        )

        # Esempio:
        # enc_sentence:  <s>  <arg>  met face-to-face with  <arg>  about  <arg>  topic at  <arg>  place</s></s> We are ashamed of them. " However, 
        # Mutko stopped short of admitting the doping scandal was state sponsored. " We are very sorry that athletes who tried 
        # to deceive  <trg>  us, and  <trg>  the world, were not caught sooner. We are very sorry because Russia is committed 
        # to upholding the highest standards in sport and is opposed to anything that threatens the Olympic values, " he said. English former 
        # heptathlete and Athens 2004 bronze medallist Kelly Sotherton was unhappy with Mutko's plea for Russia's ban to be lifted for Rio</s><pad><pad><pad>
        enc_id = arg_in["input_ids"]
        enc_sentence = bart_tokenizer.decode(enc_id)

        enc_ids.append(enc_id)
        enc_attn_masks.append(arg_in["attention_mask"])
        enc_sentences.append(enc_sentence)

    return enc_ids, enc_attn_masks, enc_sentences


def remove_special_tokens(text: str):
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.replace("[PAD]", "")
    return text.strip()



###############################################################################
# Modello EventGen
###############################################################################

class EventGenModelWrapper(LightningModule):
    def __init__(self, bart, bart_tokenizer):
        super(EventGenModelWrapper, self).__init__()
        self.model = EventGenModel(
            BartConfig.from_pretrained("facebook/bart-base"),
            bart=bart,
            bart_tokenizer=bart_tokenizer
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(
        self, 
        encoder_input_ids, 
        encoder_attention_mask,
        dec_input_ids,
        dec_attention_mask,
        training: bool = False
    ):
        return self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=dec_attention_mask,
            training=training
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(logits.view(-1, logits.shape[-1]), labels.view(-1))

        loss = torch.mean(loss)
    
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(logits.view(-1, logits.shape[-1]), labels.view(-1))

        loss = torch.mean(loss)
        self.log("valid_loss", loss, prog_bar=True)
    
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"]
        )

        # Esempio:
        # filled_templates:  tensor([[    2,  8241,   848,  8241,   634, 50265, 10320,    50, 50265,  1131,
        #    696,    23, 50265,   317,     2]], device='cuda:0')
        filled_templates = self.model.generate(
            batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"], 
            do_sample=True, 
            top_k=20, 
            top_p=0.95, 
            max_length=30, 
            num_return_sequences=1,
            num_beams=1,
            repetition_penalty=1
        )

        return (
            batch["doc_keys"], 
            filled_templates, 
            batch["dec_input_ids"]
        )

    def test_epoch_end(self, outputs):
        os.makedirs("/content/drive/MyDrive/history/results/events", exist_ok=True)

        with open(
            "/content/drive/MyDrive/history/results/events/event_gen_predictions.jsonl",
            "w"
        ) as writer:
            for tup in outputs:
                pred = {
                    "doc_key": tup[0],
                    "predicted": self.model.tokenizer.decode(
                        tup[1].squeeze(0), 
                        skip_special_tokens=True
                    ),
                    "gold": self.model.tokenizer.decode(
                        tup[2].squeeze(0), 
                        skip_special_tokens=True
                    ) 
                }
                writer.write(json.dumps(pred)+'\n')

        return {}


class EventGenModel(PreTrainedModel):
    """
    Code adapted from the paper: 
    Li S., Ji H., Han J., Document-level event argument extraction by 
    conditional generation. In: Proceedings of the 2021 Conference of the 
    North American Chapter of the Association for Computational Linguistics: 
    Human Language Technologies, 2021, pp. 894-908.
    https://github.com/raspberryice/gen-arg/
    """

    def __init__(self, config, bart, bart_tokenizer):
        super(EventGenModel, self).__init__(config)
        self.config = config
        self.tokenizer = bart_tokenizer
        self.tokenizer.add_tokens([" <arg>"," <tgr>", " <evt>"])

        self.transformer = bart
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = self.config.vocab_size = len(self.tokenizer)
        self.register_buffer(
            "final_logits_bias", 
            torch.zeros((1, self.transformer.shared.num_embeddings))
        )

        self.evt2sent, self.sent2evt, self.evt_vocab, self.evt_ids = get_event_names_dict()

    def remove_unseen(self, lm_logits, input_ids):
        '''
        Consideriamo solo i token visti ignorando quelli non visti. Ai logit
        dei token non visti viene assegnato un valore molto basso pari a -1000.
        Questo serve per evitare di generare token non presenti nell'input originale.
        '''
        # input_ids (batch, seq)
        seen_lm_logits = torch.full_like(lm_logits, fill_value=-1000).to(lm_logits.device) #(batch, seq, vocab)
        input_ids.to(lm_logits.device)
        
        seen_vocab = set(input_ids.reshape(-1).tolist())
        for i in range(self.transformer.vocab_size):
            if i in (seen_vocab):
                seen_lm_logits[:, :, i] = lm_logits[:, :, i]
        return seen_lm_logits 

    def forward(
        self, 
        input_ids, 
        attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None, # da qui parametri passati dal generatore 
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_embeds=None,
        training: bool = False # parametro per discriminare l'output di training da quello del generatore
    ):
        # Il decoder prende una sequenza a partire dal token speciale di
        # inizio. Poi il decoder predice il prossimo token nella sequenza.
        # Le etichette sono i token di input del decoder spostati di uno
        # a sinistra. In altre parole i token di input del decoder sono 
        # spostati di uno a destra rispetto alle etichette.
        labels = None
        if training:
            labels = decoder_input_ids[:, 1:].clone() 
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_attention_mask = decoder_attention_mask[:, :-1]
            
            # I token di padding devono essere sostituiti con -100:  
            # https://discuss.huggingface.co/t/is-there-a-way-to-return-the-decoder-input-ids-from-tokenizer-prepare-seq2seq-batch/2929/3
            labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask= decoder_attention_mask,
            use_cache=use_cache, 
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        decoder_output = outputs.last_hidden_state  # (batch, dec_seq_len, hidden_dim)
        encoder_output = outputs.encoder_last_hidden_state  # (batch, enc_seq_len, hidden_dim)
            
        logits = F.linear(
            decoder_output, 
            self.transformer.shared.weight, 
            bias=self.final_logits_bias
        )

        logits = self.remove_unseen(logits, self.evt_ids)

        if training:
            return logits, labels
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def get_encoder(self):
        return self.transformer.encoder

    def get_output_embeddings(self):
        # Questo metodo è necessario per la generazione.
        vocab_size, emb_size = self.transformer.shared.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = self.transformer.shared.weight.data
        return lin_layer 

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, input_embeds, encoder_input_ids, **kwargs):
        return {
            "input_ids": encoder_input_ids,  
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # cambia questo parametro per non usare caching
            "input_embeds": input_embeds,
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        '''
        Forza la generazione di uno dei token_ids mettendo le probabilità degli
        altri token a 0: (logprob=-float("inf")).
        '''
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    # Questa funzione viene chiamata solo in fase di test.
    # Riferimento: https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/generation_utils.py
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        '''
        Genera sequenza per modelli con un head di language modeling.
        Riferimenti:
        - https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529
        - https://huggingface.co/blog/how-to-generate
        '''
        
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # sovrascritto dal batch_size dell'input 
        else:
            batch_size = 1

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # Previene la duplicazione dell'output quando si effettua il greedy decoding.
        if do_sample is False:
            if num_beams == 1:
                # Condizioni della generazione greedy con no_beam_search.
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # Condizioni della generazione greeedy con beam_search.
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # Crea un'attention_mask se necessario.
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # Imposta il pad_token_in a eos_token_it se non impostato. Questo deve
        # essere fatto dopo la creazione dell'attention_mask. EOS = end of sentence.
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # Posizione corrente e dimensione del vocabolario.
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # Imposta effective batch size e effective batch multiplier secondo do_sample.
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # Verifica se il BOS token (beginning of sentence) può essere usato
                # per il decoder_start_token_id.
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            # Recupera l'encoder e memorizza gli output dell'encoder.
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            input_embeds = encoder.embed_tokens(input_ids)  * encoder.embed_scale 

        # Espande gli input ids se num_beams > 1 o num_return_sequences > 1.
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        encoder_input_ids = input_ids 

        if self.config.is_encoder_decoder:
            # Crea decoder_input_ids vuoti.
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            # Espande batch_idx per assegnare il corretto output dell'encoder
            # per gli input_ids espansi (dato che num_beams > 1 e num_return_sequences > 1).
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # Espande gli encoder_outputs.
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # Salva gli encoder_outputs in `model_kwargs`.
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["input_embeds"] = input_embeds
            model_kwargs["encoder_input_ids"] = encoder_input_ids

        else:
            cur_len = input_ids.shape[-1]

        output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        '''
        Genera sequenze per ogni esempio senza beam search (num_beams == 1).
        Tutte le sequenze restituite sono generate in modo indipendente.
        '''
        
        # Lunghezza delle frasi generate / frasi non finite.
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )
            
            # Qui viene chiamato il forward.
            outputs = self(**model_inputs, return_dict=True) 
   
            #outputs.logits (batch, seq_len, input_seq_len)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # Se il modello ha un passato, allora imposta la variabile past
            # per velocizzare il decoding.
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperatura (temperatura più alta => maggiore probabilità di campionare token con probabilità bassa).
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering.
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample.
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding.
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Aggiorna le generazioni e le frasi finite.
            if eos_token_id is not None:
                # Aggiunge padding alle sentenze finite se esiste eos_token_id.
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # Aggiunge token e incrementa la lunghezza di 1.
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # Se la frase non è finita e il token da aggiungere è eos, allora
                # sent_lengths è riempito con la lunghezza corrente.
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents viene impostato a 0 se c'è eos nella frase.
                unfinished_sents.mul_((~eos_in_sents).long())

            # Ferma quando c'è un </s> in ogni frase, o quando viene superata la lunghezza massima.
            if unfinished_sents.max() == 0:
                break

            # Estende l'attention_mask per il nuovo input generato se solo decoder.
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids
    
    # Funzione tratta da `generation_utils.py` della libreria Transformers. 
    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # Repetition penalty
        # Riferimento: https://arxiv.org/abs/1909.05858.
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # Imposta la probabilità del token eos a 0 se la lunghezza minima non
        # è stata raggiunta.
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        '''
        Repetition penalty
        Riferimento: https://arxiv.org/abs/1909.05858.
        '''
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # Se lo score < 0, allora repetition penalty deve essere moltiplicata
                # per ridurre la probabilità del token precedente.
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty



###############################################################################
# Modello BERT per gli eventi
###############################################################################

class EventBertModel(LightningModule):
    def __init__(
        self,
        bert,
        tokenizer,
        bart_tokenizer,
        #data_dir: str = "datasets/rams",
        data_dir: str = "/content/drive/MyDrive/history/datasets/rams",
        #ontology_dir: str = "datasets",
        ontology_dir: str = "/content/drive/MyDrive/history/datasets",
        num_events: int = 140, 
        dropout_rate: float = 0.4,
        bert_hidden_dims: int = 768, 
        hidden_dims: int = 150,
        max_span_length: int = 3,
        average: str = "weighted",
        freeze_bert: bool = True
    ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.bart_tokenizer.add_tokens([" <arg>", " <trg>"])
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir

        if freeze_bert:
            for param in self.bert.named_parameters():
                param[1].requires_grad=False

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(bert_hidden_dims, num_events)
        )

        self.idx2evt, self.evt2idx = self.load_event_dicts()

        self.average = average

        self.accuracy = Accuracy(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.f1 = F1(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.prec = Precision(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )
        self.recall = Recall(
            num_classes=num_events,
            ignore_index=0,
            average=average
        )

        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def load_event_dicts(self):
        _, data_dicts = load_rams_data(self.data_dir)

        return data_dicts["idx2evt"], data_dicts["evt2idx"]
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(self, input_ids, attention_mask, training: bool = False):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        out = output[1]
        logits = self.clf(out)
                  
        return logits
    
    def training_step(self, batch, batch_idx):
        
        #print("batch[spans_trg_true]: ", batch["spans_trg_true"])
        
        logits = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            training=True
        )
        
        '''
        evt_id = torch.argmax(logits, -1)
        evt_name = ""
        if evt_id == 0:
            evt_name = self.idx2evt[1 + torch.randint(139, (1,)).item()]
        else:
            evt_name = self.idx2evt[evt_id.item()]
        '''
        
        spans_trg_true = batch["spans_trg_true"]
        evt_id = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_id]
        
        loss_ce = CrossEntropyLoss()
        
        #print("docs_trg_true: ", docs_trg_true)
        #print("torch.argmax(logits, -1): ", torch.argmax(logits, -1))
        #print("logits.view(-1, logits.shape[-1]): ", logits.view(-1, logits.shape[-1]))
        #print("docs_trg_true.view(-1).long(): ", docs_trg_true.view(-1).long())

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )
        
        acc = self.accuracy(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())

        try:
            f1 = self.f1(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())
        except Exception as ex:
            print("Error in train f1, setting to 0")
            f1 = torch.tensor(0)
            
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "train_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            training=True
        )
        
        spans_trg_true = batch["spans_trg_true"]
        evt_id = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_id]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )
        
        acc = self.accuracy(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())

        try:
            f1 = self.f1(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())
        except Exception as ex:
            print("Error in valid f1, setting to 0")
            f1 = torch.tensor(0)
            
        self.log("valid_loss", loss, prog_bar=True)
        self.log(
            "valid_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "valid_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], 
            batch["attention_masks"], 
            training=True
        )
        
        spans_trg_true = batch["spans_trg_true"]
        evt_id = torch.nonzero(spans_trg_true)
        docs_trg_true = spans_trg_true[evt_id]
        
        loss_ce = CrossEntropyLoss()

        loss = loss_ce(
            logits.view(-1, logits.shape[-1]), 
            docs_trg_true.view(-1).long()
        )
        
        acc = self.accuracy(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())

        try:
            f1 = self.f1(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())
        except Exception as ex:
            print("Error in test f1, setting to 0")
            f1 = torch.tensor(0)
            
        self.log("test_loss", loss, prog_bar=True)
        self.log(
            "test_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        prec = self.prec(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())
        recall = self.recall(logits.view(-1, logits.shape[-1]), docs_trg_true.view(-1).long())
        
        self.log(
            "test_prec", prec, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "test_recall", recall, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        
        return (acc, f1, prec, recall)
    
    def test_epoch_end(self, outputs):
        save_dir = "/content/drive/MyDrive/history/results/event_bert"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_acc": self.accuracy.compute().item(),
            "final_f1": self.f1.compute().item(),
            "final_prec": self.prec.compute().item(),
            "final_recall": self.recall.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                "acc": tup[0].item(),  
                "f1": tup[1].item(),
                "prec": tup[2].item(),
                "recall": tup[3].item() 
            }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/event_bert_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)



###############################################################################
# Modello per gli argomenti
###############################################################################

class ArgumentModelWrapper(LightningModule):
    def __init__(self, bart, bart_tokenizer):
        super(ArgumentModelWrapper, self).__init__()
        self.model = ArgumentModel(
            BartConfig.from_pretrained("facebook/bart-base"),
            bart=bart,
            bart_tokenizer=bart_tokenizer
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(
        self, 
        encoder_input_ids, 
        encoder_attention_mask,
        dec_input_ids,
        dec_attention_mask,
        training: bool = False
    ):
        return self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=dec_attention_mask,
            training=training
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )
        
        '''
        Esempio:
        logits.view(-1, logits.shape[-1]).shape:  torch.Size([71, 50267])
        labels.view(-1).shape:  torch.Size([71])
        logits.view(-1, logits.shape[-1]):  tensor([[ 2.9516e+01,  6.1094e+00,  1.6625e+01,  ..., -1.0000e+03,
                -2.3145e+00,  1.9072e+00],
                [-2.8184e+00, -5.3984e+00,  4.0156e+00,  ..., -1.0000e+03,
                2.3262e+00,  3.6523e-01],
                [-4.9219e+00, -4.6016e+00,  2.0742e+00,  ..., -1.0000e+03,
                1.0312e+00, -1.2573e-01],
                ...,
                [-5.1602e+00, -3.9414e+00,  8.0703e+00,  ..., -1.0000e+03,
                -3.4607e-02,  3.7427e-01],
                [-6.8867e+00, -4.7734e+00,  4.3711e+00,  ..., -1.0000e+03,
                1.4668e+00,  1.7297e-01],
                [-5.3242e+00, -4.6953e+00,  2.9473e+00,  ..., -1.0000e+03,
                2.0547e+00,  6.0400e-01]], device='cuda:0', dtype=torch.float16,
            grad_fn=<ViewBackward0>)
        labels.view(-1):  tensor([  831, 21498,    19, 21176,    11,  1854,    32,  2061,    15,  7550,
                59, 50265,  5674,    23,   382,   317,     2,  -100,  -100,  -100,
                -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                -100], device='cuda:0')
        '''
        
        #print("logits.view(-1, logits.shape[-1]).shape: ", logits.view(-1, logits.shape[-1]).shape)
        #print("labels.view(-1).shape: ", labels.view(-1).shape)
        #print("logits.view(-1, logits.shape[-1]): ", logits.view(-1, logits.shape[-1]))
        #print("labels.view(-1): ", labels.view(-1))

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(logits.view(-1, logits.shape[-1]), labels.view(-1))

        loss = torch.mean(loss)
    
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"],
            training=True
        )

        loss_ce = CrossEntropyLoss()

        loss = loss_ce(logits.view(-1, logits.shape[-1]), labels.view(-1))

        loss = torch.mean(loss)
        self.log("valid_loss", loss, prog_bar=True)
    
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels = self(
            batch["encoder_input_ids"], 
            batch["encoder_attention_mask"],
            batch["dec_input_ids"],
            batch["dec_attention_mask"]
        )

        # Esempio:
        # filled_templates:  tensor([[    2,  8241,   848,  8241,   634, 50265, 10320,    50, 50265,  1131,
        #    696,    23, 50265,   317,     2]], device='cuda:0')
        filled_templates = self.model.generate(
            batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"], 
            do_sample=True, 
            top_k=20, 
            top_p=0.95, 
            max_length=30, 
            num_return_sequences=1,
            num_beams=1,
            repetition_penalty=1
        )

        return (
            batch["doc_keys"], 
            filled_templates, 
            batch["dec_input_ids"]
        )

    def test_epoch_end(self, outputs):
        base_dir = "/content/drive/MyDrive/history/results/arguments"
        os.makedirs(base_dir, exist_ok=True)

        with open(
            f"{base_dir}/predictions.jsonl",
            "w"
        ) as writer:
            for tup in outputs:
                pred = {
                    "doc_key": tup[0],
                    "predicted": self.model.tokenizer.decode(
                        tup[1].squeeze(0), 
                        skip_special_tokens=True
                    ),
                    "gold": self.model.tokenizer.decode(
                        tup[2].squeeze(0), 
                        skip_special_tokens=True
                    ) 
                }
                writer.write(json.dumps(pred)+'\n')

        return {} 


class ArgumentModel(PreTrainedModel):
    '''
    Code adapted from the paper: 
    Li S., Ji H., Han J., Document-level event argument extraction by 
    conditional generation. In: Proceedings of the 2021 Conference of the 
    North American Chapter of the Association for Computational Linguistics: 
    Human Language Technologies, 2021, pp. 894-908.
    https://github.com/raspberryice/gen-arg/
    '''

    def __init__(self, config, bart, bart_tokenizer):
        super(ArgumentModel, self).__init__(config)
        self.config = config
        self.tokenizer = bart_tokenizer
        self.tokenizer.add_tokens([" <arg>"," <tgr>"])

        self.transformer = bart
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = self.config.vocab_size = len(self.tokenizer)

    def forward(
        self, 
        input_ids, 
        attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None, # da qui parametri passati dal generatore
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_embeds=None,
        training: bool = False # parametro per discriminare l'output di training da quello del generatore
    ):
        # Esempio input_ids decodificato:
        # ['<s>', ' <arg>', 'Ġkilled', ' <arg>', 'Ġusing', ' <arg>', 'Ġinstrument', 'Ġor', 
        # ' <arg>', 'Ġmedical', 'Ġissue', 'Ġat', ' <arg>', 'Ġplace', '</s>', '</s>', 
        # 'ĠTransportation', 'Ġofficials', 'Ġare', 'Ġurging', 'Ġcar', 'pool', 'Ġand', 
        # 'Ġtele', 'working', 'Ġas', 'Ġoptions', 'Ġto', 'Ġcombat', 'Ġan', 'Ġexpected', 
        # 'Ġflood', 'Ġof', 'Ġdrivers', 'Ġon', 'Ġthe', 'Ġroad', 'Ġ.', 'Ġ(', 'ĠPaul', 'ĠDug', 'gan', 'Ġ)', 
        # 'Ġ--', 'ĠA', 'ĠBaltimore', 'Ġprosecutor', 'Ġaccused', 'Ġa', 'Ġpolice', 'Ġdetective', 'Ġof', 'Ġ"', 
        # 'Ġsabot', 'aging', 'Ġ"', 'Ġinvestigations', 'Ġrelated', 'Ġto', 'Ġthe', 'Ġdeath', 'Ġof', 'ĠFreddie', 
        # 'ĠGray', 'Ġ,', 'Ġaccusing', 'Ġhim', 'Ġof', 'Ġfabric', 'ating', 'Ġnotes', 'Ġto', 'Ġsuggest', 'Ġthat', 
        # 'Ġthe', 'Ġstate', "Ġ'", 's', 'Ġmedical', 'Ġexaminer', 'Ġbelieved', 'Ġthe', 'Ġmanner', 'Ġof', 'Ġdeath', 
        # 'Ġwas', 'Ġan', 'Ġaccident', 'Ġrather', 'Ġthan', 'Ġa', ' <tgr>', 'Ġhomicide', ' <tgr>', 'Ġ.', 'ĠThe', 
        # 'Ġheated', 'Ġexchange', 'Ġcame', 'Ġin', 'Ġthe', 'Ġchaotic', 'Ġsixth', 'Ġday', 'Ġof', 'Ġthe', 'Ġtrial', 
        # 'Ġof', 'ĠBaltimore', 'ĠOfficer', 'ĠCaesar', 'ĠGood', 'son', 'ĠJr', '.', 'Ġ,', 'Ġwho', 'Ġdrove', 'Ġthe', 
        # 'Ġpolice', 'Ġvan', 'Ġin', 'Ġwhich', 'ĠGray', 'Ġsuffered', 'Ġa', 'Ġfatal', 'Ġspine', 'Ġinjury', 'Ġin', 
        # 'Ġ2015', 'Ġ.', 'Ġ(', 'ĠDerek', 'ĠHawkins', 'Ġand', 'ĠLyn', 'h', 'ĠB', 'ui', 'Ġ)', '</s>', '<pad>', ...]
        
        # Esempio input del decoder decodificato:
        # ['<s>', 'ĠOfficer', 'ĠCaesar', 'ĠGood', 'son', 'ĠJr', '.', 'Ġkilled', 'ĠFreddie', 
        # 'ĠGray', 'Ġusing', ' <arg>', 'Ġinstrument', 'Ġor', ' <arg>', 'Ġmedical', 'Ġissue', 'Ġat', 'ĠBaltimore', 'Ġplace', '</s>', '<pad>', ...]
        
   
        # Il decoder prende una sequenza a partire dal token special di
        # inizio. Poi il decoder predice il prossimo token nella sequenza.
        # Le etichette sono i token di input del decoder spostati di uno
        # a sinistra. In altre parole i token di input del decoder sono 
        # spostati di uno a destra rispetto alle etichette.
        labels = None
        if training:
            # Questi sono gli id dell'input del decoder.
            labels = decoder_input_ids[:, 1:].clone()
            # Questi sono sempre gli id dell'input del decoder, tranne il primo token (che aveva id 0). 
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_attention_mask = decoder_attention_mask[:, :-1]

            # I token di padding devono essere sostituiti con -100:  
            # https://discuss.huggingface.co/t/is-there-a-way-to-return-the-decoder-input-ids-from-tokenizer-prepare-seq2seq-batch/2929/3
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        # Questi sono gli output del modello Seq2Seq.
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask= decoder_attention_mask,
            use_cache=use_cache, 
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Questi sono embedding.
        decoder_output = outputs.last_hidden_state  # (batch, dec_seq_len, hidden_dim)
        encoder_output = outputs.encoder_last_hidden_state  # (batch, enc_seq_len, hidden_dim)
        
        # Prendiamo solo gli embedding dei token di input per limitare
        # le probabilità del vocabolario a quei token.
        # In questo modo genereremo solo parole che sono state passate come
        # input all'encoder.
        if input_embeds == None:  # input_embeds è passato dal generatore.  
            input_tokens_emb = self.transformer.encoder.embed_tokens(input_ids) * self.transformer.encoder.embed_scale  # (batch, enc_seq_len, hidden_dim)
        else:
            input_tokens_emb = input_embeds

        # Calcola la formula dall'articolo:
        # h_i^T Emb(w)
        # Dobbiamo farlo per gli elementi nel batch.
        # Questo è sostanzialmente un Batch Matrix Multiplication (BMM).
        # Per ogni batch abbiamo la probabilità di ogni parola nell'input
        # dell'encoder (enc_seq_len) di essere generata dal decoder (dec_seq_len).   
        prod = torch.einsum(
            "bij,bjk->bik", 
            decoder_output, 
            torch.transpose(input_tokens_emb, 1, 2)
        )  # (batch, dec_seq_len, enc_seq_len)

        batch_size = prod.size(0)
        dec_seq_len = prod.size(1)
        logits = torch.full(
            (
                batch_size, 
                dec_seq_len, 
                self.transformer.config.vocab_size
            ), 
            fill_value=-1000,
            dtype=prod.dtype
        ).to(prod.device)

        # Possibili indici duplicati.
        index = input_ids.unsqueeze(dim=1).expand_as(prod)
        
        # Con dim=2 questo è equivalente a:
        # self[i][j][index[i][j][k]] = src[i][j][k]
        logits.scatter_(dim=2, index=index, src=prod)

        if training:
            return logits, labels
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def get_encoder(self):
        return self.transformer.encoder

    def get_output_embeddings(self):
        # Questo metodo è necessario per la generazione.
        vocab_size, emb_size = self.transformer.shared.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = self.transformer.shared.weight.data
        return lin_layer 

    def prepare_inputs_for_generation(self, decoder_input_ids, past, 
                                      attention_mask, use_cache, 
                                      encoder_outputs, input_embeds, 
                                      encoder_input_ids, **kwargs):
        return {
            "input_ids": encoder_input_ids,  
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # cambia questo parametro per non usare caching
            "input_embeds": input_embeds,
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        '''
        Forza la generazione di uno dei token_ids mettendo le probabilità degli
        altri token a 0: (logprob=-float("inf")).
        '''
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    # Questa funzione viene chiamata solo in fase di test.
    # Riferimento: https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/generation_utils.py
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        '''
        Genera sequenza per modelli con un head di language modeling.
        Riferimenti:
        - https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529
        - https://huggingface.co/blog/how-to-generate
        '''

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # sovrascritto dal batch_size dell'input 
        else:
            batch_size = 1

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # Previene la duplicazione dell'output quando si effettua il greedy decoding.
        if do_sample is False:
            if num_beams == 1:
                # Condizioni della generazione greedy con no_beam_search.
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # Condizioni della generazione greedy con beam_search.
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # Crea un'attention_mask se necessario.
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # Imposta il pad_token_in a eos_token_it se non impostato. Questo deve
        # essere fatto dopo la creazione dell'attention_mask.
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # Posizione corrente e dimensione del vocabolario.
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size
        # Esempio: vocab_size:  50267.
        #print("generate vocab_size: ", vocab_size)

        # Imposta effective batch size e effective batch multiplier secondo do_sample.
        # Qui il batch è quello impostato inizialmente nel modello.
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1
        # Esempio con batch_size = 1:
        # effective_batch_size:  1
        # effective_batch_mult:  1
        #print("generate effective_batch_size: ", effective_batch_size)
        #print("generate effective_batch_mult: ", effective_batch_mult)

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # Verifica se il BOS token (beginning of sentence) può essere usato
                # per il decoder_start_token_id.
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            # Recupera l'encoder e memorizza gli output dell'encoder.
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            input_embeds = encoder.embed_tokens(input_ids)  * encoder.embed_scale 

        # Espande gli input ids se num_beams > 1 o num_return_sequences > 1.
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        encoder_input_ids = input_ids 

        if self.config.is_encoder_decoder:
            # Crea decoder_input_ids vuoti.
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            # Espande batch_idx per assegnare il corretto output dell'encoder
            # per gli input_ids espansi (dato che num_beams > 1 e num_return_sequences > 1).
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            
            # Esempio: expanded_batch_idxs:  tensor([0], device='cuda:0').
            #print("generate expanded_batch_idxs: ", expanded_batch_idxs)

            # Espande gli encoder_outputs.
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # Salva gli encoder_outputs in `model_kwargs`.
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["input_embeds"] = input_embeds
            model_kwargs["encoder_input_ids"] = encoder_input_ids

        else:
            cur_len = input_ids.shape[-1]

        output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        '''
        Genera sequenze per ogni esempio senza beam search (num_beams == 1).
        Tutte le sequenze restituite sono generate in modo indipendente.
        '''
        
        # Lunghezza delle frasi generate / frasi non finite.
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        # Esempio: 
        # unfinished_sents:  tensor([1], device='cuda:0')
        # sent_lengths:  tensor([30], device='cuda:0')
        #print("_generate_no_beam_search unfinished_sents: ", unfinished_sents)
        #print("_generate_no_beam_search sent_lengths: ", sent_lengths)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )
            
            # Qui viene chiamato il forward.
            outputs = self(**model_inputs, return_dict=True) 
    
            #outputs.logits (batch, seq_len, input_seq_len)
            next_token_logits = outputs.logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # Se il modello ha un passato, allora imposta la variabile past
            # per velocizzare il decoding.
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            if do_sample:
                # Temperatura (temperatura più alta => maggiore probabilità di campionare token con probabilità bassa).
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering.
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample.
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding.
                next_token = torch.argmax(next_token_logits, dim=-1)
            # Esempio:
            # next_token:  tensor([37], device='cuda:0')
            #print("_generate_no_beam_search next_token: ", next_token)

            # Aggiorna le generazioni e le frasi finite.
            if eos_token_id is not None:
                # Aggiunge padding alle sentenze finite se esiste eos_token_id.
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            # Esempio:
            # tokens_to_add:  tensor([37], device='cuda:0')
            #print("_generate_no_beam_search tokens_to_add: ", tokens_to_add)

            # Aggiunge token e incrementa la lunghezza di 1.
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # Se la frase non è finita e il token da aggiungere è eos, allora
                # sent_lengths è riempito con la lunghezza corrente.
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents viene impostato a 0 se c'è eos nella frase.
                unfinished_sents.mul_((~eos_in_sents).long())
                # Quando eos_in_sents è True, allora ci si ferma. In questo caso unfinished_sents ha valore 0.
                # Altrimenti ha valore 1.
                # Esempio:
                # eos_in_sents:  tensor([False], device='cuda:0')
                # is_sents_unfinished_and_token_to_add_is_eos:  tensor([False], device='cuda:0')
                # sent_lengths:  tensor([30], device='cuda:0')
                # unfinished_sents:  tensor([1], device='cuda:0')
                #print("_generate_no_beam_search eos_in_sents: ", eos_in_sents)
                #print("_generate_no_beam_search is_sents_unfinished_and_token_to_add_is_eos: ", is_sents_unfinished_and_token_to_add_is_eos)
                #print("_generate_no_beam_search sent_lengths: ", sent_lengths)
                #print("_generate_no_beam_search unfinished_sents: ", unfinished_sents)

            # Ferma quando c'è un </s> in ogni frase, o quando viene superata la lunghezza massima.
            if unfinished_sents.max() == 0:
                break

            # Estende l'attention_mask per il nuovo input generato se solo decoder.
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids
    
    # Funzione tratta da `generation_utils.py` della libreria Transformers. 
    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # Repetition penalty
        # Riferimento: https://arxiv.org/abs/1909.05858.
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # Imposta la probabilità del token eos a 0 se la lunghezza minima non
        # è stata raggiunta.
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        '''
        Repetition penalty
        Riferimento: https://arxiv.org/abs/1909.05858.
        '''
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # Se lo score < 0, allora repetition penalty deve essere moltiplicata
                # per ridurre la probabilità del token precedente.
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

