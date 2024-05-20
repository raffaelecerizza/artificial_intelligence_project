import json
import os
import torch
import traceback

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from torch import nn
from torch.nn import (
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Dropout,
    Linear, LSTM
)
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, F1, Precision, Recall
from transformers import (
    BertModel
)


###############################################################################
# Data Loader
###############################################################################

class WikiDataModule(LightningDataModule):

    def __init__(
        self, data_dir: str = "/content/drive/MyDrive/history/datasets/wiki",
        batch_size: int = 128, 
        num_workers: int = 0, 
        num_tokens_labels: int = 5
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = 512
        # Le label dei token sono dei numeri che rappresentano il formato BIO.
        # Per questo le label possono assumere 5 valori: 0, 1, 2, 3, -100.
        self.num_tokens_labels = num_tokens_labels

        self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}

    def data_load_util(self, data_split: str):
        return {
            "input_ids": torch.load(f"{self.data_dir}/{data_split}/input_ids.pkl"),
            "attention_mask": torch.load(f"{self.data_dir}/{data_split}/attention_mask.pkl"),
            "tokens_labels": torch.load(f"{self.data_dir}/{data_split}/tokens_labels.pkl"),
            "labels": torch.load(f"{self.data_dir}/{data_split}/labels.pkl"),
            "tag2idx": torch.load(f"{self.data_dir}/{data_split}/tag2idx.pkl"),
            "idx2tag": torch.load(f"{self.data_dir}/{data_split}/idx2tag.pkl")   
        }

    def prepare_data(self):
        self.train_dict = self.data_load_util("train")
        self.valid_dict = self.data_load_util("valid")
        self.test_dict = self.data_load_util("test")

    def setup(self, stage=None):

        # Assegna i dataset di train/val per essere usati nel DataLoader. 
        if stage == "fit" or stage is None:
            pass

        # Assegna il dataset di test per essere usato nel DataLoader.
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        wiki_train = TensorDataset(
            torch.tensor(self.train_dict["input_ids"]), 
            torch.tensor(self.train_dict["attention_mask"]),
            torch.tensor(self.train_dict["tokens_labels"]),
            torch.tensor(self.train_dict["labels"], dtype=torch.float32)
        )

        return DataLoader(
            wiki_train, batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        wiki_valid = TensorDataset(
            torch.tensor(self.valid_dict["input_ids"]), 
            torch.tensor(self.valid_dict["attention_mask"]),
            torch.tensor(self.valid_dict["tokens_labels"]),
            torch.tensor(self.valid_dict["labels"], dtype=torch.float32)
        )

        return DataLoader(
            wiki_valid, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        wiki_test = TensorDataset(
            torch.tensor(self.test_dict["input_ids"]), 
            torch.tensor(self.test_dict["attention_mask"]),
            torch.tensor(self.test_dict["tokens_labels"]),
            torch.tensor(self.test_dict["labels"], dtype=torch.float32)
        )
        

        return DataLoader(
            wiki_test, batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class FreezeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch <= 3:
            print(f"Epoch number {epoch}, freezing base.")
            pl_module.freeze_base()
        else:
            print(f"Epoch number {epoch}, unfreezing base.")
            pl_module.unfreeze_base()


###############################################################################
# Multi-task Loss
###############################################################################

class MultiTaskLoss(nn.Module):
    """
    Liebel L., Korner M., Auxiliary Tasks in Multi-task Learning. 
    In: arXiv:1805.06334v2 [cs.CV], 2018.
    Riferimento:
    - https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
    """

    def __init__(self, losses_num: int = 2, num_tokens_labels: int = 5):
        super(MultiTaskLoss, self).__init__()
        self.losses_num = losses_num
        self.num_tokens_labels = num_tokens_labels
        #self.log_vars = nn.Parameter(torch.zeros((losses_num)))
        params = torch.ones(losses_num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, seq_clf_out, tokens_clf_out, labels, tokens_labels, attention_mask):

        loss_ce = CrossEntropyLoss(ignore_index=-100)
        loss_bce = BCELoss()

        # Se attention_mask non è None, allora calcoliamo la cross-entropy loss 
        # solo per i logits e le label dei token rilevanti.
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = tokens_clf_out.view(-1, self.num_tokens_labels)
            active_labels = torch.where(
                active_loss, 
                tokens_labels.view(-1), 
                torch.tensor(loss_ce.ignore_index).type_as(tokens_labels)
            )
            loss0 = loss_ce(active_logits, active_labels)
        else:
            loss0 = loss_ce(
                tokens_clf_out.view(-1, self.num_tokens_labels), 
                tokens_labels.view(-1)
            )

        loss1 = loss_bce(seq_clf_out.view(-1), labels.view(-1))

        losses = [loss0, loss1]
        
        '''
        Altro metodo per calcolare la multi-task loss:
        Kendall A., Gal Y., Cipolla R., Multi-Task Learning Using Uncertainty to 
        Weigh Losses for Scene Geometry and Semantics. In: arXiv:1705.07115v3 [cs.CV],
        2017.
        Possibile implementazione alternativa:
        - https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb 
        
        loss = sum(
            torch.exp(-self.log_vars[i]) * losses[i] + (self.log_vars[i] / 2)
            for i in range(self.losses_num)
        )
        '''
        
        loss = sum(
            0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            for i, loss in enumerate(losses)
        )
        
        return loss

    
###############################################################################
# Modello Multi-task Learning
###############################################################################

class MultiTaskLearningModel(LightningModule):
    def __init__(
        self, base_model = None, 
        dropout_rate: float = 0.1, 
        hidden_size: int = 768, 
        num_tokens_labels: int = 5,
        average: str = "weighted"
    ):
        super(MultiTaskLearningModel, self).__init__()
        if base_model is None:
            self.base_model = BertModel.from_pretrained("bert-base-cased")
        else:
            self.base_model = base_model

        self.num_tokens_labels = num_tokens_labels

        # Questa componente è usata per la classificazione dei paragrafi
        # in storici o non storici.  
        self.seq_clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=768, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Sigmoid()
        )

        # Questa componente è usata per etichettare i token dei paragrafi. 
        # Riceve come input l'ultimo hidden layer del modello BERT
        # e lo passa attraverso un layer lineare per assegnare le etichette
        # alle entità menzionate. 
        # Qui i tag sono solo 5:
        #   - O: token non etichettato;
        #   - B-hist: token iniziale di un'entità storica;
        #   - B-not-hist: token iniziale di un'entità non storica;
        #   - I-hist: token non iniziale di un'entità storica;
        #   - I-not-hist: token non iniziale di un'entità non storica.
        # Si tratta dell'annotazione nel formato BIO.
        self.tokens_clf = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_tokens_labels)
        )

        self.average = average

        self.multi_loss = MultiTaskLoss(2, num_tokens_labels)
        
        # Per la classificazione binaria occorre impostare num_classes=1.
        # Riferimento:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5705
        self.seqc_accuracy = Accuracy(
            num_classes=1,
            average=self.average
        )
        self.tokc_accuracy = Accuracy(
            num_classes=num_tokens_labels,
            ignore_index=(num_tokens_labels - 1), # ignora non-entità
            average=self.average
        )
        self.seqc_f1 = F1(
            num_classes=1,
            average=self.average
        )
        self.tokc_f1 = F1(
            num_classes=num_tokens_labels, 
            ignore_index=(num_tokens_labels - 1), # ignora non-entità
            average=self.average
        )
        self.seqc_prec = Precision(
            num_classes=1,
            average=self.average
        )
        self.seqc_recall = Recall(
            num_classes=1,
            average=self.average
        )
        
        for param in self.seq_clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for param in self.tokens_clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = output.last_hidden_state

        seq_clf_out = self.seq_clf(torch.mean(last_hidden_state, dim=1))
        tokens_clf_out = self.tokens_clf(last_hidden_state)

        return seq_clf_out, tokens_clf_out

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)
        
        # Alcune metriche richiedono solo valori positivi. Quindi sostituiamo
        # i -100 con l'ultima label dei token e facciamo in modo che la metrica
        # ignori lo score per questa label.
        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1, 
            tokens_labels
        )
        
        loss = self.multi_loss(seqc_out, tokc_out, labels, tokens_labels, attention_mask)

        tokc_preds = torch.argmax(tokc_out, -1)
        
        seqc_acc = self.seqc_accuracy(seqc_out.view(-1), labels.int().view(-1))
        tokc_acc = self.tokc_accuracy(tokc_preds, tokens_labels)

        try:
            #seqc_f1 = self.seqc_f1(seqc_out, labels.int())
            seqc_f1 = self.seqc_f1(seqc_out.view(-1), labels.int().view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in train seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in train tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)
        loss = self.multi_loss(seqc_out, tokc_out, labels, tokens_labels, attention_mask)

        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1, 
            tokens_labels
        )

        tokc_preds = torch.argmax(tokc_out, -1)

        seqc_acc = self.seqc_accuracy(seqc_out.view(-1), labels.int().view(-1))
        tokc_acc = self.tokc_accuracy(tokc_preds, tokens_labels)

        try:
            #seqc_f1 = self.seqc_f1(seqc_out, labels.int())
            seqc_f1 = self.seqc_f1(seqc_out.view(-1), labels.int().view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in val seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in val tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, tokens_labels, labels = batch
        seqc_out, tokc_out = self(input_ids, attention_mask)

        tokens_labels = torch.where(
            tokens_labels == -100,
            self.num_tokens_labels - 1,
            tokens_labels
        )

        tokc_preds = torch.argmax(tokc_out, -1)

        seqc_acc = self.seqc_accuracy(seqc_out.view(-1), labels.int().view(-1))
        tokc_acc = self.tokc_accuracy(tokc_preds, tokens_labels)

        try:
            #seqc_f1 = self.seqc_f1(seqc_out, labels.int())
            seqc_f1 = self.seqc_f1(seqc_out.view(-1), labels.int().view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in test seqc_f1, setting to 0")
            seqc_f1 = torch.tensor(0)

        try:    
            tokc_f1 = self.tokc_f1(tokc_preds.view(-1), tokens_labels.view(-1))
        except Exception as ex:
            traceback.print_exc()
            print("Error in test tokc_f1, setting to 0")
            tokc_f1 = torch.tensor(0)
            
        #seqc_prec = self.seqc_prec(seqc_out, labels.int())
        #seqc_recall = self.seqc_recall(seqc_out, labels.int())
        seqc_prec = self.seqc_prec(seqc_out.view(-1), labels.int().view(-1))
        seqc_recall = self.seqc_recall(seqc_out.view(-1), labels.int().view(-1))    

        self.log("test_seqc_acc", seqc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_tokc_acc", tokc_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_seqc_f1", seqc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_tokc_f1", tokc_f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_seqc_prec", seqc_prec, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_seqc_recall", seqc_recall, prog_bar=True, on_step=True, on_epoch=True)
        
        return seqc_acc, tokc_acc, seqc_f1, tokc_f1, seqc_prec, seqc_recall

    def test_epoch_end(self, outputs):
        save_dir = "/content/drive/MyDrive/history/results/document_classification"
        os.makedirs(save_dir, exist_ok=True)

        res = {
            "final_seqc_acc": self.seqc_accuracy.compute().item(),
            "final_tokc_acc": self.tokc_accuracy.compute().item(),
            "final_seqc_f1": self.seqc_f1.compute().item(),
            "final_tokc_f1": self.tokc_f1.compute().item(),
            "final_seqc_prec": self.seqc_prec.compute().item(),
            "final_seqc_recall": self.seqc_recall.compute().item(),
            "step_results": []
        }

        for tup in outputs:
            pred = {
                    "seqc_acc": tup[0].item(), 
                    "tokc_acc": tup[1].item(), 
                    "seqc_f1": tup[2].item(), 
                    "tokc_f1": tup[3].item(),
                    "seqc_prec": tup[4].item(),
                    "seqc_recall": tup[5].item()
                }
            res["step_results"].append(pred)

        with open(
            f"{save_dir}/mtl_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)
            
    def predict_paragraph_class(self, input_ids, attention_mask):
        seqc_out, tokc_out = self(input_ids, attention_mask)
        #print("seqc_out: ", seqc_out)
        return seqc_out[0]
        
    def freeze_base(self):
        # Qui viene effettuato il freeze dei soli pesi del modello base
        # di BERT. I pesi del classificatore (clf) vengono sempre addestrati. 
        for param in self.base_model.named_parameters():
            param[1].requires_grad=False
        print("Base frozen.")

    def unfreeze_base(self):
        for param in self.base_model.named_parameters():
            param[1].requires_grad=True
        print("Base unfrozen.")


###############################################################################
# Modello BERT per la classificazione dei paragrafi
###############################################################################

class Bert_clf(LightningModule):
    def __init__(
        self,
        dropout_rate=0.2, 
        hidden_dim=128,
        freeze_bert=False,
        average="weighted"
    ):
        super(Bert_clf, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")

        if freeze_bert:
            for param in self.bert.named_parameters():
                param[1].requires_grad=False

        self.dropout1 = Dropout(
            p=dropout_rate
        )
        self.clf = nn.Linear(768, 1)
        
        for param in self.clf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.average = average

        # Per la classificazione binaria occorre impostare num_classes=1.
        # Riferimento:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5705
        self.accuracy = Accuracy(
            num_classes=1,
            average=self.average
        )
        self.f1 = F1(
            num_classes=1,
            average=self.average
        )
        self.prec = Precision(
            num_classes=1,
            average=self.average
        )
        self.recall = Recall(
            num_classes=1,
            average=self.average
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=5e-5, 
            eps=1e-8
        )
        return optimizer

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        out = output[1]
        out = self.dropout1(out)
        logits = self.clf(out)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(logits.view(-1), labels)

        acc = self.accuracy(logits.view(-1), labels.long())

        try:
            f1 = self.f1(logits.view(-1), labels.long())
        except Exception as ex:
            traceback.print_exc()
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
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        loss_ce = BCEWithLogitsLoss()

        loss = loss_ce(logits.view(-1), labels)

        acc = self.accuracy(logits.view(-1), labels.long())

        try:
            f1 = self.f1(logits.view(-1), labels.long())
        except Exception as ex:
            traceback.print_exc()
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
        input_ids, attention_mask, _, labels = batch
        logits = self(input_ids, attention_mask)

        acc = self.accuracy(logits.view(-1), labels.long())

        try:
            f1 = self.f1(logits.view(-1), labels.long())
        except Exception as ex:
            traceback.print_exc()
            print("Error in test f1, setting to 0")
            f1 = torch.tensor(0)

        self.log(
            "test_acc", acc, prog_bar=True, 
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1", f1, prog_bar=True, 
            on_step=True, on_epoch=True
        )

        prec = self.prec(logits.view(-1), labels.long())
        recall = self.recall(logits.view(-1), labels.long())
        
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
        save_dir = "/content/drive/MyDrive/history/results/document_classification"
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
            f"{save_dir}/bert_clf_results.json", "w", 
            encoding="utf-8"
        ) as f:
            json.dump(res, f, indent=4)
