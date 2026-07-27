import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

try:
    import pytorch_lightning as pl
    import torchmetrics
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError:
    pl = None
    torchmetrics = None


class CFG:
    seed = 42
    model_name = 'bert-base-uncased'
    num_classes = 21

    # Fixed: 5e-3 was ~100-500x too high for fine-tuning BERT and was
    # blowing up the pretrained weights, causing the model to collapse
    # to predicting the majority class. 2e-5 is a standard BERT fine-tuning LR.
    max_lr = 2e-5
    num_epochs = 40
    batch_size = 8
    drop_rate = 0.1
    embed_dim = 768
    n_fold = 10

    # Training configuration
    optimizer = 'Adam'
    pct_start = 0.1          # shorter warmup now that peak LR is reasonable
    div_factor = 10.0        # start closer to max_lr instead of 1000x below it
    final_div_factor = 100.0 # don't decay all the way to near-zero
    accum = 1
    base_dir = ""
    filename = 'policy_all_in_one_filter_purged.csv'
    DEBUG = False
    max_length = 512


if pl is not None:
    seed_everything(CFG.seed)


def get_next_run_dir(base_dir='trainingLogs'):
    os.makedirs(base_dir, exist_ok=True)
    existing_versions = [
        int(entry[len('version'):]) 
        for entry in os.listdir(base_dir) 
        if entry.startswith('version') and entry[len('version'):].isdigit()
    ]
    next_version = max(existing_versions, default=-1) + 1
    run_dir = os.path.join(base_dir, f'version{next_version}')
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


class PrivacyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Call tokenizer object directly (replaces deprecated encode_plus)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


if pl is not None:
    class LitPrivacy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(CFG.model_name)
            self.dropout = nn.Dropout(CFG.drop_rate)
            self.classifier = nn.Linear(CFG.embed_dim, CFG.num_classes)
            
            self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=CFG.num_classes)
            self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=CFG.num_classes)
            self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=CFG.num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.save_hyperparameters()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output 
            output = self.dropout(pooled_output)
            return self.classifier(output)

        def configure_optimizers(self):
            initial_lr = CFG.max_lr / CFG.div_factor
            optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr) 
            steps_per_epoch = math.ceil(CFG.steps_per_epoch / CFG.accum)

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                epochs=CFG.num_epochs,
                steps_per_epoch=steps_per_epoch,
                max_lr=CFG.max_lr,
                pct_start=CFG.pct_start,
                div_factor=CFG.div_factor,
                final_div_factor=CFG.final_div_factor
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            
            output = self(input_ids, attention_mask)
            loss = self.criterion(output, label)
            score = self.train_metric(output.argmax(dim=1), label)
            
            self.log_dict({'train_loss': loss, 'train_acc': score}, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            
            output = self(input_ids, attention_mask)
            loss = self.criterion(output, label)
            score = self.valid_metric(output.argmax(dim=1), label)
            
            self.log_dict({'valid_loss': loss, 'valid_acc': score}, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            
            output = self(input_ids, attention_mask)
            loss = self.criterion(output, label)
            score = self.test_metric(output.argmax(dim=1), label)
            
            self.log_dict({'test_loss': loss, 'test_acc': score}, on_step=False, on_epoch=True, prog_bar=True)
            return loss


if __name__ == "__main__":
    if pl is None:
        raise ModuleNotFoundError("Training requires pytorch_lightning, torchmetrics, and sklearn")

    # Read data
    df_all = pd.read_csv(os.path.join(CFG.base_dir, CFG.filename))
    df_all.dropna(inplace=True)

    if CFG.DEBUG:
        df_all = df_all[:200]
        CFG.num_epochs = 10

    texts_all = df_all["review"].values
    labels_all = df_all["label"].values
    
    tokenizer = BertTokenizer.from_pretrained(CFG.model_name)

    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    fold_indices = [valid_idx for _, valid_idx in skf.split(texts_all, labels_all)]
    run_dir = get_next_run_dir()

    # Optimal worker setting to prevent DataLoader warnings/freezing
    num_workers = min(2, os.cpu_count() or 1)

    for test_fold in range(CFG.n_fold):
        valid_fold = (test_fold + 1) % CFG.n_fold
        train_folds = [i for i in range(CFG.n_fold) if i not in (test_fold, valid_fold)]

        train_idx = np.concatenate([fold_indices[i] for i in train_folds])
        valid_idx = fold_indices[valid_fold]
        test_idx = fold_indices[test_fold]

        print(f"\n--- Running Fold {test_fold + 1}/{CFG.n_fold} (train=8, tune={valid_fold + 1}, test={test_fold + 1}) ---")

        train_dataset = PrivacyDataset(texts_all[train_idx], labels_all[train_idx], tokenizer, CFG.max_length)
        valid_dataset = PrivacyDataset(texts_all[valid_idx], labels_all[valid_idx], tokenizer, CFG.max_length)
        test_dataset = PrivacyDataset(texts_all[test_idx], labels_all[test_idx], tokenizer, CFG.max_length)

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

        CFG.steps_per_epoch = len(train_loader)

        lit_model = LitPrivacy()

        logger = CSVLogger(save_dir=run_dir, name='logs', version=f'fold{test_fold}')
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(run_dir, 'checkpoints', f'fold{test_fold}'),
            monitor='valid_loss',
            save_top_k=1,
            save_last=True,
            filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
            mode='min'
        )

        early_stop_callback = EarlyStopping(
            monitor='valid_loss',
            patience=5,      # stop after 5 epochs with no improvement
            mode='min',
            verbose=True
        )

        trainer = Trainer(
            max_epochs=CFG.num_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            accumulate_grad_batches=CFG.accum,
            callbacks=[checkpoint_callback,early_stop_callback],
            logger=logger,
        )

        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(lit_model, dataloaders=test_loader, ckpt_path='best')