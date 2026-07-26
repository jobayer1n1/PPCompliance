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
    from pytorch_lightning.callbacks import ModelCheckpoint
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError:
    pl = None
    torchmetrics = None


class CFG:
    seed = 42
    model_name = 'bert-base-uncased'
    num_classes = 21
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 80
    batch_size = 8
    accum = 1
    n_fold = 4
    base_dir = ""
    filename = 'policy_all_in_one_filter_purged.csv'
    embed_dim = 768
    hidden_dim = 768 * 2
    hidden_dim2 = 768 * 3
    drop_rate = 0.1  # <-- Added drop_rate here
    DEBUG = False

if pl is not None:
    seed_everything(CFG.seed)


# 1. Pre-extract embeddings function (Run once on GPU/CPU)
def extract_bert_embeddings(df, text_col='review', batch_size=32):
    """Pre-extracts [CLS] or pooled embeddings for the entire dataframe."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(CFG.model_name)
    bert_model = BertModel.from_pretrained(CFG.model_name).to(device)
    bert_model.eval()

    embeddings = []
    texts = df[text_col].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            # Using pooler_output ([CLS] vector passed through a dense layer)
            outputs = bert_model(**encoded)
            pooled_output = outputs.pooler_output.cpu().numpy()
            embeddings.append(pooled_output)

    return np.vstack(embeddings)


# 2. Picklable Dataset using static Tensors
class PrivacyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Classifier Model
class CustomEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(CFG.embed_dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.drop_rate), # <-- Added Dropout
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(CFG.drop_rate), # <-- Added Dropout
            nn.Linear(CFG.hidden_dim2, CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.drop_rate), # <-- Added Dropout
            nn.Linear(CFG.hidden_dim, CFG.num_classes)
        )

    def forward(self, x):
        return self.model(x)


if pl is not None:
    class LitPrivacy(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Separate metric instances for train/valid: sharing one
            # torchmetrics.Accuracy across both phases (including the
            # pre-training sanity-check validation pass) can cause
            # incorrect accumulated state between phases.
            self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=CFG.num_classes)
            self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=CFG.num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.save_hyperparameters(ignore=['model'])

        def forward(self, x, *args, **kwargs):
            return self.model(x)

        def configure_optimizers(self):
            # Initial LR is set to match what OneCycleLR will compute on its
            # first step (max_lr / div_factor) instead of an unrelated value
            # that would just get silently overridden.
            initial_lr = CFG.max_lr / CFG.div_factor
            optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)

            # OneCycleLR is stepped once per *optimizer* step. Under gradient
            # accumulation, optimizer steps happen once every `accum` batches,
            # so steps_per_epoch must reflect that -- not the raw number of
            # batches in the loader -- or the schedule will run out early /
            # raise a "Tried to step X times, but the total number of steps
            # is Y" error once accum > 1.
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
            embedding_res, label = batch
            output = self.model(embedding_res)
            loss = self.criterion(output, label)
            score = self.train_metric(output.argmax(dim=1), label)
            self.log_dict({'train_loss': loss, 'train_acc': score}, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            embedding_res, label = batch
            output = self.model(embedding_res)
            loss = self.criterion(output, label)
            score = self.valid_metric(output.argmax(dim=1), label)
            self.log_dict({'valid_loss': loss, 'valid_acc': score}, on_step=False, on_epoch=True, prog_bar=True)
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

    # Extract BERT features before running the training loop
    print("Extracting BERT embeddings...")
    embeddings_all = extract_bert_embeddings(df_all)
    labels_all = df_all["label"].values

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(embeddings_all, labels_all)):
        print(f"\n--- Running Fold {fold + 1}/{CFG.n_fold} ---")

        train_dataset = PrivacyDataset(embeddings_all[train_idx], labels_all[train_idx])
        valid_dataset = PrivacyDataset(embeddings_all[valid_idx], labels_all[valid_idx])

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        CFG.steps_per_epoch = len(train_loader)

        model = CustomEffNet()
        lit_model = LitPrivacy(model)

        # Explicit per-fold version instead of relying on CSVLogger's
        # auto-incrementing version_N, which doesn't tell you which fold
        # a given log directory belongs to and isn't stable across reruns.
        logger = CSVLogger(
            save_dir='logs/',
            name=CFG.model_name,
            version=f'fold_{fold}'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{CFG.model_name}/fold_{fold}',
            monitor='valid_loss',
            save_top_k=1,
            save_last=True,
            filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
            mode='min'
        )

        trainer = Trainer(
            max_epochs=CFG.num_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            accumulate_grad_batches=CFG.accum,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        # Breaking after first fold for standard training (remove break to run all folds)
        #break