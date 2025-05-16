import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def build_vocab(smiles_list):
    chars = sorted(set(''.join(smiles_list)))
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def encode_smiles(smile, char2idx, max_len=120):
    encoded = [char2idx.get(c, 0) for c in smile[:max_len]]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded

class SMILESDataset(Dataset):
    def __init__(self, df, char2idx, max_len=120):
        self.smiles = df['SMILES'].values
        self.targets = df['IOT'].astype(np.float32).values
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x = encode_smiles(self.smiles[idx], self.char2idx, self.max_len)
        y = self.targets[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

class SMILESRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)

# Загрузка и подготовка данных
df = pd.read_csv("/workspace/TransformerVAE/descriptors_table.csv")
df["SMILES"] = df["SMILES"].str.strip()
df["IOT"] = df["IOT"].str.strip().str.replace(",", ".").astype(np.float32)

# Разделение на train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

char2idx, idx2char = build_vocab(df["SMILES"].values)
train_dataset = SMILESDataset(train_df, char2idx)
val_dataset = SMILESDataset(val_df, char2idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Модель и оптимизаторы
model = SMILESRegressor(vocab_size=len(char2idx))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Каталог для сохранения весов
os.makedirs("/workspace/TransformerVAE/induction_oxidation/checkpoints", exist_ok=True)
best_val_loss = float("inf")

# Обучение
for epoch in range(1, 110):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Валидация
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            val_preds.extend(preds.numpy())
            val_targets.extend(y_batch.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_loss = mean_squared_error(val_targets, val_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)

    print(f"Epoch {epoch}: Train Loss={total_train_loss:.4f} | Val MSE={val_loss:.4f} | Val MAE={val_mae:.4f}")

    # Сохранение весов после каждой эпохи
    torch.save(model.state_dict(), f"/workspace/TransformerVAE/induction_oxidation/checkpoints/model_epoch_{epoch}.pth")

    # Сохранение лучших весов
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/workspace/TransformerVAE/induction_oxidation/checkpoints/best_model.pth")
        print(f"✅ Best model updated at epoch {epoch}")
