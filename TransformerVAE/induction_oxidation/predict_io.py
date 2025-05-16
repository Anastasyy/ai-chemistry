import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ⚙️ Модель
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

# 🔠 Кодировка
def build_vocab(smiles_list):
    chars = sorted(set(''.join(smiles_list)))
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    return char2idx

def encode_smiles(smile, char2idx, max_len=120):
    encoded = [char2idx.get(c, 0) for c in smile[:max_len]]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded

# 📄 Загрузка SMILES из txt
def load_smiles_txt(txt_path):
    with open(txt_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list

# 🚀 Предсказание на всех SMILES
def run_batch_inference(txt_path="input.txt", data_path="descriptors_table.csv", model_path="checkpoints/best_model.pth", output_csv="output.csv"):
    # Загружаем базу для словаря
    df_vocab = pd.read_csv(data_path)
    df_vocab["SMILES"] = df_vocab["SMILES"].str.strip()
    char2idx = build_vocab(df_vocab["SMILES"].values)

    # Загружаем модель
    model = SMILESRegressor(vocab_size=len(char2idx))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Загружаем SMILES из txt
    smiles_list = load_smiles_txt(txt_path)

    # Предсказания
    predictions = []
    for smile in smiles_list:
        encoded = encode_smiles(smile, char2idx)
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        with torch.no_grad():
            pred = model(input_tensor).item()
        predictions.append(pred)

    # Создание и сохранение результата
    result_df = pd.DataFrame({
        "SMILES": smiles_list,
        "IOT": predictions
    })
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Предсказания сохранены в {output_csv}")

# 🔧 Запуск
if __name__ == "__main__":
    run_batch_inference(
        txt_path="/workspace/TransformerVAE/generation/results/final/smiles.txt",
        data_path="/workspace/TransformerVAE/descriptors_table.csv",
        model_path="/workspace/TransformerVAE/induction_oxidation/checkpoints/best_model.pth",
        output_csv="/workspace/res_final.csv"
    )
