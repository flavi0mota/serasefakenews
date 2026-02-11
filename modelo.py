import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")
os.environ['WANDB_DISABLED'] = 'true'

#objeto de configuração
CONFIG = {
    'seed': 42,
    'model_name': 'neuralmind/bert-base-portuguese-cased',
    'max_len': 300,
    'batch_size': 8,
    'accum_steps': 4,
    'epochs': 4,       
    'lr': 2e-5,
    'repo_url': 'https://github.com/roneysco/Fake.br-Corpus.git',
    'base_path': './Fake.br-Corpus/full_texts',
    'fake_weight': 10.0,
    'decision_threshold': 0.40,
    'dropout_rate': 0.2, 
    'patience': 2        
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#shallow clone e depois sanitização básica para a rapaziada
def carregar_dados_blindados():
    data = []
    if not os.path.exists(CONFIG['base_path']):
        os.system(f"git clone --depth 1 https://github.com/roneysco/Fake.br-Corpus.git Fake.br-Corpus")

    path_whatsapp = './FakeWhatsApp.Br'
    if not os.path.exists(path_whatsapp):
        os.system(f"git clone --depth 1 https://github.com/cabrau/FakeWhatsApp.Br.git {path_whatsapp}")

    fontes_para_ler = [
        (os.path.join(CONFIG['base_path'], 'true'), 0),
        (os.path.join(CONFIG['base_path'], 'fake'), 1),
        (os.path.join(path_whatsapp, 'data', '2018', 'misinformation'), 1),
        (os.path.join(path_whatsapp, 'data', '2018', 'fact_checking'), 0)
    ]

    print("Carregando dados")
    for folder_path, label in fontes_para_ler:
        if not os.path.exists(folder_path): continue
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file in tqdm(files, leave=False):
            try:
                with open(os.path.join(folder_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                txt = re.sub(r'\n+', ' ', txt).strip()
                if len(txt) > 20: data.append({'text': txt, 'label': label})
            except: continue

    df = pd.DataFrame(data)
    if len(df) == 0: return df

    min_count = min(len(df[df['label']==0]), len(df[df['label']==1]))
    df_balanced = pd.concat([
        df[df['label']==0].sample(min_count, random_state=42),
        df[df['label']==1].sample(min_count, random_state=42)
    ]).sample(frac=1).reset_index(drop=True)

    print(f"Dataset Balanceado: {min_count} de cada classe.")
    return df_balanced

def treinar(model, loader, optim, scheduler, device, total_len):
    model.train()
    total_loss = 0
    loss_weights = torch.tensor([1.0, CONFIG['fake_weight']]).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights)

    optim.zero_grad()
    for step, d in enumerate(tqdm(loader, desc="Treino")):
        input_ids = d["input_ids"].to(device)
        mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask)
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
        loss = loss / CONFIG['accum_steps']
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % CONFIG['accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

    return total_loss / len(loader)

#Ativando a síndrome de "Os cara tá no teto" no bot
def avaliar_paranoico(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for d in loader:
            input_ids = d["input_ids"].to(device)
            mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=mask)

            loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
            total_val_loss += loss.item()

            probs = F.softmax(outputs.logits, dim=1)[:, 1]
            preds = (probs > CONFIG['decision_threshold']).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(loader)
    return np.array(all_labels), np.array(all_preds), avg_val_loss

def main():
    print("Treinamento Iniciado...")

    df = carregar_dados_blindados()
    if len(df) == 0: return

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=CONFIG['seed'])

    print(f"BERT com Dropout de {CONFIG['dropout_rate']*100}%...")

    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2,
        hidden_dropout_prob=CONFIG['dropout_rate'],         
        attention_probs_dropout_prob=CONFIG['dropout_rate'] 
    )
    model.to(DEVICE)

    train_ds = FakeNewsDataset(train_df.text.values, train_df.label.values, tokenizer, CONFIG['max_len'])
    val_ds = FakeNewsDataset(val_df.text.values, val_df.label.values, tokenizer, CONFIG['max_len'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*CONFIG['epochs']//CONFIG['accum_steps'])

    best_val_loss = float('inf')
    patience_counter = 0
    PASTA_FINAL = './modelo_whatsapp_pronto'

    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Época {epoch+1}/{CONFIG['epochs']} ---")

        train_loss = treinar(model, train_loader, optimizer, scheduler, DEVICE, len(train_ds))
        y_true, y_pred, val_loss = avaliar_paranoico(model, val_loader, DEVICE)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        print(f"📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"📊 Recall (Pega Fakes): {recall:.1%} | Precision: {precision:.1%}")

       
        if val_loss < best_val_loss:
            print(f"Melhoria por época({best_val_loss:.4f} -> {val_loss:.4f}). Salvando modelo...")
            best_val_loss = val_loss
            patience_counter = 0

            model.save_pretrained(PASTA_FINAL)
            tokenizer.save_pretrained(PASTA_FINAL)
        else:
            patience_counter += 1
            print(f"Sem melhoria relevante. Paciência: {patience_counter}/{CONFIG['patience']}")

            if patience_counter >= CONFIG['patience']:
                print("\nO modelo parou de aprender.")
                break

    print("\nTreinamento Finalizado. Modelo Salvo")

if __name__ == '__main__':
    main()