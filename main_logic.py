import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['WANDB_DISABLED'] = 'true'

CONFIG = {
    'seed': 42,
    'model_name': 'neuralmind/bert-base-portuguese-cased',
    'max_len': 300,
    'batch_size': 8,
    'accum_steps': 4,           
    'epochs': 3,
    'lr': 2e-5,
    'repo_url': 'https://github.com/roneysco/Fake.br-Corpus.git',
    'base_path': './Fake.br-Corpus/full_texts',
    'fake_weight': 10.0,        
    'decision_threshold': 0.40  # 20% de chance já considera Fake
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 Acelerador: GPU NVIDIA ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  AVISO: Rodando em CPU. O treinamento será extremamente lento.")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

def carregar_dados():

    if not os.path.exists(CONFIG['base_path']):
        print(f"📡 Clonando repositório Fake.br-Corpus...")
        os.system(f"git clone --depth 1 {CONFIG['repo_url']} Fake.br-Corpus")
    else:
        print(f"💾 Dataset encontrado em cache local.")

    data = []
    labels_map = {'true': 0, 'fake': 1} # 0=Real, 1=Fake

    for label_name, label_idx in labels_map.items():
        folder = os.path.join(CONFIG['base_path'], label_name)
        if not os.path.exists(folder): continue

        files = [f for f in os.listdir(folder) if f.endswith('.txt')]

        for file in tqdm(files, desc=f"Lendo {label_name}"):
            try:
                path = os.path.join(folder, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                txt = re.sub(r'\n+', ' ', txt).strip()
                if len(txt) > 50:
                    data.append({'text': txt, 'label': label_idx})
            except Exception:
                continue

    return pd.DataFrame(data)

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

        encoding = self.tokenizer.encode_plus(
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

def treinar(model, loader, optim, scheduler, device, total_len):
    model.train()
    total_loss = 0

    loss_weights = torch.tensor([1.0, CONFIG['fake_weight']]).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights)

    optim.zero_grad()

    for step, d in enumerate(tqdm(loader, desc="Treinando (Gradient Accumulation)")):
        input_ids = d["input_ids"].to(device)
        mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask)

        # Cálculo manual da Loss Ponderada
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))

        loss = loss / CONFIG['accum_steps']
        loss.backward()

        total_loss += loss.item()

        if (step + 1) % CONFIG['accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

    return total_loss / len(loader)

def avaliar_paranoico(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    threshold = CONFIG['decision_threshold']

    with torch.no_grad():
        for d in loader:
            input_ids = d["input_ids"].to(device)
            mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=mask)

            probs = F.softmax(outputs.logits, dim=1)[:, 1]

            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def inferencia_single(texto, model, tokenizer, device):
    inputs = tokenizer(
        texto, return_tensors="pt", truncation=True,
        max_length=CONFIG['max_len'], padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
        prob_fake = F.softmax(out.logits, dim=1)[0][1].item()

    is_fake = prob_fake > CONFIG['decision_threshold']

    print("\n" + "="*60)
    print(f"👁️  SENTINELA DIGITAL (Limiar: {CONFIG['decision_threshold']:.0%})")
    print("-" * 60)

    if is_fake:
        status = "⛔ BLOQUEADO (Risco Detectado)"
        color = "\033[91m"
    else:
        status = "✅ LIBERADO"
        color = "\033[92m"

    print(f"{color}{status}\033[0m")
    print(f"Probabilidade de Fake: {prob_fake:.4f}")

    if prob_fake > 0.1 and not is_fake:
        print("⚠️  Aviso: Contém traços leves de desinformação.")
    print("="*60 + "\n")


def main():
    print("🛡️ Inicializando Protocolo de Detecção Paranoico")

    df = carregar_dados()
    if len(df) == 0: return print("❌ Erro: Dados não encontrados.")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=CONFIG['seed'])

    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=2)
    model.to(DEVICE)

    train_ds = FakeNewsDataset(train_df.text.values, train_df.label.values, tokenizer, CONFIG['max_len'])
    val_ds = FakeNewsDataset(val_df.text.values, val_df.label.values, tokenizer, CONFIG['max_len'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * CONFIG['epochs'] // CONFIG['accum_steps'])

    print(f"\n🔥 Treinando com Weighted Loss (Fakes têm peso {CONFIG['fake_weight']}x)")

    for epoch in range(CONFIG['epochs']):
        loss = treinar(model, train_loader, optimizer, scheduler, DEVICE, len(train_ds))
        y_true, y_pred = avaliar_paranoico(model, val_loader, DEVICE)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Recall (Fakes Pegas): {recall:.1%} | Precision: {precision:.1%}")

    print("\n" + "="*50)
    print("MATRIZ DE CONFUSÃO FINAL")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Reais (Liberadas): {cm[0][0]} | Reais (Bloqueadas/FP): {cm[0][1]}")
    print(f"Fakes (Passaram/FN): {cm[1][0]} | Fakes (Bloqueadas): {cm[1][1]}")
    print(f"⚠️  Fakes que enganaram o sistema: {cm[1][0]}")
    print("="*50)

    while True:
        txt = input("\n📝 Cole a notícia (ou 'q'): ")
        if txt.lower() == 'q': break
        inferencia_single(txt, model, tokenizer, DEVICE)

if __name__ == '__main__':
    main()