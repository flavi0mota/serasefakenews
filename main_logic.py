import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # <--- Correção: AdamW nativo do PyTorch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split # <--- Correção: Import que faltava
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Configurações de Engenharia
warnings.filterwarnings("ignore")
os.environ['WANDB_DISABLED'] = 'true'

CONFIG = {
    'seed': 42,
    'model_name': 'neuralmind/bert-base-portuguese-cased',
    'max_len': 300,
    'batch_size': 16,  # Se der erro de memória (CUDA OOM), baixe para 8
    'epochs': 3,       # 3 épocas costumam ser suficientes para fine-tuning
    'lr': 2e-5,
    'patience': 2,
    'heuristic_weight': 0.35,  # Peso da "Lógica Simbólica" (35%)
    'repo_url': 'https://github.com/roneysco/Fake.br-Corpus.git',
    'base_path': './Fake.br-Corpus/full_texts'
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 Acelerador: GPU NVIDIA ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  AVISO: Rodando em CPU. O treinamento será lento.")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==============================================================================
# 2. MOTOR DE HEURÍSTICA (A "LÓGICA SIMBÓLICA")
# ==============================================================================
class ExtratorDeSintomas:
    """
    Analisa a estrutura sintática (semântica superficial) para
    corrigir ou validar a rede neural.
    """
    def __init__(self):
        self.palavras_alarmistas = [
            'urgente', 'cuidado', 'atenção', 'morte', 'secreto', 'revelado',
            'compartilhe', 'acabou', 'fim', 'bomba', 'vazou', 'pânico', 'gravíssimo'
        ]
        self.indicadores_credibilidade = [
            'segundo', 'informou', 'afirmou', 'relatório', 'estudo', 'pesquisa',
            'oficial', 'disse', 'conforme', 'nota', 'documento', 'publicado em'
        ]

    def analisar(self, texto):
        texto_lower = texto.lower()
        score = 0.0

        # A. Análise de Caixa Alta (Grito Digital)
        chars_validos = [c for c in texto if c.isalpha()]
        if len(chars_validos) > 0:
            uppers = sum(1 for c in chars_validos if c.isupper())
            ratio = uppers / len(chars_validos)
            if ratio > 0.4: score += 0.45
            elif ratio > 0.2: score += 0.25

        # B. Vocabulário de Pânico
        count_alarme = sum(1 for p in self.palavras_alarmistas if p in texto_lower)
        score += min(count_alarme * 0.1, 0.4)

        # C. Pontuação Emocional
        if '!!' in texto or '??' in texto: score += 0.2

        # D. Fatores de Credibilidade (Redutores de score fake)
        count_cred = sum(1 for p in self.indicadores_credibilidade if p in texto_lower)
        score -= min(count_cred * 0.08, 0.5)

        # E. Formatação de Notícia Real (Datas e Citações)
        if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', texto): score -= 0.15
        if '"' in texto or '“' in texto: score -= 0.1

        return max(min(score, 1.0), -1.0)

def carregar_dados():
    if not os.path.exists(CONFIG['base_path']):
        print(f"📡 Clonando repositório Fake.br-Corpus...")
        os.system(f"git clone --depth 1 {CONFIG['repo_url']} Fake.br-Corpus")

    data = []
    labels_map = {'true': 0, 'fake': 1} # 0=Real, 1=Fake

    print("📂 Carregando arquivos...")
    for label_name, label_idx in labels_map.items():
        folder = os.path.join(CONFIG['base_path'], label_name)
        if not os.path.exists(folder): continue

        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        # Limitando para acelerar teste se necessário (remova o [:limit] para full)
        for file in tqdm(files, desc=f"Lendo {label_name}"):
            try:
                path = os.path.join(folder, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                # Limpeza básica de quebras de linha
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
        self.heuristica = ExtratorDeSintomas()

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        h_score = self.heuristica.analisar(text)

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
            'heuristic_score': torch.tensor(h_score, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def treinar(model, loader, optim, scheduler, device, total_len):
    model.train()
    losses = []
    correct = 0
    for d in tqdm(loader, desc="Treinando"):
        input_ids = d["input_ids"].to(device)
        mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        optim.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        
        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        losses.append(loss.item())
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()

    return correct / total_len, np.mean(losses)

def avaliar(model, loader, device, total_len):
    model.eval()
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for d in loader:
            input_ids = d["input_ids"].to(device)
            mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            h_scores = d["heuristic_score"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=mask)
            
            # --- FUSÃO HÍBRIDA ---
            probs = F.softmax(outputs.logits, dim=1)
            prob_fake_bert = probs[:, 1]
            
            # Ajuste linear baseado na heurística
            final_prob_fake = prob_fake_bert + (h_scores * CONFIG['heuristic_weight'])
            preds = (final_prob_fake > 0.5).long()

            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total_len, all_labels, all_preds

def analisar_noticia_interativa(texto, model, tokenizer, device):
    """Módulo de inferência individual para uso humano"""
    print(f"\n🔍 Processando análise...")

    heuristica = ExtratorDeSintomas()
    h_score = heuristica.analisar(texto)

    # 2. BERT
    inputs = tokenizer(
        texto, return_tensors="pt", truncation=True, 
        max_length=CONFIG['max_len'], padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
        probs = F.softmax(out.logits, dim=1)
        prob_fake = probs[0][1].item()

    final_score = prob_fake + (h_score * CONFIG['heuristic_weight'])
    final_prob = max(0.0, min(1.0, final_score)) # Clamp 0-1
    is_fake = final_prob > 0.5

    bar = "█" * int(final_prob * 30) + "░" * (30 - int(final_prob * 30))
    
    print("\n" + "="*60)
    print(f"📊 RELATÓRIO DE INTELIGÊNCIA ARTIFICIAL")
    print("="*60)
    if is_fake:
        print(f"🚨 CLASSIFICAÇÃO: FAKE NEWS")
        print(f"📉 Probabilidade de Fraude: {final_prob:.2%} [{bar}]")
    else:
        print(f"✅ CLASSIFICAÇÃO: NOTÍCIA REAL")
        print(f"🛡️ Índice de Credibilidade: {(1-final_prob):.2%} [{bar}]")
    
    print("-" * 60)
    print("⚙️  Caixa Preta (Debug):")
    print(f"   • Confiança Neural (BERT): {prob_fake:.4f}")
    print(f"   • Ajuste Simbólico (Regras): {h_score:+.4f}")
    print("="*60 + "\n")

def main():
    print("🧠 Iniciando Sistema Híbrido de Detecção de Fake News")
    
    # 1. Dados
    df = carregar_dados()
    if len(df) == 0: return print("❌ Erro: Nenhum dado carregado.")

    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=CONFIG['seed'])
    print(f"✅ Dataset: {len(train_df)} treino | {len(val_df)} validação")

    # 2. Setup Modelo
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=2)
    model.to(DEVICE)

    # 3. Dataloaders
    train_ds = FakeNewsDataset(train_df.text.values, train_df.label.values, tokenizer, CONFIG['max_len'])
    val_ds = FakeNewsDataset(val_df.text.values, val_df.label.values, tokenizer, CONFIG['max_len'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    # 4. Otimização (AdamW nativo PyTorch)
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*CONFIG['epochs'])

    # 5. Loop de Treino
    best_acc = 0
    print("\n🔥 Iniciando Fine-Tuning...")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nÉpoca {epoch+1}/{CONFIG['epochs']}")
        train_acc, train_loss = treinar(model, train_loader, optimizer, scheduler, DEVICE, len(train_ds))
        val_acc, y_true, y_pred = avaliar(model, val_loader, DEVICE, len(val_ds))
        
        print(f"   Treino: Loss {train_loss:.4f} | Acc {train_acc:.1%}")
        print(f"   Valid : Acc {val_acc:.1%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.bin')
            print("   💾 Checkpoint salvo!")

    # 6. Resultados Finais
    model.load_state_dict(torch.load('best_model.bin'))
    print("\n" + "="*50)
    print("MATRIZ DE CONFUSÃO (Validação)")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Reais Corretos: {cm[0][0]} | Fakes Corretos: {cm[1][1]}")
    print(f"Erros (FP+FN): {cm[0][1] + cm[1][0]}")
    print("="*50)

    # 7. Modo Interativo
    print("\nInferência")
    while True:
        try:
            txt = input("\n📝 Cole a notícia (ou 'sair'): ")
            if txt.lower() in ['sair', 'q']: break
            if len(txt) < 10: print("⚠️ Texto muito curto."); continue
            
            analisar_noticia_interativa(txt, model, tokenizer, DEVICE)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
