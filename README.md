# 🛡️ (Fake) || ~(Fake): Fake News com BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-BERT-yellow)
![Selenium](https://img.shields.io/badge/Selenium-Automation-green)

> Um sistema de Inteligência Artificial baseado em Deep Learning (Transfer Learning) para detecção automática de desinformação em Português, integrado ao WhatsApp e Telegram.

---

## Sobre o Projeto

Solução desenvolvida para combater a disseminação viral de notícias falsas. Diferente de abordagens baseadas em listas negras (blacklist) ou palavras-chave, este projeto utiliza o modelo de linguagem **BERT**.

## Arquitetura do Modelo

O núcleo do sistema utiliza o modelo pré-treinado `neuralmind/bert-base-portuguese-cased`.

1.  **Entrada:** Texto cru ou URL.
2.  **Tokenização:** `BertTokenizer` converte texto em IDs e Máscaras de Atenção.
3.  **Processamento:** O BERT gera embeddings contextuais (vetores densos).
4.  **Classificação:** Uma camada densa (Feed Forward) analisa o token `[CLS]` e gera a probabilidade (Softmax).
5.  **Saída:** `FAKE` (Prob > 0.40) ou `REAL`.

---

## Datasets Utilizados

O treinamento foi realizado com uma fusão balanceada (50/50) de dois corpus principais:

1.  **Fake.br-Corpus:** Notícias longas e artigos jornalísticos (Verdadeiros e Falsos).
2.  **FakeWhatsApp.Br:** Mensagens curtas e correntes virais de WhatsApp.

---

## Instalação e Configuração

### Pré-requisitos
* Python 3.8+
* Conta no Google Colab (recomendado para treino via GPU)
* Conta no Ngrok (para túnel no WhatsApp)

### 1. Clonar o Repositório

### 2. Instalar Dependências

```bash
pip install torch transformers pandas scikit-learn selenium webdriver-manager flask pyngrok newspaper3k python-telegram-bot nest_asyncio

```

## Como Usar

Executar o Bot (Escolha uma opção)

#### Opção A: Bot do Telegram (Mais Fácil)

1. Obtenha seu token com o `@BotFather`.
2. Cole o token no arquivo `bot_telegram.py`.
3. Execute:
```bash
python bot_telegram.py
```

#### Opção B: Bot do WhatsApp (Selenium + Ngrok)

Este método requer que o **Servidor (Colab)** e o **Cliente (Seu PC)** estejam conectados.

1. **No Colab:** Execute o script do servidor Flask + Ngrok. Copie a URL gerada (ex: `https://abcd.ngrok-free.app`).
2. **No seu PC:** Edite `bot_whatsapp_selenium.py` e cole a URL na variável `URL_CEREBRO`.
3. Execute no seu PC:
```bash
python bot_whatsapp_selenium.py
```


4. Escaneie o QR Code quando o Chrome abrir.


## Tecnologias

* [PyTorch](https://pytorch.org/) - Framework de Deep Learning.
* [Hugging Face Transformers](https://huggingface.co/) - Biblioteca SOTA para NLP.
* [Selenium](https://www.selenium.dev/) - Automação de Navegador Web.
* [Newspaper3k](https://newspaper.readthedocs.io/) - Extração de texto de artigos.

---

## Aviso Legal

Este projeto é uma ferramenta de auxílio e pesquisa acadêmica. Nenhum modelo de IA é 100% preciso. Sempre verifique informações críticas em fontes oficiais e agências de checagem (Lupa, Aos Fatos, E-Farsas).

**Licença:** MIT

```
