import logging
import torch
import torch.nn.functional as F
import nest_asyncio 
import asyncio

nest_asyncio.apply()

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from transformers import BertTokenizer, BertForSequenceClassification
from newspaper import Article

TOKEN_TELEGRAM = "[REDACTED]" 
PASTA_MODELO = './modelo_whatsapp_pronto' 

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⏳ Carregando Cérebro em {DEVICE}...")

try:
    tokenizer = BertTokenizer.from_pretrained(PASTA_MODELO)
    model = BertForSequenceClassification.from_pretrained(PASTA_MODELO)
    model.to(DEVICE)
    model.eval()
    print("✅ Modelo Carregado!")
except Exception as e:
    print(f"ERRO: Pasta '{PASTA_MODELO}' não encontrada.")
    raise e

def extrair_texto_link(url):
    try:
        article = Article(url, language='pt')
        article.download()
        article.parse()
        return f"{article.title}. {article.text}"
    except: return None

def analisar_com_bert(texto):
    inputs = tokenizer(
        texto, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding='max_length'
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    
    prob_fake = probs[0][1].item()
    is_fake = prob_fake > 0.40 
    return is_fake, prob_fake

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name
    await update.message.reply_text(
        f"Olá, {user}! 🛡️\nSou o Detector de Fake News da matéria de P2.\nMande uma notícia e eu verifico para você."
    )

async def processar_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto_usuario = update.message.text
    if not texto_usuario: return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    texto_final = texto_usuario
    eh_link = False
    
    if 'http' in texto_usuario:
        msg_temp = await update.message.reply_text("Lendo link...")
        texto_extraido = extrair_texto_link(texto_usuario)
        if texto_extraido:
            texto_final = texto_extraido
            eh_link = True
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg_temp.message_id)
        else:
            await update.message.reply_text("Erro ao ler site.")
            return

    if len(texto_final) < 20:
        await update.message.reply_text("Texto muito curto.")
        return

    is_fake, confianca = analisar_com_bert(texto_final)
    
    if is_fake:
        resp = f"🚨 <b>FAKE NEWS DETECTADA</b>\nChance: {confianca:.1%}. RECOMENDO QUE NÃO COMPARTILHE"
    else:
        resp = f"✅ <b>NOTÍCIA REAL</b>\nConfiança: {(1-confianca):.1%}."

    await update.message.reply_text(resp, parse_mode='HTML')

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN_TELEGRAM).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), processar_mensagem))
    
    print("Bot rodando")
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(application.run_polling())
        else:
            application.run_polling()
    except RuntimeError:
        application.run_polling()
