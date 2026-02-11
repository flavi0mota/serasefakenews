import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


URL_CEREBRO = "https://unspiring-compulsively-danae.ngrok-free.dev/analisar" 


print("🚀 Iniciando Sentinela V4 (Correção Jsonify + Arquivados)...")

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--log-level=3") 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://web.whatsapp.com")

print("\nQR CODE")
time.sleep(40)
print("\nAguardando mensagens.")

def analisar_no_colab(texto):
    try:
        texto_limpo = texto.replace("\n", " ").strip()
        print(f"Enviando ao modelo: {texto_limpo[:30]}...")
        r = requests.post(URL_CEREBRO, json={"texto": texto_limpo}, timeout=10)
        
        if r.status_code == 200:
            return r.json()
        else:
            print(f"Erro HTTP do Colab: {r.status_code}")
            return None
    except Exception as e:
        print(f"Erro de conexão: {e}")
        return None

while True:
    try:
        conversa_para_clicar = None

        try:
            painel_chats = driver.find_element(By.ID, "pane-side")
            
            icones = painel_chats.find_elements(By.XPATH, ".//span[@aria-label='Unread']")
            
            if not icones:
                 icones = painel_chats.find_elements(By.XPATH, ".//span[contains(@aria-label, 'unread')]")

            if icones:
                conversa_para_clicar = icones[0]
                print("Mensagem detectada!")
        except:
            pass
        
        if conversa_para_clicar:
            conversa_para_clicar.click()
            time.sleep(2) 

            baloes = driver.find_elements(By.CSS_SELECTOR, "div.message-in")
            
            if baloes:
                ultimo = baloes[-1]
                texto_msg = ""
                
                try:
                    texto_msg = ultimo.find_element(By.CSS_SELECTOR, "span.copyable-text").text
                except:
                    try:
                        texto_msg = ultimo.find_element(By.CSS_SELECTOR, "span[dir='ltr']").text
                    except:
                        raw_text = ultimo.text
                        if len(raw_text) > 5:
                             texto_msg = raw_text.split("\n")[0]

                if texto_msg and len(texto_msg) > 1:
                    print(f"Lido: {texto_msg}")

                    resultado = analisar_no_colab(texto_msg)
                    
                    if resultado and "erro" not in resultado:
                        is_fake = resultado.get('is_fake')
                        prob = resultado.get('prob', 0)
                        
                        if is_fake:
                            resp = f"🚨 *IA:* Chance de ser FAKE: {prob:.1%}. Na dúvida, não compartilhe."
                        else:
                            resp = f"✅ *IA:* Notícia confiável."
                        
                        try:
                            caixa = driver.find_element(By.XPATH, '//div[@contenteditable="true"][@data-tab="10"]')
                            caixa.click()
                            time.sleep(0.5)
                            driver.execute_script("document.execCommand('insertText', false, arguments[0])", resp)
                            time.sleep(0.5)
                            caixa.send_keys(Keys.ENTER)
                            print("   🤖 Resposta enviada!")
                            
                            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                            
                        except Exception as e:
                            print(f"   ❌ Erro ao digitar: {e}")
            else:
                print("Aberto, porém texto não foi encontrado. Tentando pular...")
                try:
                     driver.find_element(By.ID, "pane-side").click()
                except: pass
            
            time.sleep(3)
        else:
            time.sleep(2)

    except Exception as e:
        time.sleep(3)
