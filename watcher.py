import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

COVALENT_API_KEY = os.getenv("COVALENT_API_KEY", "")
BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WATCH_ADDRESS = (os.getenv("WATCH_ADDRESS", "") or "").strip().lower()
INTERVAL_SECONDS = 60
MIN_USD_ALERT = float(os.getenv("MIN_USD_ALERT", "50"))

CHAINS = {
    1: "Ethereum",
    56: "BSC",
    137: "Polygon",
    42161: "Arbitrum",
    10: "Optimism",
    8453: "Base",
    43114: "Avalanche C-Chain",
    250: "Fantom",
}

if not COVALENT_API_KEY:
    raise SystemExit("Configure COVALENT_API_KEY no .env.")
if not WATCH_ADDRESS:
    raise SystemExit("Defina WATCH_ADDRESS (0x...) no .env para monitorar uma carteira.")
if not (BOT and CHAT_ID):
    print("⚠️ Sem TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID. Logs apenas no console.")

seen = {cid: set() for cid in CHAINS.keys()}

def covalent_txs(chain_id, addr, page=0, page_size=50):
    url = f"https://api.covalenthq.com/v1/{chain_id}/address/{addr}/transactions_v2/"
    params = {"key": COVALENT_API_KEY, "page-number": page, "page-size": page_size, "quote-currency": "USD"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", {})

def send_telegram(msg):
    if not (BOT and CHAT_ID):
        print("[NO-TG]", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
    except Exception as e:
        print("Erro ao enviar telegram:", e)

def format_tx(chain, tx):
    ts = tx.get("block_signed_at") or tx.get("timestamp")
    v = float(tx.get("value_quote") or 0.0)
    fees = float((tx.get("fees_paid_quote") or tx.get("gas_quote") or 0.0))
    h = tx.get("tx_hash") or tx.get("hash")
    frm = tx.get("from_address") or tx.get("from")
    to = tx.get("to_address") or tx.get("to")
    ok = tx.get("successful", True)
    status = "✅" if ok else "❌"
    direction = "OUT" if (frm or "").lower() == WATCH_ADDRESS else "IN" if (to or "").lower() == WATCH_ADDRESS else "OTHER"
    return h, (f"{status} <b>{chain}</b> • ${v:.2f}\n"
               f"<b>Dir:</b> {direction}\n"
               f"<b>From:</b> <code>{frm}</code>\n"
               f"<b>To:</b>   <code>{to}</code>\n"
               f"<b>Fee:</b> ${fees:.2f}\n"
               f"<b>Time:</b> {ts}\n"
               f"<b>Hash:</b> <code>{h}</code>")

def loop():
    print(f"Watcher iniciado para {WATCH_ADDRESS}. Intervalo: {INTERVAL_SECONDS}s. Mínimo: ${MIN_USD_ALERT:.2f}")
    while True:
        try:
            for cid, cname in CHAINS.items():
                try:
                    data = covalent_txs(cid, WATCH_ADDRESS, page=0, page_size=50)
                    txs = data.get("items") or []
                    for tx in txs:
                        v = float(tx.get("value_quote") or 0.0)
                        h, msg = format_tx(cname, tx)
                        if v >= MIN_USD_ALERT and h not in seen[cid]:
                            send_telegram(msg)
                            seen[cid].add(h)
                except Exception as e:
                    print(f"[{cname}] erro: {e}")
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("Encerrando watcher...")
            break

if __name__ == "__main__":
    loop()
