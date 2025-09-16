# app.py
import os
import re
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# ==========================
# CONFIG INICIAL
# ==========================
st.set_page_config(page_title="Track Wallet Multichain", page_icon="🧭", layout="wide")

APP_DIR = Path(__file__).parent
ENV_PATH = APP_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

COVALENT_API_KEY = (os.getenv("COVALENT_API_KEY") or "").strip()
if not COVALENT_API_KEY:
    st.sidebar.error(f"COVALENT_API_KEY não encontrada. Verifique {ENV_PATH}")
    st.stop()

HEADERS = {}

# ==========================
# CONSTANTES
# ==========================
EXPLORERS = {
    1:     "https://etherscan.io",
    56:    "https://bscscan.com",
    137:   "https://polygonscan.com",
    42161: "https://arbiscan.io",
    10:    "https://optimistic.etherscan.io",
    8453:  "https://basescan.org",
    43114: "https://snowtrace.io",
    250:   "https://ftmscan.com",
}
CHAINS = {
    1: {"name": "Ethereum"},
    56: {"name": "BSC"},
    137: {"name": "Polygon"},
    42161: {"name": "Arbitrum"},
    10: {"name": "Optimism"},
    8453: {"name": "Base"},
    43114: {"name": "Avalanche C-Chain"},
    250: {"name": "Fantom"},
}
DATE_PRESETS = {
    "Últimos 7 dias": 7,
    "Últimos 10 dias": 10,
    "Últimos 15 dias": 15,
    "Últimos 30 dias": 30,
    "Últimos 90 dias": 90,
    "Últimos 365 dias": 365,
    "Tudo (sem filtro de data)": None,
}
COINGECKO_PLATFORM = {
    1: "ethereum",
    56: "binance-smart-chain",
    137: "polygon-pos",
    42161: "arbitrum-one",
    10: "optimistic-ethereum",
    8453: "base",
    43114: "avalanche",
    250: "fantom",
}
DEX_CHAIN_SLUG = {
    1: "ethereum",
    56: "bsc",
    137: "polygon",
    42161: "arbitrum",
    10: "optimism",
    8453: "base",
    43114: "avalanche",
    250: "fantom",
}

# ==========================
# REGEX & FORMATADORES
# ==========================
_EVM_ADDR   = re.compile(r"^0x[a-fA-F0-9]{40}$")
_BTC_BECH32 = re.compile(r"^bc1[0-9ac-hj-np-z]{11,71}$", re.IGNORECASE)
_BTC_BASE58 = re.compile(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")
_BTC_TXID   = re.compile(r"^[A-Fa-f0-9]{64}$")

def is_evm_address(txt: str) -> bool:
    return bool(_EVM_ADDR.fullmatch((txt or "").strip()))

def is_btc_address(txt: str) -> bool:
    t = (txt or "").strip()
    return bool(_BTC_BECH32.fullmatch(t) or _BTC_BASE58.fullmatch(t))

def is_btc_txid(txt: str) -> bool:
    return bool(_BTC_TXID.fullmatch((txt or "").strip()))

def _human_iso(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts or ""

def epoch_to_human(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def fmt_usd(x: float, nd=2) -> str:
    try:
        return "$" + f"{float(x):,.{nd}f}"
    except Exception:
        return "$0.00"

def fmt_usd_precise(x: float, nd=6) -> str:
    try:
        s = f"{float(x):,.{nd}f}".rstrip("0").rstrip(".")
        return "$" + (s if s else "0")
    except Exception:
        return "$0"

def fmt_amt(x: float, nd=8) -> str:
    try:
        return f"{float(x):,.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "0"

def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"

def _since_dt(label: str):
    days = DATE_PRESETS[label]
    if days is None:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)

def to_checksum(addr: str) -> str:
    return (addr or "").strip()

# ==========================
# CACHES BÁSICOS
# ==========================
@st.cache_data(ttl=1800, show_spinner=False)
def _url_exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=8)
        if r.status_code == 405:
            r = requests.get(url, stream=True, timeout=8)
        return r.ok
    except Exception:
        return False

@st.cache_data(ttl=1800, show_spinner=False)
def get_logo_url(chain_id: int, contract_addr: str, covalent_logo: Optional[str], run_nonce: int = 0):
    if covalent_logo:
        return covalent_logo
    if contract_addr:
        tw_chain = {
            1: "ethereum",
            56: "smartchain",
            137: "polygon",
            42161: "arbitrum",
            10: "optimism",
            8453: "base",
            43114: "avalanchec",
            250: "fantom",
        }.get(chain_id)
        if tw_chain:
            tw_url = f"https://cdn.jsdelivr.net/gh/trustwallet/assets/blockchains/{tw_chain}/assets/{contract_addr}/logo.png"
            if _url_exists(tw_url):
                return tw_url
    return None

# ==========================
# EVM (Covalent)
# ==========================
@st.cache_data(ttl=180, show_spinner=False)
def get_balances(chain_id: int, addr: str, _nonce: int):
    url = f"https://api.covalenthq.com/v1/{chain_id}/address/{addr}/balances_v2/"
    params = {"key": COVALENT_API_KEY, "nft": False, "no-nft-fetch": True, "quote-currency": "USD"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("data", {})

@st.cache_data(ttl=180, show_spinner=False)
def get_transactions_pages(chain_id: int, addr: str, page_size: int, pages: int, _nonce: int):
    all_items = []
    url_v3 = f"https://api.covalenthq.com/v1/{chain_id}/address/{addr}/transactions_v3/"
    url_v2 = f"https://api.covalenthq.com/v1/{chain_id}/address/{addr}/transactions_v2/"
    for p in range(pages):
        params = {"key": COVALENT_API_KEY, "page-number": p, "page-size": page_size, "quote-currency": "USD"}
        r = requests.get(url_v3, params={**params, "no-logs": False}, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            r = requests.get(url_v2, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
        items = data.get("items") or data.get("transactions") or []
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size:
            break
    return all_items

def normalize_tokens(chain_id: int, data: dict) -> pd.DataFrame:
    rows = []
    for it in data.get("items", []):
        decimals = it.get("contract_decimals") or 0
        bal_raw = int(it.get("balance", "0") or 0)
        denom = 10 ** decimals if decimals > 0 else 1
        amount = bal_raw / denom
        rows.append({
            "chain": CHAINS[chain_id]["name"],
            "chain_id": chain_id,
            "logo": get_logo_url(chain_id, it.get("contract_address") or "", it.get("logo_url"), run_nonce=st.session_state.get("run_nonce", 0)),
            "token": it.get("contract_ticker_symbol") or "",
            "name": it.get("contract_name") or "",
            "token_address": it.get("contract_address") or "",
            "amount": amount,
            "price_usd": float(it.get("quote_rate") or 0.0),
            "value_usd": float((it.get("quote") or 0.0) or (amount * float(it.get("quote_rate") or 0.0))),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("value_usd", ascending=False, ignore_index=True)
    return df

def normalize_txs(chain_id: int, items: list, addr: str) -> pd.DataFrame:
    addr_lower = (addr or "").lower()
    rows = []
    for tx in items:
        rows.append({
            "chain": CHAINS[chain_id]["name"],
            "chain_id": chain_id,
            "hash": tx.get("tx_hash") or tx.get("hash"),
            "successful": tx.get("successful", True),
            "timestamp": tx.get("block_signed_at") or tx.get("timestamp"),
            "from": (tx.get("from_address") or tx.get("from") or ""),
            "to": (tx.get("to_address") or tx.get("to") or ""),
            "value_usd": float(tx.get("value_quote") or 0.0),
            "fees_usd": float((tx.get("fees_paid_quote") or tx.get("gas_quote") or 0.0)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["direction"] = df.apply(
            lambda r: "OUT" if r["from"].lower() == addr_lower else ("IN" if r["to"].lower() == addr_lower else "OTHER"),
            axis=1
        )
        df["direction_chip"] = df["direction"].map({"IN":"⬇️ IN","OUT":"⬆️ OUT"}).fillna("↔️ OTHER")
        df["timestamp_human"] = df["timestamp"].apply(_human_iso)
        df = df.sort_values("timestamp", ascending=False, ignore_index=True)
    return df

# ===== DeBank + Fallback =====
@st.cache_data(ttl=300, show_spinner=False)
def debank_token_list(address: str) -> pd.DataFrame:
    try:
        url = "https://api.debank.com/user/token_list"
        params = {"id": address, "is_all": "true"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if not r.ok:
            return pd.DataFrame()
        data = (r.json() or {}).get("data") or []
        rows = []
        for t in data:
            chain = t.get("chain") or t.get("chain_id")
            price = float(t.get("price") or 0)
            amount = float(t.get("amount") or 0)
            rows.append({
                "chain": chain,
                "logo": t.get("logo_url"),
                "token": (t.get("symbol") or "").upper(),
                "name": t.get("name") or "",
                "token_address": t.get("id") or t.get("address") or "",
                "amount": amount,
                "price_usd": price,
                "value_usd": price * amount,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("value_usd", ascending=False, ignore_index=True)
        return df
    except Exception:
        return pd.DataFrame()

def covalent_wallet_tokens_fallback(chain_ids: list[int], address: str) -> pd.DataFrame:
    all_dfs = []
    for cid in chain_ids:
        try:
            d = get_balances(cid, address, _nonce=st.session_state.get("run_nonce", 0))
            df = normalize_tokens(cid, d)
            if not df.empty:
                all_dfs.append(df)
        except Exception:
            pass
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ==========================
# BTC APIs
# ==========================
@st.cache_data(ttl=120, show_spinner=False)
def btc_address_summary(addr: str) -> Dict[str, Any]:
    r = requests.get(f"https://mempool.space/api/address/{addr}", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def btc_address_txs(addr: str, limit=50) -> List[Dict[str, Any]]:
    r = requests.get(f"https://mempool.space/api/address/{addr}/txs?limit={limit}", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def btc_tx_details(txid: str) -> Dict[str, Any]:
    r = requests.get(f"https://mempool.space/api/tx/{txid}", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def get_btc_price_usd() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": "bitcoin", "vs_currencies": "usd"},
                         timeout=10, headers={"User-Agent":"TrackWallet/1.0"})
        if r.ok:
            v = (r.json() or {}).get("bitcoin", {}).get("usd")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=10)
        if r.ok:
            v = float((r.json() or {}).get("data", {}).get("amount", 0))
            if v > 0:
                return v
    except Exception:
        pass
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=10)
        if r.ok:
            v = float((r.json() or {}).get("price", 0))
            if v > 0:
                return v
    except Exception:
        pass
    return 0.0

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .exec-btn button {
        width: 100% !important;
        background: linear-gradient(135deg, #0ea5e9, #22c55e) !important;
        color: #fff !important;
        border: 0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        letter-spacing: .2px !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 12px rgba(0,0,0,.25) !important;
        transition: transform .12s ease, filter .12s ease !important;
    }
    section[data-testid="stSidebar"] .exec-btn button:hover { transform: translateY(-1px) scale(1.01); }
    section[data-testid="stSidebar"] .exec-btn button:active { transform: translateY(0) scale(.99); }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.form("run_form", clear_on_submit=False):
    address = st.text_input("Carteira (0x... ou bc1/1/3... ou TXID BTC)", value=st.session_state.get("address", ""))
    st.markdown('<div class="exec-btn">', unsafe_allow_html=True)
    run = st.form_submit_button("▶️ Executar")
    st.markdown("</div>", unsafe_allow_html=True)

default_chains = [meta["name"] for meta in CHAINS.values()]
selected_chain_names = st.sidebar.multiselect(
    "Redes (EVM)", default_chains, default=st.session_state.get("sel_chains", default_chains)
)
preset_label = st.sidebar.selectbox(
    "Período para transações", list(DATE_PRESETS.keys()), index=st.session_state.get("preset_idx", 3)
)
min_usd = st.sidebar.number_input(
    "Filtro mínimo (USD) p/ tokens e txs", min_value=0.0, value=float(st.session_state.get("min_usd", 0.0)), step=10.0
)
page_size = st.sidebar.slider("Página (linhas) de transações (por requisição)", 10, 200, int(st.session_state.get("page_size", 50)), 10)
search_token = st.sidebar.text_input("🔍 Buscar token/nome (detalhamento)", value=st.session_state.get("search_token", ""))

st.session_state["address"] = address
st.session_state["sel_chains"] = selected_chain_names
st.session_state["preset_idx"] = list(DATE_PRESETS.keys()).index(preset_label)
st.session_state["min_usd"] = min_usd
st.session_state["page_size"] = page_size
st.session_state["search_token"] = search_token

selected_chain_ids = [cid for cid, meta in CHAINS.items() if meta["name"] in selected_chain_names]

# ==========================
# TÍTULO E ESTADO
# ==========================
st.title("🧭 Track Wallet Multichain (Single Wallet)")
st.caption("Cole a carteira (ou TXID BTC), clique em **Executar**, e use os filtros à esquerda.")

if "run_nonce" not in st.session_state:
    st.session_state["run_nonce"] = 0
if run:
    st.session_state["run_nonce"] += 1

show_wallet = bool(address.strip()) and st.session_state["run_nonce"] > 0
if not address.strip():
    st.info("Cole a carteira (0x... / bc1/1/3...) ou um TXID BTC e clique em **Executar** para ver o painel.")
elif st.session_state["run_nonce"] == 0:
    st.info("Clique em **Executar** para buscar os dados.")

# ==========================
# BTC DASHBOARD (MESMO LAYOUT)
# ==========================
def render_btc_dashboard(addr_or_txid: str):
    if is_btc_txid(addr_or_txid):
        st.subheader("📄 Detalhes da transação BTC")
        try:
            tx = btc_tx_details(addr_or_txid)
            st.json(tx)
        except Exception as e:
            st.error(f"Erro ao buscar TX: {e}")
        return

    # endereço
    try:
        info = btc_address_summary(addr_or_txid)
    except Exception as e:
        st.error(f"BTC: {e}")
        return

    btc_price = get_btc_price_usd()
    cs = info.get("chain_stats", {}) or {}
    ms = info.get("mempool_stats", {}) or {}
    confirmed = (cs.get("funded_txo_sum", 0) - cs.get("spent_txo_sum", 0))
    mempool = (ms.get("funded_txo_sum", 0) - ms.get("spent_txo_sum", 0))
    balance_btc = (confirmed + mempool) / 1e8
    tx_count = int(cs.get("tx_count", 0)) + int(ms.get("tx_count", 0))

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Valor total (USD)", fmt_usd(balance_btc * btc_price, 2))
    c2.metric("🪙 Nº de tokens", "1")
    c3.metric("🔗 Nº de redes", "1")

    # Distribuição por rede
    st.markdown("### Distribuição por rede")
    by_chain = pd.Series({"Bitcoin": balance_btc * btc_price})
    total = float(by_chain.sum())
    bdf = by_chain.reset_index().rename(columns={"index":"Rede", 0:"Valor (USD)"})
    bdf["%"] = (bdf["Valor (USD)"] / total * 100).round(2) if total > 0 else 0
    colA, colB = st.columns([0.45, 0.55])
    with colA:
        show_df = bdf.copy()
        show_df["Valor (USD)"] = show_df["Valor (USD)"].apply(fmt_usd)
        st.dataframe(
            show_df, use_container_width=True, hide_index=True,
            column_config={
                "Valor (USD)": st.column_config.TextColumn("Valor (USD)"),
                "%": st.column_config.NumberColumn("%", format="%.2f%%"),
            },
        )
    with colB:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            wedges, texts, autotexts = ax.pie(
                bdf["Valor (USD)"],
                labels=bdf["Rede"],
                startangle=140,
                autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                pctdistance=0.75,
                wedgeprops=dict(width=0.35),
                textprops={"fontsize": 9},
            )
            centre = plt.Circle((0, 0), 0.58, fc="white")
            ax.add_artist(centre)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)
        except Exception:
            st.info("Para ver o donut, instale:  `pip install matplotlib`")

    # Tabela de tokens
    st.subheader("💰 Tokens por Rede")
    df_tok = pd.DataFrame([{
        "Logo": "https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=029",
        "Ticker": "BTC",
        "Nome": "Bitcoin",
        "Contrato": "",
        "Quantidade": balance_btc,
        "Preço (USD)": btc_price,
        "Valor (USD)": balance_btc * btc_price,
        "Explorer": f"https://mempool.space/address/{addr_or_txid}",
    }])
    df_tok_fmt = df_tok.copy()
    df_tok_fmt["Quantidade"] = df_tok_fmt["Quantidade"].apply(lambda v: fmt_amt(v, 8))
    df_tok_fmt["Preço (USD)"] = df_tok_fmt["Preço (USD)"].apply(fmt_usd_precise)
    df_tok_fmt["Valor (USD)"] = df_tok_fmt["Valor (USD)"].apply(fmt_usd)
    st.dataframe(
        df_tok_fmt, use_container_width=True, hide_index=True,
        column_config={
            "Logo": st.column_config.ImageColumn(""),
            "Explorer": st.column_config.LinkColumn("Scan", display_text="Abrir"),
        },
    )

    # Transações
    st.subheader("🧾 Transações (BTC)")
    try:
        txs = btc_address_txs(addr_or_txid, limit=page_size)
    except Exception as e:
        st.warning(f"BTC TXs: {e}")
        txs = []
    rows = []
    for t in txs:
        txid = t.get("txid")
        bt = t.get("status", {}).get("block_time")
        rows.append({
            "Rede": "Bitcoin",
            "Tx": f"https://mempool.space/tx/{txid}" if txid else None,
            "Data/Hora": epoch_to_human(bt) if bt else "",
            "Direção": "", "De": "", "Para": "",
            "Valor (USD)": fmt_usd(0, 2), "Taxa (USD)": fmt_usd(0, 4),
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Tx": st.column_config.LinkColumn("Hash", display_text="Abrir"),
        },
    )

# ==========================
# PAINEL DA CARTEIRA (EVM + BTC)
# ==========================
if show_wallet:
    nonce = st.session_state["run_nonce"]
    since_dt = _since_dt(preset_label)

    entry = address.strip()

    # --- BTC?
    if is_btc_address(entry) or is_btc_txid(entry):
        render_btc_dashboard(entry)
    # --- EVM?
    elif is_evm_address(entry):
        with st.spinner("Buscando dados nas redes selecionadas..."):
            tokens_all, txs_all = [], []
            for cid in selected_chain_ids:
                try:
                    bd = get_balances(cid, entry, nonce)
                    dft = normalize_tokens(cid, bd)
                    if not dft.empty:
                        tokens_all.append(dft)
                except Exception as e:
                    st.warning(f"[{CHAINS[cid]['name']}] Tokens: {e}")

                try:
                    items = get_transactions_pages(cid, entry, page_size=page_size, pages=3, _nonce=nonce)
                    dfx = normalize_txs(cid, items, entry)
                    if not dfx.empty:
                        txs_all.append(dfx)
                except Exception as e:
                    st.warning(f"[{CHAINS[cid]['name']}] TXs: {e}")

        tokens_df_raw = pd.concat(tokens_all, ignore_index=True) if tokens_all else pd.DataFrame()
        tx_df_raw = pd.concat(txs_all, ignore_index=True) if txs_all else pd.DataFrame()

        tokens_df = tokens_df_raw[tokens_df_raw["value_usd"] >= float(min_usd)] if not tokens_df_raw.empty else pd.DataFrame()
        if not tokens_df.empty and search_token.strip():
            q = search_token.strip().lower()
            tokens_df = tokens_df[
                tokens_df["token"].str.lower().str.contains(q, na=False) |
                tokens_df["name"].str.lower().str.contains(q, na=False)
            ]

        tx_df = tx_df_raw.copy()
        if not tx_df.empty:
            if since_dt is not None:
                tx_df["_dt"] = pd.to_datetime(tx_df["timestamp"], utc=True, errors="coerce")
                tx_df = tx_df[tx_df["_dt"] >= since_dt].drop(columns=["_dt"])
            if float(min_usd) > 0:
                tx_df = tx_df[tx_df["value_usd"] >= float(min_usd)]

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        total_usd = float(tokens_df["value_usd"].sum()) if not tokens_df.empty else 0.0
        c1.metric("💰 Valor total (USD)", fmt_usd(total_usd, 2))
        c2.metric("🪙 Nº de tokens", int(tokens_df.shape[0]) if not tokens_df.empty else 0)
        c3.metric("🔗 Nº de redes", tokens_df["chain"].nunique() if not tokens_df.empty else 0)
        c4.metric("🧾 TXs filtradas", int(tx_df.shape[0]) if not tx_df.empty else 0)

        # Distribuição por rede
        if not tokens_df.empty:
            st.markdown("### Distribuição por rede")
            chart_type = st.radio(
                "Tipo de gráfico", ["Donut (recomendado)", "Barra horizontal"], horizontal=True, key="chart_type_chain",
            )
            by_chain = tokens_df.groupby("chain")["value_usd"].sum().sort_values(ascending=False)
            total = float(by_chain.sum())

            if chart_type.startswith("Donut"):
                try:
                    import matplotlib.pyplot as plt
                except Exception:
                    plt = None

                if plt is None:
                    st.info("Para ver o donut, instale:  `pip install matplotlib`")
                else:
                    dfp = by_chain.reset_index().rename(columns={"chain": "Rede", "value_usd": "Valor"})
                    dfp["pct"] = (dfp["Valor"] / total) * 100
                    keep = dfp["pct"] >= 2.0
                    donut_df = dfp.loc[keep, ["Rede", "Valor"]].copy()
                    outras = float(dfp.loc[~keep, "Valor"].sum())
                    if outras > 0:
                        donut_df.loc[len(donut_df)] = ["Outras", outras]

                    fig, ax = plt.subplots(figsize=(3.5, 3.5))
                    wedges, texts, autotexts = ax.pie(
                        donut_df["Valor"],
                        labels=donut_df["Rede"],
                        startangle=140,
                        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                        pctdistance=0.75,
                        wedgeprops=dict(width=0.35),
                        textprops={"fontsize": 9},
                    )
                    centre = plt.Circle((0, 0), 0.58, fc="white")
                    ax.add_artist(centre)
                    ax.axis("equal")
                    st.pyplot(fig, use_container_width=False)

                    aux = by_chain.reset_index().rename(columns={"chain":"Rede","value_usd":"Valor (USD)"})
                    aux["%"] = (aux["Valor (USD)"] / total * 100).round(2)
                    aux["Valor (USD)"] = aux["Valor (USD)"].apply(fmt_usd)
                    st.dataframe(aux, use_container_width=True, hide_index=True)
            else:
                bdf = by_chain.reset_index().rename(columns={"chain":"Rede","value_usd":"Valor (USD)"})
                bdf["%"] = (bdf["Valor (USD)"] / total * 100).round(2)
                colA, colB = st.columns([0.45, 0.55])
                with colA:
                    show = bdf.copy()
                    show["Valor (USD)"] = show["Valor (USD)"].apply(fmt_usd)
                    st.dataframe(show, use_container_width=True, hide_index=True)
                with colB:
                    norm = bdf.set_index("Rede")["Valor (USD)"]
                    st.bar_chart(norm, use_container_width=True, height=340)

        # Tabela de tokens
        st.subheader("💰 Tokens por Rede")
        top_n = st.slider("Mostrar Top N por rede", 3, 50, 15, 1)
        if not tokens_df.empty:
            summary = (tokens_df.groupby("chain")
                       .agg(total_usd=("value_usd","sum"), qtd_tokens=("token","count"))
                       .reset_index().sort_values("total_usd", ascending=False, ignore_index=True))
            show_sum = summary.copy()
            show_sum.rename(columns={"chain":"Rede","qtd_tokens":"Qtd. Tokens"}, inplace=True)
            show_sum["total_usd"] = show_sum["total_usd"].apply(fmt_usd)
            st.markdown("**Resumo por rede**")
            st.dataframe(show_sum, use_container_width=True, hide_index=True)

            st.markdown("**Detalhamento de tokens (separado por rede)**")
            order = tokens_df.groupby("chain")["value_usd"].sum().sort_values(ascending=False).index.tolist()
            tabs = st.tabs(order)
            for tab, ch in zip(tabs, order):
                with tab:
                    raw = tokens_df[tokens_df["chain"] == ch].copy().sort_values("value_usd", ascending=False).head(top_n)
                    def _scan_link(row):
                        base = EXPLORERS.get(int(row["chain_id"]), "")
                        return f"{base}/token/{row['token_address']}" if base and row["token_address"] else None
                    raw["Explorer"] = raw.apply(_scan_link, axis=1)
                    show = raw.rename(columns={
                        "logo":"Logo","token":"Ticker","name":"Nome","token_address":"Contrato",
                        "amount":"Quantidade","price_usd":"Preço (USD)","value_usd":"Valor (USD)",
                    })[["Logo","Ticker","Nome","Contrato","Quantidade","Preço (USD)","Valor (USD)","Explorer"]]
                    show["Quantidade"] = show["Quantidade"].apply(lambda v: fmt_amt(v, 8))
                    show["Preço (USD)"] = show["Preço (USD)"].apply(fmt_usd_precise)
                    show["Valor (USD)"] = show["Valor (USD)"].apply(fmt_usd)
                    st.dataframe(show, use_container_width=True, hide_index=True,
                                 column_config={
                                     "Logo": st.column_config.ImageColumn(""),
                                     "Explorer": st.column_config.LinkColumn("Scan", display_text="Abrir"),
                                 })

        # Detalhe por token + TXs
        st.subheader("🔎 Detalhe por token")
        token_opts = tokens_df["token"].fillna("").unique().tolist() if not tokens_df.empty else []
        sel_token = st.selectbox("Escolha um token (opcional)", ["(todos)"] + token_opts)
        if sel_token != "(todos)" and not tokens_df.empty:
            sel_rows = tokens_df[tokens_df["token"] == sel_token].copy()
            st.write(f"Encontrados **{len(sel_rows)}** registros de **{sel_token}** em {sel_rows['chain'].nunique()} rede(s).")
            addresses = set(sel_rows["token_address"].dropna().unique().tolist())
            tx_sel = tx_df[tx_df["to"].str.lower().isin([a.lower() for a in addresses]) |
                           tx_df["from"].str.lower().isin([a.lower() for a in addresses])] if not tx_df.empty else pd.DataFrame()
            if not tx_sel.empty:
                def _tx(row):
                    base = EXPLORERS.get(int(row["chain_id"]), "")
                    return f"{base}/tx/{row['hash']}" if base and row["hash"] else None
                tx_sel = tx_sel.copy()
                tx_sel["Tx"] = tx_sel.apply(_tx, axis=1)
                nice = tx_sel.rename(columns={
                    "chain":"Rede","timestamp_human":"Data/Hora","direction_chip":"Direção",
                    "from":"De","to":"Para","value_usd":"Valor (USD)","fees_usd":"Taxa (USD)"
                })[["Rede","Tx","Data/Hora","Direção","De","Para","Valor (USD)","Taxa (USD)"]]
                nice["Valor (USD)"] = nice["Valor (USD)"].apply(fmt_usd)
                nice["Taxa (USD)"] = nice["Taxa (USD)"].apply(lambda v: fmt_usd(v, 4))
                st.markdown("**Últimas transações desse token**")
                st.dataframe(nice, use_container_width=True, hide_index=True,
                             column_config={"Tx": st.column_config.LinkColumn("Hash", display_text="Abrir")})
            else:
                st.info("Sem transações recentes para esse token (dentro do período e redes filtradas).")

        # Transações gerais
        st.subheader("🧾 Transações (após filtros)")
        if not tx_df.empty:
            def _tx_link(row):
                base = EXPLORERS.get(int(row["chain_id"]), "")
                return f"{base}/tx/{row['hash']}" if base and row["hash"] else None
            tx_df = tx_df.copy()
            tx_df["Tx"] = tx_df.apply(_tx_link, axis=1)
            nice = tx_df.rename(columns={
                "chain":"Rede","timestamp_human":"Data/Hora","direction_chip":"Direção",
                "from":"De","to":"Para","value_usd":"Valor (USD)","fees_usd":"Taxa (USD)",
            })[["Rede","Tx","Data/Hora","Direção","De","Para","Valor (USD)","Taxa (USD)"]]
            nice["Valor (USD)"] = nice["Valor (USD)"].apply(fmt_usd)
            nice["Taxa (USD)"] = nice["Taxa (USD)"].apply(lambda v: fmt_usd(v, 4))
            st.dataframe(nice, use_container_width=True, hide_index=True,
                         column_config={"Tx": st.column_config.LinkColumn("Hash", display_text="Abrir")})
        else:
            st.info("Nenhuma transação encontrada (após filtros).")
    else:
        st.error("Endereço/TX inválido. Use 0x... (EVM), bc1/1/3 (BTC) ou TXID BTC.")

# ==========================
# DESCOBERTA — SEMPRE VISÍVEL (igual ao seu)
# ==========================
st.markdown("---")
st.header("🔎 Descoberta (Top Gainers → DexScreener → Holders → DeBank)")

@st.cache_data(ttl=300, show_spinner=False)
def cg_top_gainers(period: str, limit=50) -> pd.DataFrame:
    valid = {"1h":"1h","24h":"24h","7d":"7d","14d":"14d","30d":"30d","200d":"200d","1y":"1y"}
    p = valid.get(period, "24h")
    per_page = max(50, min(100, int(limit)))
    pages_needed = (int(limit) - 1) // per_page + 1
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params_base = dict(
        vs_currency="usd",
        order="market_cap_desc",
        price_change_percentage="1h,24h,7d,14d,30d,200d,1y",
        sparkline="false",
    )
    headers = {"Accept":"application/json","User-Agent":"Mozilla/5.0 (TrackWallet/1.0; +https://local)"}
    rows = []
    for page in range(1, pages_needed + 1):
        tries, wait = 0, 1.5
        while True:
            tries += 1
            params = {**params_base, "per_page": per_page, "page": page}
            r = requests.get(url, params=params, headers=headers, timeout=25)
            if r.status_code == 200:
                rows.extend(r.json())
                break
            if r.status_code in (429,500,502,503,504) and tries < 5:
                delay = wait
                if "Retry-After" in r.headers:
                    try: delay = max(delay, float(r.headers["Retry-After"]))
                    except Exception: pass
                time.sleep(delay); wait *= 1.8; continue
            break
    if not rows: return pd.DataFrame()

    key_map = {
        "1h": ("price_change_percentage_1h_in_currency", "price_change_percentage_1h"),
        "24h": ("price_change_percentage_24h_in_currency", "price_change_percentage_24h"),
        "7d": ("price_change_percentage_7d_in_currency", "price_change_percentage_7d"),
        "14d": ("price_change_percentage_14d_in_currency", "price_change_percentage_14d"),
        "30d": ("price_change_percentage_30d_in_currency", "price_change_percentage_30d"),
        "200d": ("price_change_percentage_200d_in_currency", "price_change_percentage_200d"),
        "1y": ("price_change_percentage_1y_in_currency", "price_change_percentage_1y"),
    }
    k_in, k_fb = key_map[p]
    out = []
    for c in rows:
        chg = c.get(k_in)
        if chg is None: chg = c.get(k_fb, 0.0)
        out.append({
            "id": c.get("id"),
            "symbol": (c.get("symbol") or "").upper(),
            "name": c.get("name", ""),
            "price": c.get("current_price", 0.0),
            "chg": chg or 0.0,
            "mcap": c.get("market_cap", 0),
            "logo": c.get("image"),
        })
    df = pd.DataFrame(out)
    if df.empty: return df
    return df.sort_values("chg", ascending=False, ignore_index=True).head(int(limit))

@st.cache_data(ttl=900, show_spinner=False)
def cg_coin_contract_on_chain(coin_id: str, chain_id: int) -> Optional[str]:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    r = requests.get(url, params={
        "localization":"false","tickers":"false","market_data":"false",
        "community_data":"false","developer_data":"false","sparkline":"false"
    }, timeout=20)
    if not r.ok:
        return None
    platforms = (r.json() or {}).get("platforms") or {}
    platform_key = COINGECKO_PLATFORM.get(chain_id)
    addr = (platforms.get(platform_key) or "").strip()
    return addr or None

@st.cache_data(ttl=900, show_spinner=False)
def cg_contract_and_link(coin_id: str, chain_id: int):
    addr = cg_coin_contract_on_chain(coin_id, chain_id)
    base = EXPLORERS.get(chain_id, "")
    link = f"{base}/token/{addr}" if (base and addr) else None
    return addr, link

@st.cache_data(ttl=300, show_spinner=False)
def ds_pairs_by_contract(chain_id: int, contract: str):
    r = requests.get("https://api.dexscreener.com/latest/dex/search", params={"q": contract}, timeout=20)
    if not r.ok:
        return pd.DataFrame()
    pairs = (r.json() or {}).get("pairs") or []
    slug = DEX_CHAIN_SLUG.get(chain_id)
    if slug:
        pairs = [p for p in pairs if p.get("chainId") == slug]
    rows = []
    for p in pairs:
        rows.append({
            "pair": p.get("pairAddress"),
            "dex": p.get("dexId"),
            "base": p.get("baseToken", {}).get("symbol"),
            "quote": p.get("quoteToken", {}).get("symbol"),
            "priceUsd": float(p.get("priceUsd") or 0),
            "liqUsd": float((p.get("liquidity") or {}).get("usd") or 0),
            "fdv": float(p.get("fdv") or 0),
            "vol24": float((p.get("volume") or {}).get("h24") or 0),
            "tx24": int((p.get("txns") or {}).get("h24", {}).get("buys", 0)) + int((p.get("txns") or {}).get("h24", {}).get("sells", 0)),
            "url": p.get("url"),
        })
    return pd.DataFrame(rows).sort_values("vol24", ascending=False, ignore_index=True)

@st.cache_data(ttl=180, show_spinner=False)
def ds_recent_trades(chain_id: int, pair_addr: str) -> pd.DataFrame:
    slug = DEX_CHAIN_SLUG.get(chain_id)
    if not slug:
        return pd.DataFrame()
    url = f"https://api.dexscreener.com/latest/dex/trades/{slug}/{pair_addr}"
    r = requests.get(url, timeout=20)
    if not r.ok:
        return pd.DataFrame()
    trades = (r.json() or {}).get("trades") or []
    rows = []
    for t in trades:
        try:
            ts = datetime.utcfromtimestamp(int(t.get("timeMs", 0)) / 1000).isoformat() + "Z"
        except Exception:
            ts = None
        rows.append({
            "time": _human_iso(ts) if ts else "",
            "type": (t.get("type") or "").upper(),
            "priceUsd": float(t.get("priceUsd") or 0),
            "amount": float(t.get("amount") or 0),
            "totalUsd": float(t.get("totalUsd") or 0),
            "maker": t.get("maker"),
            "tx": t.get("transactionId"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def top_holders_covalent(chain_id: int, contract: str, page_size: int = 50) -> pd.DataFrame:
    url = f"https://api.covalenthq.com/v1/{chain_id}/tokens/{contract}/token_holders_v2/"
    params = {"key": COVALENT_API_KEY, "page-size": page_size, "quote-currency": "USD"}
    r = requests.get(url, params=params, timeout=30)
    if not r.ok:
        return pd.DataFrame()
    items = (r.json() or {}).get("data", {}).get("items") or []
    rows = []
    for it in items:
        rows.append({
            "address": it.get("address"),
            "balance": float(it.get("balance") or 0),
            "value_usd": float((it.get("balance_quote") or 0) or 0),
        })
    return pd.DataFrame(rows).sort_values("value_usd", ascending=False, ignore_index=True)

# Config Descoberta
cfg1, cfg2, cfg3 = st.columns([0.4, 0.3, 0.3])
with cfg1:
    period = st.selectbox("Período (CoinGecko)", ["1h","24h","7d","14d","30d","200d","1y"], index=1, key="cg_period")
with cfg2:
    lim = st.slider("Quantidade", 10, 250, 50, 10, key="cg_limit")
with cfg3:
    net = st.selectbox("Rede (p/ contrato na tabela)", [f"{cid} - {CHAINS[cid]['name']}" for cid in CHAINS.keys()],
                       index=0, key="discover_net")
chain_id_sel = int(net.split(" - ")[0])

left, right = st.columns([0.55, 0.45])
with left:
    st.subheader("1) Top Gainers (CoinGecko)")
    try:
        df_g = cg_top_gainers(period=period, limit=lim).copy()
        if df_g.empty:
            st.info("CoinGecko não retornou dados agora (limite de taxa). Tente novamente em instantes.")
        else:
            df_g["Página"] = df_g["id"].map(lambda x: f"https://www.coingecko.com/en/coins/{x}")
            contracts = df_g["id"].apply(lambda x: cg_contract_and_link(x, chain_id_sel))
            df_g["contract_addr"] = [c for (c, l) in contracts]
            df_g["Contrato"] = [l for (c, l) in contracts]
            table = df_g.rename(columns={
                "logo":"Logo","name":"Nome","symbol":"Ticker","price":"Preço (USD)",
                "chg":f"% {period}","mcap":"Mkt Cap"
            })[["Logo","Nome","Ticker","Preço (USD)",f"% {period}","Mkt Cap","Página","Contrato"]]
            table["Preço (USD)"] = table["Preço (USD)"].apply(fmt_usd_precise)
            table["Mkt Cap"] = table["Mkt Cap"].apply(lambda v: fmt_usd(v,0))
            st.dataframe(
                table,
                column_config={
                    "Logo": st.column_config.ImageColumn(""),
                    f"% {period}": st.column_config.NumberColumn(f"% {period}", format="%.2f%%"),
                    "Página": st.column_config.LinkColumn("CoinGecko", display_text="Abrir"),
                    "Contrato": st.column_config.LinkColumn("Contrato", display_text="Abrir"),
                },
                use_container_width=True, hide_index=True, height=420,
            )
    except Exception as e:
        st.warning(f"CoinGecko: {e}")

with right:
    st.subheader("2) Escolha Coin ID (opcional) e confirme")
    coin_or_contract = st.text_input(
        "Coin ID (CoinGecko) **ou** endereço do contrato (0x...)",
        help="Ex.: 'pepe' ou 0xabc... Se for 0x..., busco direto DexScreener/holders.",
        key="discover_coin",
    )
    if st.button("Buscar contrato na rede selecionada", key="btn_contract"):
        entry = (coin_or_contract or "").strip()
        if not entry:
            st.warning("Informe um Coin ID ou um endereço de contrato (0x...).")
        else:
            try:
                if is_evm_address(entry):
                    contract = to_checksum(entry)
                else:
                    contract = cg_coin_contract_on_chain(entry, chain_id_sel)
                if not contract:
                    st.error("Contrato não encontrado para essa rede.")
                else:
                    st.success(f"Contrato: {contract}")
                    st.session_state["_discover_contract"] = contract
                    st.session_state["_discover_chain"] = chain_id_sel
            except Exception as e:
                st.error(f"Erro ao buscar contrato: {e}")

contract = st.session_state.get("_discover_contract")
chain_id_discover = st.session_state.get("_discover_chain")

# 3) PARES & TRADES (TOP 2)
if contract and chain_id_discover:
    st.subheader("3) DexScreener — Pares & Trades")
    trades_limit = st.slider("Qtd. de trades por par", 10, 200, 50, 10, key="trades_limit")

    try:
        df_pairs = ds_pairs_by_contract(chain_id_discover, contract)
        if df_pairs.empty:
            st.info("Nenhum par encontrado para esse contrato na DexScreener.")
        else:
            show_pairs = df_pairs.rename(columns={
                    "pair":"Pair","dex":"DEX","base":"Base","quote":"Quote",
                    "priceUsd":"Preço","liqUsd":"Liquidez","vol24":"Vol 24h",
                    "tx24":"TX 24h","url":"Link"
                })[["DEX","Base","Quote","Preço","Liquidez","Vol 24h","TX 24h","Link"]]
            show_pairs["Preço"] = show_pairs["Preço"].apply(fmt_usd_precise)
            show_pairs["Liquidez"] = show_pairs["Liquidez"].apply(lambda v: fmt_usd(v,0))
            show_pairs["Vol 24h"] = show_pairs["Vol 24h"].apply(lambda v: fmt_usd(v,0))

            st.dataframe(
                show_pairs,
                column_config={"Link": st.column_config.LinkColumn("Par", display_text="Abrir")},
                hide_index=True,
                use_container_width=True
            )

            top2 = df_pairs.head(2).copy()
            if top2.empty:
                st.info("Nenhum par com trades para exibir.")
            else:
                for i, prow in enumerate(top2.itertuples(index=False), start=1):
                    pair_addr = getattr(prow, "pair")
                    dex = getattr(prow, "dex")
                    base = getattr(prow, "base")
                    quote = getattr(prow, "quote")
                    pair_url = getattr(prow, "url")

                    st.markdown(
                        f"**Trades recentes — {i}º par • {dex} • {base}/{quote}** &nbsp; "
                        f"[Abrir par ↗]({pair_url})"
                    )

                    df_tr = ds_recent_trades(chain_id_discover, pair_addr)
                    if df_tr.empty:
                        st.info("API de trades da DexScreener não trouxe dados para esse par.")
                        continue

                    df_top = df_tr.head(trades_limit).copy()
                    buys = (df_top["type"].str.upper() == "BUY").sum()
                    sells = (df_top["type"].str.upper() == "SELL").sum()
                    vol_usd = float(df_top["totalUsd"].sum())

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Trades (janela)", f"{len(df_top):,}")
                    k2.metric("Buys / Sells", f"{buys} / {sells}")
                    k3.metric("Volume USD (janela)", fmt_usd(vol_usd,0))

                    def _tx_link(txid: str):
                        base_scan = EXPLORERS.get(int(chain_id_discover), "")
                        return f"{base_scan}/tx/{txid}" if (base_scan and txid) else None

                    df_top = df_top.copy()
                    df_top["TxLink"] = df_top["tx"].map(_tx_link)

                    df_show = df_top.rename(columns={
                            "time":"Data/Hora","type":"Tipo","priceUsd":"Preço","amount":"Qtd Base",
                            "totalUsd":"Total USD","tx":"Tx"
                        })[["Data/Hora","Tipo","Preço","Qtd Base","Total USD","TxLink"]]
                    df_show["Preço"] = df_show["Preço"].apply(fmt_usd_precise)
                    df_show["Qtd Base"] = df_show["Qtd Base"].apply(lambda v: fmt_amt(v,6))
                    df_show["Total USD"] = df_show["Total USD"].apply(fmt_usd)
                    st.dataframe(
                        df_show,
                        column_config={"TxLink": st.column_config.LinkColumn("Tx", display_text="Abrir")},
                        hide_index=True,
                        use_container_width=True,
                        height=300,
                    )
    except Exception as e:
        st.warning(f"DexScreener: {e}")

# 4) Top holders + DeBank
st.subheader("4) Top holders (Covalent) + DeBank")

st.markdown("**Consultar qualquer carteira no DeBank**")
debank_manual = st.text_input(
    "Carteira para consultar (0x...)", 
    placeholder="0x...", 
    key="debank_manual_addr"
)

def _explorer_from_row(rowt):
    chain_hint = rowt.get("chain")
    base = None
    if isinstance(chain_hint, (int, float)):
        base = EXPLORERS.get(int(chain_hint), "")
    else:
        map_slug = {
            "eth": 1, "ethereum": 1,
            "bsc": 56, "bnb": 56,
            "matic": 137, "polygon": 137, "polygon-pos": 137,
            "op": 10, "optimism": 10,
            "arb": 42161, "arbitrum": 42161, "arbitrum-one": 42161,
            "base": 8453,
            "avax": 43114, "avalanche": 43114,
            "ftm": 250, "fantom": 250,
        }
        cid = map_slug.get(str(chain_hint).lower())
        if cid:
            base = EXPLORERS.get(cid, "")
    ca = rowt.get("token_address")
    return f"{base}/token/{ca}" if base and ca else None

if st.button("Buscar tokens dessa carteira", key="btn_debank_manual"):
    target = (debank_manual or "").strip()
    if not is_evm_address(target):
        st.warning("Informe um endereço 0x válido.")
    else:
        with st.spinner("Consultando DeBank..."):
            df_tokens_any = debank_token_list(target)
        if df_tokens_any.empty:
            st.info("DeBank não retornou tokens ou bloqueou a consulta. Usando fallback (Covalent em todas as redes).")
            df_tokens_any = covalent_wallet_tokens_fallback(list(CHAINS.keys()), target)
        if df_tokens_any.empty:
            st.error("Não consegui obter tokens dessa carteira.")
        else:
            df_tokens_any = df_tokens_any.copy()
            df_tokens_any["Explorer"] = df_tokens_any.apply(_explorer_from_row, axis=1)
            nice_any = df_tokens_any.rename(columns={
                "logo":"Logo","token":"Ticker","name":"Nome","token_address":"Contrato",
                "amount":"Quantidade","price_usd":"Preço (USD)","value_usd":"Valor (USD)",
            })[["Logo","Ticker","Nome","Contrato","Quantidade","Preço (USD)","Valor (USD)","Explorer"]]
            nice_any["Quantidade"] = nice_any["Quantidade"].apply(lambda v: fmt_amt(v,6))
            nice_any["Preço (USD)"] = nice_any["Preço (USD)"].apply(fmt_usd_precise)
            nice_any["Valor (USD)"] = nice_any["Valor (USD)"].apply(fmt_usd)
            st.dataframe(
                nice_any, use_container_width=True, hide_index=True, height=380,
                column_config={
                    "Logo": st.column_config.ImageColumn(""),
                    "Explorer": st.column_config.LinkColumn("Scan", display_text="Abrir"),
                },
            )

st.markdown("---")
st.markdown("**Holders do token selecionado (se houver um contrato carregado acima):**")

try:
    if contract and chain_id_discover:
        df_h = top_holders_covalent(chain_id_discover, contract, page_size=30)
        if df_h.empty:
            st.info("Top holders indisponível para este contrato na sua conta Covalent ou sem dados.")
        else:
            df_h = df_h.copy()
            df_h["DeBank"] = df_h["address"].map(lambda a: f"https://debank.com/profile/{a}")
            show_h = df_h.rename(columns={"address":"Holder","balance":"Saldo (raw)","value_usd":"Valor (USD)","DeBank":"DeBank"})[
                ["Holder","Valor (USD)","DeBank"]
            ]
            show_h["Valor (USD)"] = show_h["Valor (USD)"].apply(fmt_usd)
            st.dataframe(
                show_h,
                column_config={"DeBank": st.column_config.LinkColumn("DeBank", display_text="Abrir")},
                hide_index=True, use_container_width=True, height=320,
            )

            st.subheader("5) Tokens dos Top Holders (DeBank)")
            topN = st.slider("Quantos holders analisar", 1, min(10, len(df_h)), 3, 1, key="n_holders_debank")
            for _, row in df_h.head(topN).iterrows():
                holder = row["address"]
                with st.expander(f"Carteira {holder} — tokens"):
                    df_tokens = debank_token_list(holder)
                    if df_tokens.empty:
                        st.info("DeBank não retornou tokens ou bloqueou a consulta. Usando fallback (Covalent).")
                        df_tokens = covalent_wallet_tokens_fallback(list(CHAINS.keys()), holder)
                    if df_tokens.empty:
                        st.warning("Não foi possível obter os tokens desta carteira.")
                    else:
                        df_tokens = df_tokens.copy()
                        df_tokens["Explorer"] = df_tokens.apply(_explorer_from_row, axis=1)
                        nice = df_tokens.rename(columns={
                            "logo":"Logo","token":"Ticker","name":"Nome","token_address":"Contrato",
                            "amount":"Quantidade","price_usd":"Preço (USD)","value_usd":"Valor (USD)",
                        })[["Logo","Ticker","Nome","Contrato","Quantidade","Preço (USD)","Valor (USD)","Explorer"]]
                        nice["Quantidade"] = nice["Quantidade"].apply(lambda v: fmt_amt(v,6))
                        nice["Preço (USD)"] = nice["Preço (USD)"].apply(fmt_usd_precise)
                        nice["Valor (USD)"] = nice["Valor (USD)"].apply(fmt_usd)
                        st.dataframe(
                            nice, use_container_width=True, hide_index=True, height=360,
                            column_config={
                                "Logo": st.column_config.ImageColumn(""),
                                "Explorer": st.column_config.LinkColumn("Scan", display_text="Abrir"),
                            },
                        )
    else:
        st.info("Nenhum contrato carregado acima — use a consulta por carteira para ver tokens via DeBank.")
except Exception as e:
    st.warning(f"Covalent/DeBank: {e}")

st.caption(f"Período p/ TXs: **{preset_label}** | Valor mínimo: **{fmt_usd(min_usd,2)}** | Atualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
