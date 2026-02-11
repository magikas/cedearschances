"""
📊 Monthly Stock Performance Analyzer
Analiza el rendimiento mensual de acciones en los últimos 18 años y encuentra correlaciones.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Monthly Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3e;
    --accent: #6366f1;
    --accent2: #22d3ee;
    --green: #10b981;
    --red: #f43f5e;
    --text: #e2e8f0;
    --text-muted: #64748b;
}

.stApp { background: var(--bg); color: var(--text); }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
h1 { font-weight: 800; font-size: 2.2rem !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--accent2); }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin: 2rem 0 1rem 0;
}

.correlation-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

div[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
div[data-testid="stSidebar"] * { color: var(--text) !important; }

.stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.05em;
}

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}

.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

MONTHS_SHORT = {
    1: "ENE", 2: "FEB", 3: "MAR", 4: "ABR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AGO", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DIC"
}

CACHE_FILE = "monthly_data_cache.parquet"

# ─── SECTOR MAPPING ──────────────────────────────────────────────────────────
# Mapeo de tickers a sectores para análisis comparativo
SECTOR_MAP = {
    # Aerolíneas
    "AAL": "Aerolíneas", "DAL": "Aerolíneas", "UAL": "Aerolíneas", "LUV": "Aerolíneas",
    "JBLU": "Aerolíneas", "ALK": "Aerolíneas", "SAVE": "Aerolíneas", "HA": "Aerolíneas",
    "SKYW": "Aerolíneas", "MESA": "Aerolíneas", "RYAAY": "Aerolíneas", "VLRS": "Aerolíneas",

    # Tecnología / Software
    "AAPL": "Tecnología", "MSFT": "Tecnología", "GOOG": "Tecnología", "GOOGL": "Tecnología",
    "META": "Tecnología", "AMZN": "Tecnología", "NFLX": "Tecnología", "CRM": "Tecnología",
    "ADBE": "Tecnología", "ORCL": "Tecnología", "NOW": "Tecnología", "INTU": "Tecnología",
    "SHOP": "Tecnología", "SQ": "Tecnología", "UBER": "Tecnología", "ABNB": "Tecnología",
    "SNAP": "Tecnología", "PINS": "Tecnología", "SPOT": "Tecnología", "PLTR": "Tecnología",
    "AI": "Tecnología", "ACN": "Tecnología",

    # Semiconductores
    "NVDA": "Semiconductores", "AMD": "Semiconductores", "INTC": "Semiconductores",
    "AVGO": "Semiconductores", "QCOM": "Semiconductores", "TXN": "Semiconductores",
    "AMAT": "Semiconductores", "LRCX": "Semiconductores", "KLAC": "Semiconductores",
    "MU": "Semiconductores", "MRVL": "Semiconductores", "ON": "Semiconductores",
    "ASML": "Semiconductores", "TSM": "Semiconductores", "ARM": "Semiconductores",
    "ALAB": "Semiconductores",

    # Bancos / Financiero
    "JPM": "Bancos", "BAC": "Bancos", "WFC": "Bancos", "GS": "Bancos",
    "MS": "Bancos", "C": "Bancos", "USB": "Bancos", "PNC": "Bancos",
    "SCHW": "Bancos", "BK": "Bancos", "STT": "Bancos", "CFG": "Bancos",
    "FITB": "Bancos", "KEY": "Bancos", "RF": "Bancos", "HBAN": "Bancos",
    "V": "Bancos", "MA": "Bancos", "AXP": "Bancos", "COF": "Bancos",
    "BRK-B": "Bancos", "BBAR": "Bancos", "BMA": "Bancos", "GGAL": "Bancos",
    "SUPV": "Bancos",

    # Farmacéuticas / Salud
    "JNJ": "Salud", "PFE": "Salud", "UNH": "Salud", "ABBV": "Salud",
    "MRK": "Salud", "LLY": "Salud", "TMO": "Salud", "ABT": "Salud",
    "AMGN": "Salud", "BMY": "Salud", "GILD": "Salud", "ISRG": "Salud",
    "VRTX": "Salud", "REGN": "Salud", "ZTS": "Salud", "SYK": "Salud",
    "MDT": "Salud", "DHR": "Salud", "ELV": "Salud", "CI": "Salud",
    "HCA": "Salud", "IQV": "Salud", "BIIB": "Salud",

    # Mineras / Materiales
    "AEM": "Mineras", "NEM": "Mineras", "GOLD": "Mineras", "FNV": "Mineras",
    "WPM": "Mineras", "AUY": "Mineras", "KGC": "Mineras", "AG": "Mineras",
    "PAAS": "Mineras", "HL": "Mineras", "CDE": "Mineras", "FSM": "Mineras",
    "BHP": "Mineras", "RIO": "Mineras", "VALE": "Mineras", "FCX": "Mineras",
    "SCCO": "Mineras", "TECK": "Mineras", "CLF": "Mineras", "X": "Mineras",
    "NUE": "Mineras", "STLD": "Mineras", "AA": "Mineras",

    # Energía / Petróleo
    "XOM": "Energía", "CVX": "Energía", "COP": "Energía", "SLB": "Energía",
    "EOG": "Energía", "MPC": "Energía", "PSX": "Energía", "VLO": "Energía",
    "PXD": "Energía", "DVN": "Energía", "OXY": "Energía", "HAL": "Energía",
    "BKR": "Energía", "FANG": "Energía", "HES": "Energía",
    "YPF": "Energía", "VIST": "Energía", "PAM": "Energía",

    # Consumo
    "KO": "Consumo", "PEP": "Consumo", "PG": "Consumo", "COST": "Consumo",
    "WMT": "Consumo", "HD": "Consumo", "MCD": "Consumo", "NKE": "Consumo",
    "SBUX": "Consumo", "TGT": "Consumo", "LOW": "Consumo", "DG": "Consumo",
    "DLTR": "Consumo", "ROST": "Consumo", "TJX": "Consumo", "ANF": "Consumo",
    "ABEV": "Consumo",

    # ETFs / Índices
    "SPY": "ETFs", "QQQ": "ETFs", "IWM": "ETFs", "DIA": "ETFs",
    "ACWI": "ETFs", "ARKK": "ETFs", "XLF": "ETFs", "XLE": "ETFs",
    "XLV": "ETFs", "XLK": "ETFs", "XLI": "ETFs", "XLP": "ETFs",
    "GDX": "ETFs", "GDXJ": "ETFs", "SLV": "ETFs", "GLD": "ETFs",
    "EEM": "ETFs", "VWO": "ETFs", "EWZ": "ETFs", "EFA": "ETFs",

    # Telecom / Comunicaciones
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom", "CMCSA": "Telecom",
    "CHTR": "Telecom", "DIS": "Telecom", "WBD": "Telecom",

    # Industriales
    "CAT": "Industriales", "DE": "Industriales", "BA": "Industriales",
    "HON": "Industriales", "UNP": "Industriales", "GE": "Industriales",
    "RTX": "Industriales", "LMT": "Industriales", "NOC": "Industriales",
    "GD": "Industriales", "MMM": "Industriales", "FDX": "Industriales",
    "UPS": "Industriales",

    # Real Estate / REITs
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate", "O": "Real Estate",
    "DLR": "Real Estate", "PSA": "Real Estate", "WELL": "Real Estate",
    "AVB": "Real Estate", "EQR": "Real Estate",

    # Argentina / LATAM
    "MELI": "LATAM", "NU": "LATAM", "AMX": "LATAM", "ARCO": "LATAM",
    "GLOB": "LATAM", "DLO": "LATAM", "STNE": "LATAM", "CAAP": "LATAM",
    "CRESY": "LATAM", "CEPU": "LATAM", "EDN": "LATAM", "LOMA": "LATAM",
    "TEO": "LATAM", "TGS": "LATAM", "TX": "LATAM", "IRS": "LATAM",
}


def get_ticker_sector(ticker: str) -> str:
    """Devuelve el sector de un ticker, o 'Otros' si no está mapeado."""
    return SECTOR_MAP.get(ticker, "Otros")


def get_available_sectors(tickers: list[str]) -> dict[str, list[str]]:
    """Agrupa los tickers disponibles por sector."""
    sectors = {}
    for t in tickers:
        sector = get_ticker_sector(t)
        sectors.setdefault(sector, []).append(t)
    # Ordenar por cantidad de tickers (más representados primero)
    return dict(sorted(sectors.items(), key=lambda x: -len(x[1])))


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_tickers_from_file(path: str) -> list[str]:
    with open(path, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    return tickers


@st.cache_data(ttl=3600, show_spinner=False)
def _extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Extrae la columna 'Close' como Series.
    Maneja todos los formatos posibles de yfinance:

      Formato A (yfinance clásico, un ticker):
          columnas simples → "Close", "Open", "High", ...

      Formato B (yfinance >= 0.2.31, un ticker sin group_by):
          MultiIndex nivel 0 = Price, nivel 1 = Ticker
          → ("Close", "AAPL")

      Formato C (yfinance con group_by="ticker", un ticker):
          MultiIndex nivel 0 = Ticker, nivel 1 = Price
          → ("AAPL", "Close")
    """
    if df is None or df.empty:
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0).tolist()
            lvl1 = df.columns.get_level_values(1).tolist()

            # Formato B: ("Close", "AAPL") — nivel 0 son campos de precio
            if "Close" in lvl0:
                sub = df["Close"]
                if isinstance(sub, pd.DataFrame):
                    # Puede tener columna con el ticker o solo una columna
                    if ticker in sub.columns:
                        series = sub[ticker]
                    else:
                        series = sub.iloc[:, 0]
                else:
                    series = sub

            # Formato C: ("AAPL", "Close") — nivel 0 es el ticker
            elif ticker in lvl0:
                sub = df[ticker]
                if isinstance(sub, pd.DataFrame):
                    if "Close" in sub.columns:
                        series = sub["Close"]
                    else:
                        series = sub.iloc[:, 0]
                else:
                    series = sub

            # Fallback: buscar "Close" en cualquier nivel
            elif "Close" in lvl1:
                cols_with_close = [c for c in df.columns if c[1] == "Close"]
                series = df[cols_with_close[0]]

            else:
                return None

        # Formato A: columnas simples
        elif "Close" in df.columns:
            series = df["Close"]

        elif ticker in df.columns:
            series = df[ticker]

        else:
            return None

        # Asegurar que sea Series 1D
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = pd.to_numeric(series, errors="coerce").dropna()
        return series if len(series) > 0 else None

    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers: list[str], years: int = 18) -> pd.DataFrame:
    """Descarga precios de cierre mensual para todos los tickers."""
    end = datetime.today()
    start = end - timedelta(days=365 * years + 30)

    all_data = {}
    failed = []
    progress = st.progress(0, text="Iniciando descarga...")

    for i, ticker in enumerate(tickers):
        progress.progress(
            (i + 1) / len(tickers),
            text=f"Descargando {ticker}... ({i+1}/{len(tickers)})"
        )
        try:
            # SIN group_by para evitar MultiIndex innecesario en descarga individual
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )

            if df is None or df.empty:
                failed.append(ticker)
                continue

            close = _extract_close_series(df, ticker)

            if close is None or len(close) < 50:
                failed.append(ticker)
                continue

            close.index = pd.to_datetime(close.index)
            # "ME" = Month End; si falla usar "M" (pandas < 2.2)
            try:
                monthly = close.resample("ME").last().dropna()
            except Exception:
                monthly = close.resample("M").last().dropna()

            if len(monthly) > 10:
                all_data[ticker] = monthly

        except Exception as e:
            failed.append(f"{ticker}({e})")

    progress.empty()

    if failed:
        # Mostrar en sidebar cuántos fallaron, sin interrumpir
        st.sidebar.warning(f"⚠️ {len(failed)} tickers sin datos:\n{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    if not all_data:
        st.error(
            "No se pudieron obtener datos. Posibles causas:\n"
            "- Sin conexión a internet\n"
            "- Todos los tickers son inválidos\n"
            "- Rate limit de Yahoo Finance (esperá 1 minuto y reintentá)"
        )
        return pd.DataFrame()

    # Construir DataFrame alineando fechas
    result = pd.DataFrame(all_data)
    result.index = pd.to_datetime(result.index)
    result.dropna(axis=1, how="all", inplace=True)
    return result


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula rendimientos porcentuales mes a mes."""
    returns = prices.pct_change(fill_method=None) * 100
    returns.dropna(how="all", inplace=True)
    return returns


def get_monthly_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (ticker, mes) calcula:
    - veces_subio, veces_bajo, total_años
    - pct_sube, retorno_promedio, retorno_mediano
    """
    rows = []
    returns_copy = returns.copy()
    returns_copy.index = pd.to_datetime(returns_copy.index)

    for ticker in returns_copy.columns:
        col = returns_copy[ticker].dropna()
        for month in range(1, 13):
            vals = col[col.index.month == month]
            if len(vals) == 0:
                continue
            veces_subio = (vals > 0).sum()
            veces_bajo = (vals < 0).sum()
            total = len(vals)
            rows.append({
                "ticker": ticker,
                "month": month,
                "month_name": MONTHS_ES[month],
                "veces_subio": int(veces_subio),
                "veces_bajo": int(veces_bajo),
                "total_años": int(total),
                "pct_sube": round(veces_subio / total * 100, 1),
                "retorno_prom": round(vals.mean(), 2),
                "retorno_med": round(vals.median(), 2),
                "retorno_max": round(vals.max(), 2),
                "retorno_min": round(vals.min(), 2),
                "std": round(vals.std(), 2),
            })

    return pd.DataFrame(rows)


def find_correlations(returns: pd.DataFrame, ref_ticker: str, ref_month: int,
                      min_overlap: int = 10) -> pd.DataFrame:
    """
    Dado un ticker y un mes de referencia, encuentra qué acciones se movieron
    en dirección OPUESTA o IGUAL ese mes, y también el MES SIGUIENTE.
    """
    returns_copy = returns.copy()
    returns_copy.index = pd.to_datetime(returns_copy.index)

    ref_col = returns_copy[ref_ticker].dropna()
    ref_month_data = ref_col[ref_col.index.month == ref_month]

    if len(ref_month_data) == 0:
        return pd.DataFrame()

    # Años donde el ref_ticker bajó ese mes
    ref_down_years = ref_month_data[ref_month_data < 0].index.year.tolist()
    ref_up_years = ref_month_data[ref_month_data > 0].index.year.tolist()

    rows = []
    for ticker in returns_copy.columns:
        if ticker == ref_ticker:
            continue

        col = returns_copy[ticker].dropna()

        # ── Mismo mes ──
        same_month = col[col.index.month == ref_month]
        same_idx = same_month.index

        # Años en común con ref_ticker
        common_years = [y for y in ref_down_years if y in same_idx.year.tolist()]
        if len(common_years) >= min_overlap:
            # ¿Cuántas veces subió este ticker cuando ref bajó?
            vals_when_ref_down = same_month[same_month.index.year.isin(common_years)]
            subio = (vals_when_ref_down > 0).sum()
            rows.append({
                "ticker": ticker,
                "periodo": "Mismo mes",
                "mes": MONTHS_ES[ref_month],
                "n_años": len(common_years),
                "subio_cuando_ref_bajo": int(subio),
                "pct_subio": round(subio / len(common_years) * 100, 1),
                "retorno_prom": round(vals_when_ref_down.mean(), 2),
            })

        # ── Mes siguiente ──
        next_month = ref_month % 12 + 1
        next_month_data = col[col.index.month == next_month]
        # Si es diciembre → enero siguiente año
        if ref_month == 12:
            next_years = [y + 1 for y in ref_down_years]
        else:
            next_years = ref_down_years

        common_next = [y for y in next_years if y in next_month_data.index.year.tolist()]
        if len(common_next) >= min_overlap:
            vals_next = next_month_data[next_month_data.index.year.isin(common_next)]
            subio_next = (vals_next > 0).sum()
            rows.append({
                "ticker": ticker,
                "periodo": "Mes siguiente",
                "mes": MONTHS_ES[next_month],
                "n_años": len(common_next),
                "subio_cuando_ref_bajo": int(subio_next),
                "pct_subio": round(subio_next / len(common_next) * 100, 1),
                "retorno_prom": round(vals_next.mean(), 2),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("pct_subio", ascending=False).reset_index(drop=True)


# ─── CHART HELPERS ────────────────────────────────────────────────────────────
DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#12121a",
    font=dict(family="Space Mono, monospace", color="#e2e8f0"),
)


def chart_monthly_win_rate(stats_df: pd.DataFrame, month: int, top_n: int = 20, direction: str = "sube"):
    filtered = stats_df[stats_df["month"] == month].copy()
    if filtered.empty:
        return None

    if direction == "sube":
        filtered = filtered.nlargest(top_n, "pct_sube")
        x_col, color_col = "pct_sube", "pct_sube"
        title = f"Top {top_n} acciones que MÁS SUBEN en {MONTHS_ES[month]}"
        colorscale = [[0, "#10b981"], [1, "#34d399"]]
        xrange = [0, 100]
    else:
        filtered = filtered.nsmallest(top_n, "pct_sube")
        x_col, color_col = "pct_sube", "pct_sube"
        title = f"Top {top_n} acciones que MÁS BAJAN en {MONTHS_ES[month]}"
        colorscale = [[0, "#f43f5e"], [1, "#fb7185"]]
        xrange = [0, 100]

    filtered = filtered.sort_values(x_col)
    filtered["label"] = filtered.apply(
        lambda r: f"{r['veces_subio']}/{r['total_años']} años subió", axis=1
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered[x_col],
        y=filtered["ticker"],
        orientation="h",
        text=filtered["label"],
        textposition="outside",
        textfont=dict(size=10),
        marker=dict(
            color=filtered[color_col],
            colorscale=colorscale,
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Win rate: %{x:.1f}%<br>"
            "Retorno prom: %{customdata[0]:.2f}%<br>"
            "Retorno mediano: %{customdata[1]:.2f}%<extra></extra>"
        ),
        customdata=filtered[["retorno_prom", "retorno_med"]].values,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Syne, sans-serif")),
        xaxis=dict(title="% de años que subió", range=xrange, gridcolor="#2a2a3e"),
        yaxis=dict(tickfont=dict(family="Space Mono")),
        height=max(400, top_n * 28),
        margin=dict(l=60, r=120, t=60, b=40),
        **DARK_TEMPLATE
    )
    return fig


def chart_heatmap(stats_df: pd.DataFrame, value_col: str = "pct_sube", top_n: int = 30):
    pivot = stats_df.pivot_table(index="ticker", columns="month", values=value_col, aggfunc="first")
    pivot.columns = [MONTHS_SHORT[m] for m in pivot.columns]

    # Ordenar por promedio anual
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.nlargest(top_n, "avg").drop(columns="avg")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="RdYlGn",
        zmid=50,
        text=np.round(pivot.values, 0).astype(str),
        texttemplate="%{text}%",
        textfont=dict(size=9),
        colorbar=dict(title="% sube", tickfont=dict(family="Space Mono")),
        hovertemplate="<b>%{y}</b> — %{x}<br>Win rate: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Heatmap de Win Rate mensual", font=dict(size=16, family="Syne, sans-serif")),
        xaxis=dict(side="top"),
        yaxis=dict(tickfont=dict(family="Space Mono", size=10)),
        height=max(500, top_n * 22),
        margin=dict(l=80, r=60, t=80, b=40),
        **DARK_TEMPLATE
    )
    return fig


def chart_single_ticker(returns: pd.DataFrame, ticker: str):
    col = returns[ticker].dropna().copy()
    col.index = pd.to_datetime(col.index)

    monthly_grouped = {}
    for month in range(1, 13):
        vals = col[col.index.month == month]
        monthly_grouped[MONTHS_ES[month]] = {
            "años": list(vals.index.year),
            "retornos": list(vals.values),
        }

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[MONTHS_ES[m] for m in range(1, 13)],
        vertical_spacing=0.12, horizontal_spacing=0.08
    )

    for i, month in enumerate(range(1, 13)):
        row = (i // 4) + 1
        col_num = (i % 4) + 1
        vals = col[col.index.month == month]
        colors = ["#10b981" if v > 0 else "#f43f5e" for v in vals.values]

        fig.add_trace(
            go.Bar(
                x=vals.index.year.tolist(),
                y=vals.values.tolist(),
                marker_color=colors,
                name=MONTHS_ES[month],
                showlegend=False,
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ),
            row=row, col=col_num
        )
        fig.add_hline(y=0, line_dash="solid", line_color="#2a2a3e",
                      line_width=1, row=row, col=col_num)

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Rendimiento por mes (cada año)",
            font=dict(size=16, family="Syne, sans-serif")
        ),
        height=700,
        margin=dict(l=40, r=40, t=80, b=40),
        **DARK_TEMPLATE
    )
    return fig


def chart_correlation_results(corr_df: pd.DataFrame, ref_ticker: str, ref_month: int):
    if corr_df.empty:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Mismo mes ({MONTHS_ES[ref_month]})",
        f"Mes siguiente ({MONTHS_ES[ref_month % 12 + 1]})"
    ])

    for col_idx, periodo in enumerate(["Mismo mes", "Mes siguiente"]):
        sub = corr_df[corr_df["periodo"] == periodo].head(15)
        if sub.empty:
            continue
        sub = sub.sort_values("pct_subio")
        colors = ["#10b981" if p >= 60 else "#6366f1" if p >= 50 else "#f43f5e"
                  for p in sub["pct_subio"]]
        fig.add_trace(
            go.Bar(
                x=sub["pct_subio"],
                y=sub["ticker"],
                orientation="h",
                marker_color=colors,
                text=sub.apply(lambda r: f"{r['subio_cuando_ref_bajo']}/{r['n_años']} ({r['retorno_prom']:+.1f}%)", axis=1),
                textposition="outside",
                textfont=dict(size=9),
                name=periodo,
                hovertemplate="<b>%{y}</b><br>Subió %{x:.1f}% de las veces<extra></extra>",
            ),
            row=1, col=col_idx + 1
        )

    fig.update_layout(
        title=dict(
            text=f"Correlaciones: cuando {ref_ticker} BAJA en {MONTHS_ES[ref_month]}, ¿qué otras acciones suben?",
            font=dict(size=14, family="Syne, sans-serif")
        ),
        height=500,
        showlegend=False,
        margin=dict(l=60, r=120, t=80, b=40),
        **DARK_TEMPLATE
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(range=[0, 110], gridcolor="#2a2a3e")})
    return fig


def chart_yearly_monthly_heatmap(returns: pd.DataFrame, ticker: str):
    col = returns[ticker].dropna()
    col.index = pd.to_datetime(col.index)

    years = sorted(col.index.year.unique())
    months = list(range(1, 13))

    matrix = []
    for year in years:
        row = []
        for month in months:
            vals = col[(col.index.year == year) & (col.index.month == month)]
            row.append(vals.values[0] if len(vals) > 0 else np.nan)
        matrix.append(row)

    z = np.array(matrix, dtype=float)
    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str) + "%")

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[MONTHS_SHORT[m] for m in months],
        y=years,
        colorscale="RdYlGn",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Retorno %", tickfont=dict(family="Space Mono")),
        hovertemplate="Año %{y} — %{x}<br>Retorno: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Mapa de calor año × mes",
            font=dict(size=16, family="Syne, sans-serif")
        ),
        yaxis=dict(tickfont=dict(family="Space Mono", size=10), dtick=1),
        xaxis=dict(side="top"),
        height=600,
        margin=dict(l=60, r=60, t=80, b=40),
        **DARK_TEMPLATE
    )
    return fig


def chart_sector_monthly_comparison(stats_df: pd.DataFrame, available_tickers: list[str],
                                     selected_sectors: list[str], metric: str = "pct_sube"):
    """Crea un gráfico de líneas comparando sectores mes a mes."""
    sectors_data = get_available_sectors(available_tickers)

    rows = []
    for sector in selected_sectors:
        sector_tickers = sectors_data.get(sector, [])
        if not sector_tickers:
            continue
        sector_stats = stats_df[stats_df["ticker"].isin(sector_tickers)]
        for month in range(1, 13):
            month_data = sector_stats[sector_stats["month"] == month]
            if month_data.empty:
                continue
            rows.append({
                "sector": sector,
                "month": month,
                "month_name": MONTHS_ES[month],
                "valor": month_data[metric].mean(),
                "n_tickers": len(month_data),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    metric_label = "Win Rate %" if metric == "pct_sube" else "Retorno Promedio %"
    fig = px.line(
        df, x="month_name", y="valor", color="sector",
        markers=True,
        labels={"valor": metric_label, "month_name": "Mes", "sector": "Sector"},
        title=f"Comparativa de sectores — {metric_label} por mes",
        category_orders={"month_name": [MONTHS_ES[m] for m in range(1, 13)]},
    )
    if metric == "pct_sube":
        fig.add_hline(y=50, line_dash="dash", line_color="#6366f1", line_width=1,
                      annotation_text="50%", annotation_position="right")
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="#6366f1", line_width=1)

    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **DARK_TEMPLATE
    )
    return fig


def chart_sector_heatmap(stats_df: pd.DataFrame, available_tickers: list[str],
                          selected_sectors: list[str], metric: str = "pct_sube"):
    """Heatmap de sectores × meses."""
    sectors_data = get_available_sectors(available_tickers)

    matrix = []
    sector_labels = []
    for sector in selected_sectors:
        sector_tickers = sectors_data.get(sector, [])
        if not sector_tickers:
            continue
        sector_stats = stats_df[stats_df["ticker"].isin(sector_tickers)]
        row = []
        for month in range(1, 13):
            month_data = sector_stats[sector_stats["month"] == month]
            row.append(month_data[metric].mean() if not month_data.empty else np.nan)
        matrix.append(row)
        sector_labels.append(f"{sector} ({len(sector_tickers)})")

    if not matrix:
        return None

    z = np.array(matrix, dtype=float)

    if metric == "pct_sube":
        zmid, colorbar_title = 50, "Win Rate %"
        text = np.where(np.isnan(z), "", np.round(z, 1).astype(str) + "%")
    else:
        zmid, colorbar_title = 0, "Ret. Prom %"
        text = np.where(np.isnan(z), "", np.round(z, 2).astype(str) + "%")

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[MONTHS_SHORT[m] for m in range(1, 13)],
        y=sector_labels,
        colorscale="RdYlGn",
        zmid=zmid,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title=colorbar_title, tickfont=dict(family="Space Mono")),
        hovertemplate="<b>%{y}</b> — %{x}<br>Valor: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Heatmap por sector — {colorbar_title}",
            font=dict(size=16, family="Syne, sans-serif")
        ),
        xaxis=dict(side="top"),
        yaxis=dict(tickfont=dict(family="Space Mono", size=11)),
        height=max(350, len(selected_sectors) * 50),
        margin=dict(l=140, r=60, t=80, b=40),
        **DARK_TEMPLATE
    )
    return fig


def chart_sector_best_month(stats_df: pd.DataFrame, available_tickers: list[str],
                              selected_sectors: list[str], month: int):
    """Bar chart comparando sectores para un mes específico."""
    sectors_data = get_available_sectors(available_tickers)

    rows = []
    for sector in selected_sectors:
        sector_tickers = sectors_data.get(sector, [])
        if not sector_tickers:
            continue
        sector_stats = stats_df[
            (stats_df["ticker"].isin(sector_tickers)) & (stats_df["month"] == month)
        ]
        if sector_stats.empty:
            continue
        rows.append({
            "sector": sector,
            "win_rate": sector_stats["pct_sube"].mean(),
            "retorno_prom": sector_stats["retorno_prom"].mean(),
            "n_tickers": len(sector_stats),
            "mejor_ticker": sector_stats.loc[sector_stats["pct_sube"].idxmax(), "ticker"],
            "mejor_wr": sector_stats["pct_sube"].max(),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("win_rate", ascending=True)
    colors = ["#10b981" if wr >= 55 else "#6366f1" if wr >= 45 else "#f43f5e"
              for wr in df["win_rate"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["win_rate"],
        y=df["sector"],
        orientation="h",
        marker_color=colors,
        text=df.apply(lambda r: f"{r['win_rate']:.1f}% (mejor: {r['mejor_ticker']} {r['mejor_wr']:.0f}%)", axis=1),
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Win rate: %{x:.1f}%<br>"
            "Ret. prom: %{customdata[0]:+.2f}%<br>"
            "N tickers: %{customdata[1]}<extra></extra>"
        ),
        customdata=df[["retorno_prom", "n_tickers"]].values,
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="#6366f1", line_width=1)
    fig.update_layout(
        title=dict(
            text=f"Win rate por sector en {MONTHS_ES[month]}",
            font=dict(size=16, family="Syne, sans-serif")
        ),
        xaxis=dict(title="Win Rate %", range=[0, 100], gridcolor="#2a2a3e"),
        yaxis=dict(tickfont=dict(family="Space Mono")),
        height=max(350, len(df) * 45),
        margin=dict(l=100, r=180, t=60, b=40),
        **DARK_TEMPLATE
    )
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    ticker_file = st.text_input("📄 Archivo de tickers (.txt)", value="tickers.txt")

    years_back = st.slider("📅 Años de historia", min_value=5, max_value=18, value=18)

    st.markdown("---")
    load_btn = st.button("🚀 Cargar / Actualizar datos", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#64748b; line-height:1.8'>
    <b>Cómo funciona:</b><br>
    1. Descarga precios históricos de Yahoo Finance<br>
    2. Calcula rendimiento mensual de cada acción<br>
    3. Analiza frecuencia de subidas/bajadas por mes<br>
    4. Encuentra correlaciones inversas y directas
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Monthly Stock Analyzer")
st.markdown("<div style='color:#64748b; font-family:Space Mono; font-size:0.85rem'>Análisis de rendimiento mensual · Correlaciones históricas · 18 años</div>", unsafe_allow_html=True)
st.markdown("---")

# Load tickers
tickers = []
if os.path.exists(ticker_file):
    tickers = load_tickers_from_file(ticker_file)
    st.sidebar.success(f"✅ {len(tickers)} tickers cargados")
else:
    st.sidebar.error(f"❌ No se encontró: {ticker_file}")
    st.error(f"Coloca tu archivo `{ticker_file}` en el mismo directorio que esta app.")
    st.stop()

# Download / load data
if "prices" not in st.session_state or load_btn:
    with st.spinner("Descargando datos históricos... esto puede tardar unos minutos"):
        prices = download_data(tickers, years=years_back)
        if prices.empty:
            st.error("No se pudieron descargar datos. Verifica tu conexión a internet.")
            st.stop()
        returns = compute_monthly_returns(prices)
        stats = get_monthly_stats(returns)

        # Guardar en session
        st.session_state["prices"] = prices
        st.session_state["returns"] = returns
        st.session_state["stats"] = stats
        available_tickers = list(prices.columns)
        st.session_state["available_tickers"] = available_tickers
        st.success(f"✅ Datos cargados: {len(available_tickers)} acciones · {len(prices)} meses")

prices = st.session_state.get("prices", pd.DataFrame())
returns = st.session_state.get("returns", pd.DataFrame())
stats = st.session_state.get("stats", pd.DataFrame())
available_tickers = st.session_state.get("available_tickers", [])

if prices.empty:
    st.info("👈 Presiona **Cargar / Actualizar datos** para comenzar.")
    st.stop()

# ─── KPI METRICS ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
start_date = prices.index.min().strftime("%b %Y")
end_date = prices.index.max().strftime("%b %Y")

with col1:
    st.metric("Acciones analizadas", f"{len(available_tickers)}")
with col2:
    st.metric("Período", f"{start_date} → {end_date}")
with col3:
    best_month_row = stats.groupby("month")["pct_sube"].mean().idxmax()
    st.metric("Mejor mes histórico", MONTHS_ES[best_month_row])
with col4:
    worst_month_row = stats.groupby("month")["pct_sube"].mean().idxmin()
    st.metric("Peor mes histórico", MONTHS_ES[worst_month_row])


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗓️ Análisis por Mes",
    "🔥 Heatmap General",
    "🔍 Correlaciones",
    "📊 Ticker Individual",
    "🏭 Sectores"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Análisis por Mes
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">FILTROS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        selected_month = st.selectbox(
            "Seleccionar mes",
            options=list(range(1, 13)),
            format_func=lambda m: MONTHS_ES[m],
            index=2  # Marzo por defecto
        )
    with c2:
        direction = st.radio("Mostrar acciones que...", ["Más suben", "Más bajan"], horizontal=True)
    with c3:
        top_n = st.slider("Top N", 10, 30, 20)
    with c4:
        min_years = st.slider("Mín. años de datos", 3, 15, 8)

    # Filtrar por mínimo de años
    filtered_stats = stats[stats["total_años"] >= min_years].copy()

    direction_key = "sube" if direction == "Más suben" else "baja"
    fig = chart_monthly_win_rate(filtered_stats, selected_month, top_n, direction_key)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Tabla detallada
    st.markdown('<div class="section-header">TABLA DETALLADA</div>', unsafe_allow_html=True)
    month_stats = filtered_stats[filtered_stats["month"] == selected_month].copy()
    sort_col = "pct_sube" if direction_key == "sube" else "pct_sube"
    sort_asc = direction_key != "sube"
    month_stats = month_stats.sort_values(sort_col, ascending=sort_asc)

    display_cols = ["ticker", "veces_subio", "veces_bajo", "total_años",
                    "pct_sube", "retorno_prom", "retorno_med", "retorno_max", "retorno_min"]
    display_df = month_stats[display_cols].copy()
    display_df.columns = ["Ticker", "Años ↑", "Años ↓", "Total", "Win Rate %",
                           "Ret. Prom %", "Ret. Med %", "Máx %", "Mín %"]

    def color_row(val, col):
        if col in ["Win Rate %", "Ret. Prom %", "Ret. Med %"]:
            if isinstance(val, (int, float)):
                if (col == "Win Rate %" and val > 55) or (col != "Win Rate %" and val > 0):
                    return "color: #10b981"
                elif (col == "Win Rate %" and val < 45) or (col != "Win Rate %" and val < 0):
                    return "color: #f43f5e"
        return ""

    st.dataframe(
        display_df.head(top_n).style.format({
            "Win Rate %": "{:.1f}%",
            "Ret. Prom %": "{:+.2f}%",
            "Ret. Med %": "{:+.2f}%",
            "Máx %": "{:+.2f}%",
            "Mín %": "{:+.2f}%",
        }).background_gradient(subset=["Win Rate %"], cmap="RdYlGn", vmin=20, vmax=80),
        use_container_width=True,
        height=400
    )

    # Win rate por mes — resumen general
    st.markdown('<div class="section-header">WIN RATE PROMEDIO DE TODAS LAS ACCIONES POR MES</div>', unsafe_allow_html=True)
    avg_by_month = stats.groupby("month").agg(
        win_rate_avg=("pct_sube", "mean"),
        retorno_avg=("retorno_prom", "mean")
    ).reset_index()
    avg_by_month["mes"] = avg_by_month["month"].map(MONTHS_ES)

    fig_bar = go.Figure()
    colors_bar = ["#10b981" if v > 50 else "#f43f5e" for v in avg_by_month["win_rate_avg"]]
    fig_bar.add_trace(go.Bar(
        x=avg_by_month["mes"],
        y=avg_by_month["win_rate_avg"],
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in avg_by_month["win_rate_avg"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>",
    ))
    fig_bar.add_hline(y=50, line_dash="dash", line_color="#6366f1", line_width=1.5,
                      annotation_text="50%", annotation_position="right")
    fig_bar.update_layout(
        title="Win rate promedio del mercado por mes",
        yaxis=dict(title="%", range=[30, 70], gridcolor="#2a2a3e"),
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        **DARK_TEMPLATE
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Heatmap General
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">HEATMAP WIN RATE POR ACCIÓN Y MES</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        heatmap_n = st.slider("Número de acciones", 10, len(available_tickers), min(40, len(available_tickers)))
        heatmap_metric = st.radio("Métrica", ["% Win Rate", "Retorno promedio %"], key="hm_metric")

    val_col = "pct_sube" if heatmap_metric == "% Win Rate" else "retorno_prom"
    fig_hm = chart_heatmap(stats[stats["total_años"] >= 8], val_col, heatmap_n)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<div class="section-header">ESTACIONALIDAD — MEJORES Y PEORES MESES POR ACCIÓN</div>', unsafe_allow_html=True)

    sel_tickers_hm = st.multiselect(
        "Seleccionar acciones para comparar (máx 8)",
        available_tickers,
        default=available_tickers[:6] if len(available_tickers) >= 6 else available_tickers,
        max_selections=8,
        key="hm_tickers"
    )

    if sel_tickers_hm:
        sub = stats[stats["ticker"].isin(sel_tickers_hm)].copy()
        pivot_comp = sub.pivot_table(index="ticker", columns="month_name", values="pct_sube")
        month_order = [MONTHS_ES[m] for m in range(1, 13)]
        pivot_comp = pivot_comp.reindex(columns=[m for m in month_order if m in pivot_comp.columns])

        fig_comp = px.line(
            pivot_comp.T.reset_index().melt(id_vars="month_name", var_name="ticker", value_name="win_rate"),
            x="month_name", y="win_rate", color="ticker",
            markers=True,
            labels={"win_rate": "Win Rate %", "month_name": "Mes"},
            title="Comparativa de estacionalidad",
        )
        fig_comp.add_hline(y=50, line_dash="dash", line_color="#6366f1", line_width=1)
        fig_comp.update_layout(height=400, **DARK_TEMPLATE)
        st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Correlaciones
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='background:#12121a; border:1px solid #2a2a3e; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1.5rem'>
    <b style='color:#6366f1'>¿Cómo funciona?</b><br>
    <span style='font-size:0.85rem; color:#94a3b8'>
    Seleccionás una acción de referencia y un mes. El análisis te muestra qué otras acciones subieron 
    con mayor frecuencia cuando la acción de referencia bajó ese mes, tanto en el mismo mes como en el mes siguiente.
    </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        ref_ticker = st.selectbox("📌 Acción de referencia", available_tickers, index=0)
    with c2:
        ref_month = st.selectbox(
            "📅 Mes de análisis",
            options=list(range(1, 13)),
            format_func=lambda m: MONTHS_ES[m],
            index=2,
            key="corr_month"
        )
    with c3:
        min_overlap = st.slider("Mín. años en común", 3, 15, 8, key="corr_min")

    # Stats del ticker de referencia ese mes
    ref_stats = stats[(stats["ticker"] == ref_ticker) & (stats["month"] == ref_month)]
    if not ref_stats.empty:
        rs = ref_stats.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{ref_ticker} — {MONTHS_ES[ref_month]}", f"Win rate: {rs['pct_sube']:.1f}%")
        c2.metric("Años subió", f"{rs['veces_subio']} / {rs['total_años']}")
        c3.metric("Ret. promedio", f"{rs['retorno_prom']:+.2f}%")
        c4.metric("Ret. mediano", f"{rs['retorno_med']:+.2f}%")

    with st.spinner("Calculando correlaciones..."):
        corr_df = find_correlations(returns, ref_ticker, ref_month, min_overlap)

    if corr_df.empty:
        st.warning("No hay suficientes datos para calcular correlaciones con los parámetros seleccionados.")
    else:
        fig_corr = chart_correlation_results(corr_df, ref_ticker, ref_month)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)

        # Tabla detallada
        st.markdown('<div class="section-header">RESULTADOS DETALLADOS</div>', unsafe_allow_html=True)

        tab_mismo, tab_siguiente = st.tabs([
            f"📍 Mismo mes ({MONTHS_ES[ref_month]})",
            f"➡️ Mes siguiente ({MONTHS_ES[ref_month % 12 + 1]})"
        ])

        for tab_c, periodo in [(tab_mismo, "Mismo mes"), (tab_siguiente, "Mes siguiente")]:
            with tab_c:
                sub = corr_df[corr_df["periodo"] == periodo].copy()
                if sub.empty:
                    st.info("Sin datos suficientes para este período.")
                    continue

                sub_display = sub[["ticker", "n_años", "subio_cuando_ref_bajo", "pct_subio", "retorno_prom"]].copy()
                sub_display.columns = ["Ticker", "Años en común", "Veces subió", "% Veces subió", "Ret. Prom %"]

                st.dataframe(
                    sub_display.style.format({
                        "% Veces subió": "{:.1f}%",
                        "Ret. Prom %": "{:+.2f}%",
                    }).background_gradient(subset=["% Veces subió"], cmap="RdYlGn", vmin=30, vmax=80),
                    use_container_width=True,
                    height=400
                )

                # Resumen narrativo top 3
                top3 = sub.head(3)
                st.markdown(f"**Top 3 acciones que más suben cuando {ref_ticker} baja en {MONTHS_ES[ref_month]}:**")
                for _, row in top3.iterrows():
                    trend = "🟢" if row["pct_subio"] >= 60 else "🟡" if row["pct_subio"] >= 50 else "🔴"
                    st.markdown(
                        f"{trend} **{row['ticker']}** subió {row['subio_cuando_ref_bajo']} de {row['n_años']} años "
                        f"({row['pct_subio']:.1f}%) · retorno promedio: **{row['retorno_prom']:+.2f}%**"
                    )

    # ── Matriz de correlación entre tickers ──
    st.markdown('<div class="section-header">CORRELACIÓN ENTRE ACCIONES (RETORNOS MENSUALES)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        corr_n = st.slider("N acciones", 5, 25, 15, key="corr_n_matrix")
    with c1:
        selected_for_corr = st.multiselect(
            "Acciones para la matriz (o usa las top por actividad)",
            available_tickers,
            default=[],
            key="corr_matrix_sel"
        )

    if selected_for_corr:
        tickers_for_corr = selected_for_corr
    else:
        # Usar las que tienen más datos
        tickers_for_corr = returns.count().nlargest(corr_n).index.tolist()

    corr_matrix = returns[tickers_for_corr].corr()
    fig_matrix = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],  # Reemplaza a zmid=0
        aspect="auto",
        title="Correlación de retornos mensuales entre acciones",
    )
    fig_matrix.update_layout(height=500, **DARK_TEMPLATE)
    st.plotly_chart(fig_matrix, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Ticker Individual
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns([2, 3])
    with c1:
        sel_ticker = st.selectbox("🔍 Seleccionar acción", available_tickers, key="ind_ticker")

    ticker_stats = stats[stats["ticker"] == sel_ticker].sort_values("month")

    if not ticker_stats.empty:
        # KPIs
        best_month_t = ticker_stats.loc[ticker_stats["pct_sube"].idxmax()]
        worst_month_t = ticker_stats.loc[ticker_stats["pct_sube"].idxmin()]
        avg_ret = ticker_stats["retorno_prom"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor mes", f"{best_month_t['month_name']}", f"{best_month_t['pct_sube']:.0f}% win rate")
        c2.metric("Peor mes", f"{worst_month_t['month_name']}", f"{worst_month_t['pct_sube']:.0f}% win rate")
        c3.metric("Ret. prom. anualizado", f"{avg_ret * 12:.1f}%")
        c4.metric("Meses analizados", f"{ticker_stats['total_años'].max()}")

        # Heatmap año × mes
        st.plotly_chart(chart_yearly_monthly_heatmap(returns, sel_ticker), use_container_width=True)

        # Mini subplots por mes
        st.plotly_chart(chart_single_ticker(returns, sel_ticker), use_container_width=True)

        # Tabla mensual
        st.markdown('<div class="section-header">ESTADÍSTICAS MENSUALES</div>', unsafe_allow_html=True)
        display_ticker = ticker_stats[["month_name", "veces_subio", "veces_bajo", "total_años",
                                        "pct_sube", "retorno_prom", "retorno_med",
                                        "retorno_max", "retorno_min", "std"]].copy()
        display_ticker.columns = ["Mes", "↑ Años", "↓ Años", "Total",
                                    "Win Rate %", "Ret. Prom %", "Mediana %",
                                    "Máx %", "Mín %", "Std %"]
        st.dataframe(
            display_ticker.style.format({
                "Win Rate %": "{:.1f}%",
                "Ret. Prom %": "{:+.2f}%",
                "Mediana %": "{:+.2f}%",
                "Máx %": "{:+.2f}%",
                "Mín %": "{:+.2f}%",
                "Std %": "{:.2f}%",
            }).background_gradient(subset=["Win Rate %"], cmap="RdYlGn", vmin=25, vmax=75),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Sectores
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style='background:#12121a; border:1px solid #2a2a3e; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1.5rem'>
    <b style='color:#6366f1'>Comparación por Sector</b><br>
    <span style='font-size:0.85rem; color:#94a3b8'>
    Compará el rendimiento mensual promedio de distintos sectores (aerolíneas, semiconductores, bancos, mineras, etc.).
    Se agrupa cada ticker por sector y se calcula el win rate o retorno promedio del sector para cada mes.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Obtener sectores disponibles
    sectors_available = get_available_sectors(available_tickers)

    # Mostrar resumen de sectores
    sector_summary_cols = st.columns(min(len(sectors_available), 6))
    for i, (sector, tickers_list) in enumerate(sectors_available.items()):
        col_idx = i % len(sector_summary_cols)
        with sector_summary_cols[col_idx]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value neutral' style='font-size:1.3rem'>{len(tickers_list)}</div>"
                f"<div class='metric-label'>{sector}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("")

    # Selector de sectores
    all_sector_names = list(sectors_available.keys())
    default_sectors = [s for s in all_sector_names if s != "Otros"][:6]

    selected_sectors = st.multiselect(
        "Seleccionar sectores a comparar",
        all_sector_names,
        default=default_sectors,
        key="sector_select"
    )

    if not selected_sectors:
        st.info("Seleccioná al menos un sector para ver la comparación.")
    else:
        # ── Controles ──
        c1, c2 = st.columns([2, 2])
        with c1:
            sector_metric = st.radio(
                "Métrica", ["Win Rate %", "Retorno Promedio %"],
                horizontal=True, key="sector_metric"
            )
        with c2:
            sector_month_filter = st.selectbox(
                "Mes para ranking de sectores",
                options=list(range(1, 13)),
                format_func=lambda m: MONTHS_ES[m],
                index=0,
                key="sector_month"
            )

        metric_col = "pct_sube" if sector_metric == "Win Rate %" else "retorno_prom"

        # ── Heatmap Sector × Mes ──
        st.markdown('<div class="section-header">HEATMAP SECTOR × MES</div>', unsafe_allow_html=True)
        fig_sector_hm = chart_sector_heatmap(stats, available_tickers, selected_sectors, metric_col)
        if fig_sector_hm:
            st.plotly_chart(fig_sector_hm, use_container_width=True)

        # ── Líneas comparativas ──
        st.markdown('<div class="section-header">EVOLUCIÓN MENSUAL POR SECTOR</div>', unsafe_allow_html=True)
        fig_sector_lines = chart_sector_monthly_comparison(stats, available_tickers, selected_sectors, metric_col)
        if fig_sector_lines:
            st.plotly_chart(fig_sector_lines, use_container_width=True)

        # ── Ranking de sectores para un mes ──
        st.markdown(f'<div class="section-header">RANKING DE SECTORES — {MONTHS_ES[sector_month_filter].upper()}</div>', unsafe_allow_html=True)
        fig_sector_bar = chart_sector_best_month(stats, available_tickers, selected_sectors, sector_month_filter)
        if fig_sector_bar:
            st.plotly_chart(fig_sector_bar, use_container_width=True)

        # ── Tabla detallada por sector y mes ──
        st.markdown('<div class="section-header">DETALLE POR SECTOR Y MES</div>', unsafe_allow_html=True)

        detail_sector = st.selectbox(
            "Seleccionar sector para ver detalle",
            selected_sectors,
            key="sector_detail"
        )

        sector_tickers_detail = sectors_available.get(detail_sector, [])
        if sector_tickers_detail:
            st.markdown(
                f"**Tickers en {detail_sector}:** {', '.join(sector_tickers_detail)}",
            )

            detail_stats = stats[stats["ticker"].isin(sector_tickers_detail)].copy()

            # Pivot: ticker × mes con win rate
            pivot_detail = detail_stats.pivot_table(
                index="ticker", columns="month", values=metric_col, aggfunc="first"
            )
            pivot_detail.columns = [MONTHS_SHORT[m] for m in pivot_detail.columns]
            pivot_detail["Promedio"] = pivot_detail.mean(axis=1)
            pivot_detail = pivot_detail.sort_values("Promedio", ascending=False)

            if metric_col == "pct_sube":
                fmt = "{:.1f}%"
                vmin, vmax = 25, 75
            else:
                fmt = "{:+.2f}%"
                vmin, vmax = -5, 5

            format_dict = {c: fmt for c in pivot_detail.columns}

            st.dataframe(
                pivot_detail.style.format(format_dict).background_gradient(
                    cmap="RdYlGn", vmin=vmin, vmax=vmax
                ),
                use_container_width=True,
                height=min(400, max(200, len(pivot_detail) * 35 + 50))
            )


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-family:Space Mono; font-size:0.7rem; color:#2a2a3e'>"
    "Datos via Yahoo Finance · Análisis histórico no garantiza resultados futuros"
    "</div>",
    unsafe_allow_html=True
)
