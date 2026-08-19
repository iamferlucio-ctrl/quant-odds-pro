import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import exp, factorial
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# CONFIGURACIÓN Y ESTILOS DE TERMINAL INSTITUCIONAL
# ==============================================================================
st.set_page_config(
    page_title="QuantOdds Terminal Pro + Odds API",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    .mono { font-family: 'JetBrains Mono', monospace; }
    
    /* Header Principal */
    .match-header {
        background: linear-gradient(180deg, #151C28 0%, #0D121D 100%);
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        text-align: center;
    }
    .match-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .match-subtitle {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Panel de Pronóstico Direccional */
    .forecast-container {
        background: #111823;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .forecast-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 6px;
    }
    .forecast-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    @media (min-width: 768px) {
        .forecast-grid { grid-template-columns: repeat(4, 1fr); }
    }
    .forecast-card {
        background: #0B0E14;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px;
    }
    .forecast-label {
        font-size: 0.68rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .forecast-val {
        font-size: 1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 2px;
    }
    .forecast-sub {
        font-size: 0.72rem;
        color: #10B981;
    }

    /* Grid de KPIs */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-bottom: 16px;
    }
    @media (min-width: 768px) {
        .kpi-grid { grid-template-columns: repeat(5, 1fr); }
    }
    .kpi-card {
        background: #111722;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.68rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.15rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .val-positive { color: #10B981; }
    .val-negative { color: #EF4444; }
    .val-neutral { color: #F59E0B; }
    .val-white { color: #F8FAFC; }

    /* Order Ticket */
    .ticket-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid #1E293B;
    }
    .ticket-no-trade {
        background: rgba(239, 68, 68, 0.05);
        border-color: rgba(239, 68, 68, 0.3);
    }
    .ticket-trade {
        background: rgba(16, 185, 129, 0.05);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .ticket-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }
    .badge-no-trade { background: #EF4444; color: #FFF; }
    .badge-trade { background: #10B981; color: #FFF; }

    .exec-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    @media (min-width: 768px) {
        .exec-grid { grid-template-columns: repeat(4, 1fr); }
    }
    .exec-item-label { font-size: 0.7rem; color: #94A3B8; }
    .exec-item-val { font-size: 0.95rem; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace; }

    .warning-chip {
        background: rgba(245, 158, 11, 0.1);
        border-left: 3px solid #F59E0B;
        padding: 8px 12px;
        font-size: 0.8rem;
        color: #FCD34D;
        margin-top: 10px;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# INTEGRACIÓN API (THE ODDS API)
# ==============================================================================
SPORTS_DICT = {
    "Premier League (Inglaterra)": "soccer_epl",
    "La Liga (España)": "soccer_spain_la_liga",
    "Serie A (Italia)": "soccer_italy_serie_a",
    "Bundesliga (Alemania)": "soccer_germany_bundesliga",
    "Ligue 1 (Francia)": "soccer_france_ligue_one",
    "UEFA Champions League": "soccer_uefa_champs_league",
    "Copa Libertadores": "soccer_conmebol_copa_libertadores",
    "MLS (EE.UU.)": "soccer_usa_mls",
    "Liga Profesional (Argentina)": "soccer_argentina_primera_division",
    "Brasileirão Serie A": "soccer_brazil_campeonato"
}

@st.cache_data(ttl=300)
def fetch_odds_api(api_key: str, sport_key: str, region: str = "eu") -> Tuple[Optional[List[Dict]], str]:
    if not api_key:
        return None, "Falta API Key."
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={region}&markets=h2h,spreads,totals&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            remaining = res.headers.get("x-requests-remaining", "N/A")
            return data, f"Conexión exitosa. Consultas restantes: {remaining}"
        elif res.status_code == 401:
            return None, "API Key inválida o no autorizada."
        else:
            return None, f"Error API HTTP {res.status_code}"
    except Exception as e:
        return None, f"Error de red: {str(e)}"

# ==============================================================================
# MOTORES MATEMÁTICOS Y CONCILIACIÓN
# ==============================================================================
class DixonColesEngine:
    @staticmethod
    def tau(x: int, y: int, lh: float, ma: float, rho: float) -> float:
        if x == 0 and y == 0: return 1.0 - (lh * ma * rho)
        elif x == 0 and y == 1: return 1.0 + (lh * rho)
        elif x == 1 and y == 0: return 1.0 + (ma * rho)
        elif x == 1 and y == 1: return 1.0 - rho
        return 1.0

    @classmethod
    def generate_matrix(cls, lh: float, ma: float, rho: float = -0.05, max_g: int = 6) -> np.ndarray:
        m = np.zeros((max_g, max_g))
        for x in range(max_g):
            for y in range(max_g):
                px = (exp(-lh) * (lh ** x)) / factorial(x)
                py = (exp(-ma) * (ma ** y)) / factorial(y)
                m[x, y] = max(0.0, px * py * cls.tau(x, y, lh, ma, rho))
        s = np.sum(m)
        return m / s if s > 0 else m

class ShinEngine:
    @classmethod
    def deoverround(cls, o1: float, ox: float, o2: float) -> Tuple[float, float, float, float]:
        if o1 <= 1.0 or ox <= 1.0 or o2 <= 1.0: return 0.333, 0.333, 0.333, 0.0
        raw = np.array([1.0/o1, 1.0/ox, 1.0/o2])
        beta = np.sum(raw)
        z = max(0.0001, (beta - 1.0) / 0.25)
        for _ in range(20):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (raw / beta)) - z) / (2 * (1 - z))
            p = p / np.sum(p)
            n_beta = np.sum(raw / (z + (1 - z) * p))
            if abs(n_beta - beta) < 1e-6: break
            z = max(0.0001, min(0.4, z + (beta - n_beta) * 0.1))
        return float(p[0]), float(p[1]), float(p[2]), float(z)

class ReverseEngine:
    @classmethod
    def extract_xg(cls, target_p1: float, target_px: float, target_p2: float) -> Tuple[float, float]:
        def loss(p):
            lh, ma = p
            if lh <= 0.05 or ma <= 0.05: return 999.0
            m = DixonColesEngine.generate_matrix(lh, ma)
            p1, px, p2 = float(np.sum(np.tril(m, -1))), float(np.sum(np.diag(m))), float(np.sum(np.triu(m, 1)))
            return (p1 - target_p1)**2 + (px - target_px)**2 + (p2 - target_p2)**2
        res = minimize(loss, [1.35, 1.05], bounds=[(0.05, 5.0), (0.05, 5.0)], method='L-BFGS-B')
        return round(float(res.x[0]), 2), round(float(res.x[1]), 2)

@dataclass
class DirectionalForecast:
    favorite_team: str
    underdog_team: str
    expected_winner: str
    winner_confidence: float
    expected_goals_trend: str
    goals_confidence: float
    handicap_forecast: str
    most_probable_score: str
    score_probability: float

@dataclass
class TradeOrder:
    action: str
    is_approved: bool
    best_market: str
    target_selection: str
    captured_odds: float
    fair_odds: float
    ev_pct: float
    kelly_stake_pct: float
    max_risk_cap_pct: float
    warnings: List[str]
    narrative_reason: str

def analyze_match_complete(
    home_team: str, away_team: str,
    f_lh: float, f_mu: float,
    o1: float, ox: float, o2: float,
    ah_line: float, ah_home_o: float,
    ou_line: float, over_o: float,
    pin_ah_o: float, bankroll: float = 10000.0,
    min_ev_threshold: float = 0.025
) -> Tuple[np.ndarray, Dict, DirectionalForecast, TradeOrder]:

    m = DixonColesEngine.generate_matrix(f_lh, f_mu)
    m_p1 = float(np.sum(np.tril(m, -1)))
    m_px = float(np.sum(np.diag(m)))
    m_p2 = float(np.sum(np.triu(m, 1)))

    sp1, spx, sp2, z = ShinEngine.deoverround(o1, ox, o2)
    imp_lh, imp_mu = ReverseEngine.extract_xg(sp1, spx, sp2)

    # Probabilidades de Consenso (60% Mercado Sharp + 40% Modelo)
    c_p1 = (sp1 * 0.6) + (m_p1 * 0.4)
    c_px = (spx * 0.6) + (m_px * 0.4)
    c_p2 = (sp2 * 0.6) + (m_p2 * 0.4)

    # 1. PRONÓSTICO DIRECCIONAL
    fav_team, und_team = (home_team, away_team) if o1 < o2 else (away_team, home_team)

    if c_p1 > c_p2 and c_p1 > c_px:
        exp_winner = f"Victoria {home_team}"
        win_conf = c_p1
    elif c_p2 > c_p1 and c_p2 > c_px:
        exp_winner = f"Victoria {away_team}"
        win_conf = c_p2
    else:
        exp_winner = "Empate Técnico"
        win_conf = c_px

    p_over_consensus = sum(m[x, y] for x in range(m.shape[0]) for y in range(m.shape[1]) if (x + y) > ou_line)
    if p_over_consensus > 0.52:
        goals_trend = f"Over {ou_line} Goles"
        goals_conf = p_over_consensus
    else:
        goals_trend = f"Under {ou_line} Goles"
        goals_conf = 1.0 - p_over_consensus

    if c_p1 >= 0.50:
        handicap_forecast = f"{home_team} cubre {ah_line}"
    else:
        handicap_forecast = f"{away_team} cubre +{abs(ah_line)}"

    max_idx = np.unravel_index(np.argmax(m), m.shape)
    best_score = f"{max_idx[0]} - {max_idx[1]}"
    best_score_prob = m[max_idx[0], max_idx[1]]

    forecast = DirectionalForecast(
        favorite_team=fav_team,
        underdog_team=und_team,
        expected_winner=exp_winner,
        winner_confidence=round(win_conf * 100, 1),
        expected_goals_trend=goals_trend,
        goals_confidence=round(goals_conf * 100, 1),
        handicap_forecast=handicap_forecast,
        most_probable_score=best_score,
        score_probability=round(best_score_prob * 100, 1)
    )

    # 2. CÁLCULO FINANCIERO Y AUDITORÍA
    ev_1x2_home = (m_p1 * o1) - 1.0
    p_ah_home = m_p1 * 0.88 + m_px * 0.5
    ev_ah = (p_ah_home * ah_home_o) - 1.0
    ev_ou = (p_over_consensus * over_o) - 1.0

    ev_dict = {
        "1X2 Local": (ev_1x2_home, o1, m_p1, f"Local ({home_team}) @ {o1}"),
        f"AH {ah_line} Local": (ev_ah, ah_home_o, p_ah_home, f"AH {ah_line} {home_team} @ {ah_home_o}"),
        f"Over {ou_line} Goles": (ev_ou, over_o, p_over_consensus, f"Over {ou_line} @ {over_o}")
    }

    best_market_name = max(ev_dict, key=lambda k: ev_dict[k][0])
    best_ev, best_odds, best_prob, best_selection = ev_dict[best_market_name]

    warnings = []
    is_approved = True

    if best_ev < min_ev_threshold:
        is_approved = False
        warnings.append(f"Sin ventaja (+EV < {min_ev_threshold*100:.1f}%). Prevalece la eficiencia del mercado.")

    if ev_1x2_home > 0.035 and ev_ah < -0.01:
        is_approved = False
        warnings.append("🚨 Trampa de Liquidez: Discrepancia sospechosa entre 1X2 e Hándicap Asiático.")

    if abs(imp_lh - f_lh) > 0.55:
        is_approved = False
        warnings.append(f"🚨 Divergencia xG: La casa proyecta {imp_lh} goles locales vs tus {f_lh}.")

    kelly_pct = 0.0
    if is_approved and best_ev > 0:
        b = best_odds - 1.0
        q = 1.0 - best_prob
        f_k = (b * best_prob - q) / b
        kelly_pct = max(0.0, min(0.03, float(f_k * 0.25)))

    if is_approved:
        action_text = "EJECUTAR ORDEN (+EV VALIDADO)"
        narrative = f"Ventaja detectada en **{best_market_name}** con un **EV de {best_ev*100:+.2f}%**, alineado con el pronóstico direccional."
    else:
        action_text = "ABSTENERSE / MERCADO EFICIENTE"
        narrative = f"Pronóstico favorece a **{exp_winner}**, pero el precio ({best_odds}) ya descuenta la probabilidad real."

    order = TradeOrder(
        action=action_text,
        is_approved=is_approved,
        best_market=best_market_name,
        target_selection=best_selection,
        captured_odds=best_odds,
        fair_odds=round(1.0 / max(0.001, best_prob), 2),
        ev_pct=round(best_ev * 100, 2),
        kelly_stake_pct=round(kelly_pct * 100, 2),
        max_risk_cap_pct=round(kelly_pct * bankroll * 0.01, 2),
        warnings=warnings,
        narrative_reason=narrative
    )

    metrics = {
        "ev_1x2": round(ev_1x2_home * 100, 2),
        "ev_ah": round(ev_ah * 100, 2),
        "ev_ou": round(ev_ou * 100, 2),
        "imp_lh": imp_lh, "imp_mu": imp_mu,
        "shin_z": round(z, 4),
        "m_p1": m_p1, "m_px": m_px, "m_p2": m_p2,
        "sp1": sp1, "spx": spx, "sp2": sp2
    }

    return m, metrics, forecast, order

# ==============================================================================
# INTERFAZ Y SIDEBAR DINÁMICO CON ODDS-API
# ==============================================================================
st.sidebar.title("⚡ QuantOdds Terminal")
st.sidebar.markdown("---")

# 1. MÓDULO API KEY
st.sidebar.subheader("🔑 Conexión The-Odds-API")
api_key = st.sidebar.text_input("Ingresa tu Odds-API Key", type="password", help="Obtén una gratis en https://the-odds-api.com")

selected_league_label = st.sidebar.selectbox("Seleccionar Liga", list(SPORTS_DICT.keys()))
sport_key = SPORTS_DICT[selected_league_label]

api_matches = []
def_home, def_away = "Macará", "Santos"
def_o1, def_ox, def_o2 = 1.80, 3.75, 4.50
def_ah_line, def_ah_o = -0.75, 1.95
def_ou_line, def_over_o = 2.50, 1.85

if api_key:
    raw_data, status_msg = fetch_odds_api(api_key, sport_key)
    st.sidebar.caption(f"Status: {status_msg}")
    
    if raw_data:
        api_matches = raw_data
        match_options = [f"{m['home_team']} vs {m['away_team']}" for m in api_matches]
        if match_options:
            selected_match_str = st.sidebar.selectbox("🏟️ Partidos Disponibles", match_options)
            match_idx = match_options.index(selected_match_str)
            match_data = api_matches[match_idx]
            
            def_home = match_data['home_team']
            def_away = match_data['away_team']
            
            # Autocompletado de Cuotas desde la API (Tomando la primera casa o Pinnacle si existe)
            if match_data.get('bookmakers'):
                bm = match_data['bookmakers'][0] # Casa principal
                for mkt in bm.get('markets', []):
                    if mkt['key'] == 'h2h':
                        for o in mkt['outcomes']:
                            if o['name'] == def_home: def_o1 = float(o['price'])
                            elif o['name'] == def_away: def_o2 = float(o['price'])
                            elif o['name'] == 'Draw': def_ox = float(o['price'])
                    elif mkt['key'] == 'spreads':
                        for o in mkt['outcomes']:
                            if o['name'] == def_home:
                                def_ah_line = float(o.get('point', -0.5))
                                def_ah_o = float(o['price'])
                    elif mkt['key'] == 'totals':
                        for o in mkt['outcomes']:
                            if o['name'] == 'Over':
                                def_ou_line = float(o.get('point', 2.5))
                                def_over_o = float(o['price'])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración del Partido")

home_team = st.sidebar.text_input("Equipo Local", value=def_home)
away_team = st.sidebar.text_input("Equipo Visitante", value=def_away)
bankroll = st.sidebar.number_input("Capital Total ($)", value=10000.0, step=500.0)

st.sidebar.markdown("**Métricas xG (Modelo Propio)**")
f_lh = st.sidebar.number_input("xG Local (λ)", value=1.75, step=0.05)
f_mu = st.sidebar.number_input("xG Visitante (μ)", value=0.95, step=0.05)

st.sidebar.markdown("**Cuotas Mercado (1X2)**")
o1 = st.sidebar.number_input("Cuota Local (1)", value=def_o1, step=0.01)
ox = st.sidebar.number_input("Cuota Empate (X)", value=def_ox, step=0.01)
o2 = st.sidebar.number_input("Cuota Visitante (2)", value=def_o2, step=0.01)

st.sidebar.markdown("**Líneas de Derivados**")
ah_line = st.sidebar.number_input("Línea AH", value=def_ah_line, step=0.25)
ah_o = st.sidebar.number_input("Cuota AH Local", value=def_ah_o, step=0.01)
ou_line = st.sidebar.number_input("Línea Totales", value=def_ou_line, step=0.25)
over_o = st.sidebar.number_input("Cuota Over", value=def_over_o, step=0.01)
pin_ah = st.sidebar.number_input("Pinnacle AH Ref", value=1.91, step=0.01)

# ==============================================================================
# EJECUCIÓN DEL CÁLCULO Y RENDERIZADO
# ==============================================================================
matrix, metrics, forecast, order = analyze_match_complete(
    home_team, away_team, f_lh, f_mu, o1, ox, o2, ah_line, ah_o, ou_line, over_o, pin_ah, bankroll
)

# 1. HEADER
st.markdown(f"""
<div class="match-header">
    <div class="match-title">🏟️ {home_team} vs {away_team}</div>
    <div class="match-subtitle">INTEGRACIÓN API + ANÁLISIS DIRECCIONAL DE CONSENSO</div>
</div>
""", unsafe_allow_html=True)

# 2. PANEL DE PRONÓSTICO DIRECCIONAL
st.markdown(f"""
<div class="forecast-container">
    <div class="forecast-title">🔮 PRONÓSTICO DIRECCIONAL DE MERCADO (CONSENSO MODELO + CASA)</div>
    <div class="forecast-grid">
        <div class="forecast-card">
            <div class="forecast-label">Ganador Probable</div>
            <div class="forecast-val">{forecast.expected_winner}</div>
            <div class="forecast-sub">Certeza: {forecast.winner_confidence}%</div>
        </div>
        <div class="forecast-card">
            <div class="forecast-label">Marcador Exacto Dominante</div>
            <div class="forecast-val">{forecast.most_probable_score}</div>
            <div class="forecast-sub">Probabilidad: {forecast.score_probability}%</div>
        </div>
        <div class="forecast-card">
            <div class="forecast-label">Proyección de Goles</div>
            <div class="forecast-val">{forecast.expected_goals_trend}</div>
            <div class="forecast-sub">Probabilidad: {forecast.goals_confidence}%</div>
        </div>
        <div class="forecast-card">
            <div class="forecast-label">Línea de Hándicap</div>
            <div class="forecast-val">{forecast.handicap_forecast}</div>
            <div class="forecast-sub">Favorito: {forecast.favorite_team}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. METRICAS CLAVE
def get_cls(val):
    if val > 0: return "val-positive"
    if val < 0: return "val-negative"
    return "val-neutral"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">EV Hándicap</div>
        <div class="kpi-value {get_cls(metrics['ev_ah'])}">{metrics['ev_ah']:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">EV Mercado 1X2</div>
        <div class="kpi-value {get_cls(metrics['ev_1x2'])}">{metrics['ev_1x2']:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">EV Over/Under</div>
        <div class="kpi-value {get_cls(metrics['ev_ou'])}">{metrics['ev_ou']:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">xG Casa Impl.</div>
        <div class="kpi-value val-white">{metrics['imp_lh']} / {metrics['imp_mu']}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Shin (Insider z)</div>
        <div class="kpi-value val-neutral">{metrics['shin_z']:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. TICKET FINANCIERO DE EJECUCIÓN
ticket_cls = "ticket-trade" if order.is_approved else "ticket-no-trade"
badge_cls = "badge-trade" if order.is_approved else "badge-no-trade"
warnings_html = "".join([f'<div class="warning-chip">{w}</div>' for w in order.warnings])

st.markdown(f"""
<div class="ticket-card {ticket_cls}">
    <div>
        <span class="ticket-badge {badge_cls}">{order.action}</span>
    </div>
    <p style="margin: 8px 0 0 0; font-size: 0.85rem; color: #CBD5E1;">
        {order.narrative_reason}
    </p>
    <div class="exec-grid">
        <div>
            <div class="exec-item-label">Selección Recomendada</div>
            <div class="exec-item-val">{order.target_selection if order.is_approved else 'Ninguna'}</div>
        </div>
        <div>
            <div class="exec-item-label">Cuota Justa / Mercado</div>
            <div class="exec-item-val">{order.fair_odds} / {order.captured_odds if order.is_approved else 'N/A'}</div>
        </div>
        <div>
            <div class="exec-item-label">Esperanza (+EV)</div>
            <div class="exec-item-val {get_cls(order.ev_pct)}">{order.ev_pct:+.2f}%</div>
        </div>
        <div>
            <div class="exec-item-label">Stake Sugerido (Kelly)</div>
            <div class="exec-item-val">{order.kelly_stake_pct}% (${order.max_risk_cap_pct})</div>
        </div>
    </div>
    {warnings_html}
</div>
""", unsafe_allow_html=True)

# 5. GRÁFICOS INTERACTIVOS
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("##### 📊 Comparativa de Probabilidades (Modelo vs Shin)")
    if HAS_PLOTLY:
        categories = [f"{home_team} (1)", "Empate (X)", f"{away_team} (2)"]
        fig_bar = go.Figure(data=[
            go.Bar(name='Tu Modelo', x=categories, y=[metrics['m_p1']*100, metrics['m_px']*100, metrics['m_p2']*100], marker_color='#10B981', text=[f"{metrics['m_p1']*100:.1f}%", f"{metrics['m_px']*100:.1f}%", f"{metrics['m_p2']*100:.1f}%"], textposition='auto'),
            go.Bar(name='Mercado Shin', x=categories, y=[metrics['sp1']*100, metrics['spx']*100, metrics['sp2']*100], marker_color='#6366F1', text=[f"{metrics['sp1']*100:.1f}%", f"{metrics['spx']*100:.1f}%", f"{metrics['sp2']*100:.1f}%"], textposition='auto')
        ])
        fig_bar.update_layout(
            barmode='group', height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#9CA3AF")),
            font=dict(color="#9CA3AF", size=10),
            yaxis=dict(gridcolor='#1E2937', title="", showticklabels=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

with col_g2:
    st.markdown("##### 🔥 Matriz de Marcadores Probables")
    if HAS_PLOTLY:
        df_m = np.round(matrix[:5, :5] * 100, 1)
        fig_hm = px.imshow(
            df_m,
            x=[f"V{i}" for i in range(5)],
            y=[f"L{i}" for i in range(5)],
            color_continuous_scale="Viridis",
            text_auto=True
        )
        fig_hm.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False, font=dict(color="#9CA3AF", size=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
