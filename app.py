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
# CONFIGURACIÓN DE PÁGINA Y CSS PREMIUM (DARK FINTECH THEME)
# ==============================================================================
st.set_page_config(
    page_title="QuantOdds Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados para corregir Streamlit en móviles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #090D16;
        color: #E2E8F0;
    }
    
    /* Header del Partido */
    .match-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    .match-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .match-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Grid de Métricas / KPIs */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    }
    @media (min-width: 768px) {
        .kpi-grid { grid-template-columns: repeat(5, 1fr); }
    }
    
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.2rem;
        font-weight: 700;
    }
    .val-positive { color: #10B981; }
    .val-negative { color: #EF4444; }
    .val-neutral { color: #F59E0B; }
    .val-white { color: #F9FAFB; }

    /* Tarjetas de Estado y Dictamen */
    .status-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .status-denied {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-approved {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .badge-denied { background: #EF4444; color: #FFFFFF; }
    .badge-approved { background: #10B981; color: #FFFFFF; }
    
    .warning-box {
        background: #1F1924;
        border-left: 3px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #FCD34D;
        margin-top: 8px;
    }
    
    /* Contenedores y Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        background-color: #111827;
        color: #9CA3AF;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOGICA DE MOTORES CUANTITATIVOS
# ==============================================================================
class OddsAPIService:
    LEAGUES = {
        "Copa Libertadores": "soccer_conmebol_copa_libertadores",
        "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
        "Brasileirão Série A": "soccer_brazil_campeonato",
        "Liga Profesional (ARG)": "soccer_argentina_primera_division",
        "UEFA Champions League": "soccer_uefa_champs_league",
        "Premier League": "soccer_epl",
        "La Liga": "soccer_spain_la_liga",
    }

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
        return st.sidebar.text_input("🔑 API Key:", type="password")

    @classmethod
    def fetch_upcoming_matches(cls, sport_key: str, api_key: str) -> List[Dict]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {"apiKey": api_key, "regions": "eu,us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal"}
        try:
            res = requests.get(url, params=params, timeout=10)
            return res.json() if res.status_code == 200 else []
        except Exception:
            return []

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
class AuditData:
    is_valid: bool
    status_title: str
    ev_1x2: float
    ev_ah: float
    ev_ou: float
    shin_z: float
    imp_lh: float
    imp_mu: float
    m_p1: float
    m_px: float
    m_p2: float
    s_p1: float
    s_px: float
    s_p2: float
    warnings: List[str]

def audit_match(f_lh, f_mu, o1, ox, o2, ah_line, ah_o, ou_line, over_o, pin_ah) -> Tuple[np.ndarray, AuditData]:
    m = DixonColesEngine.generate_matrix(f_lh, f_mu)
    p1, px, p2 = float(np.sum(np.tril(m, -1))), float(np.sum(np.diag(m))), float(np.sum(np.triu(m, 1)))
    
    # Totals y AH simplificados
    m_over = sum(m[x, y] for x in range(m.shape[0]) for y in range(m.shape[1]) if (x + y) > ou_line)
    m_ah = p1 * 0.85 # Aproximación estocástica de cobertura
    
    sp1, spx, sp2, z = ShinEngine.deoverround(o1, ox, o2)
    imp_lh, imp_mu = ReverseEngine.extract_xg(sp1, spx, sp2)
    
    ev1 = (p1 * o1) - 1.0
    ev_ah = (m_ah * ah_o) - 1.0
    ev_ou = (m_over * over_o) - 1.0
    
    warnings = []
    is_valid = True
    
    if ev1 > 0.035 and ev_ah < -0.015:
        warnings.append(f"Trampa de Liquidez: 1X2 inflado (+{ev1*100:.1f}%), pero Mercado Ancla AH ({ah_line}) en negativo ({ev_ah*100:.1f}%).")
        is_valid = False

    if abs(imp_lh - f_lh) > 0.5:
        warnings.append(f"Desviación xG: La casa proyecta Local en {imp_lh}, tu modelo usa {f_lh}. Revisa alineaciones.")
        is_valid = False

    status = "Orden Aprobada (+EV)" if is_valid else "Riesgo Estructural / Trampa Detectada"
    
    return m, AuditData(
        is_valid=is_valid, status_title=status,
        ev_1x2=round(ev1*100, 2), ev_ah=round(ev_ah*100, 2), ev_ou=round(ev_ou*100, 2),
        shin_z=round(z, 4), imp_lh=imp_lh, imp_mu=imp_mu,
        m_p1=p1, m_px=px, m_p2=p2, s_p1=sp1, s_px=spx, s_p2=sp2,
        warnings=warnings
    )

# ==============================================================================
# CONTROLES Y SIDEBAR
# ==============================================================================
st.sidebar.title("⚙️ Parámetros")
api_key = OddsAPIService.get_api_key()

auto_o1, auto_ox, auto_o2 = 1.80, 3.75, 4.50
match_name = "Olimpia Asunción vs Vasco da Gama"

if api_key:
    league_label = st.sidebar.selectbox("Competición", list(OddsAPIService.LEAGUES.keys()))
    if st.sidebar.button("🔍 Cargar Partidos"):
        matches = OddsAPIService.fetch_upcoming_matches(OddsAPIService.LEAGUES[league_label], api_key)
        st.session_state["matches"] = matches

    if "matches" in st.session_state and st.session_state["matches"]:
        m_list = st.session_state["matches"]
        idx = st.sidebar.selectbox("Partido", range(len(m_list)), format_func=lambda i: f"{m_list[i]['home_team']} vs {m_list[i]['away_team']}")
        match_data = m_list[idx]
        match_name = f"{match_data['home_team']} vs {match_data['away_team']}"

f_lh = st.sidebar.number_input("xG Local (λ)", value=1.75, step=0.05)
f_mu = st.sidebar.number_input("xG Visitante (μ)", value=0.95, step=0.05)

o1 = st.sidebar.number_input("Cuota Local (1)", value=auto_o1, step=0.01)
ox = st.sidebar.number_input("Cuota Empate (X)", value=auto_ox, step=0.01)
o2 = st.sidebar.number_input("Cuota Visitante (2)", value=auto_o2, step=0.01)

ah_line = st.sidebar.number_input("Línea AH", value=-0.75, step=0.25)
ah_o = st.sidebar.number_input("Cuota AH Local", value=1.95, step=0.01)
ou_line = st.sidebar.number_input("Línea Over/Under", value=2.50, step=0.25)
over_o = st.sidebar.number_input("Cuota Over", value=1.85, step=0.01)
pin_ah = st.sidebar.number_input("Pinnacle AH Ref", value=1.91, step=0.01)

# CÁLCULO
matrix, audit = audit_match(f_lh, f_mu, o1, ox, o2, ah_line, ah_o, ou_line, over_o, pin_ah)

# ==============================================================================
# INTERFAZ PRINCIPAL (RESPONSIVE DASHBOARD)
# ==============================================================================

# 1. HEADER
st.markdown(f"""
<div class="match-header">
    <div class="match-title">🏟️ {match_name}</div>
    <div class="match-subtitle">Terminal de Inteligencia Cuantitativa</div>
</div>
""", unsafe_allow_html=True)

# 2. GRID DE KPIs DENSOS (Estilo FinTech)
def get_cls(val):
    if val > 0: return "val-positive"
    if val < 0: return "val-negative"
    return "val-neutral"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">EV Hándicap</div>
        <div class="kpi-value {get_cls(audit.ev_ah)}">{audit.ev_ah:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">EV Mercado 1X2</div>
        <div class="kpi-value {get_cls(audit.ev_1x2)}">{audit.ev_1x2:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">EV Over/Under</div>
        <div class="kpi-value {get_cls(audit.ev_ou)}">{audit.ev_ou:+.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">xG Casa Impl.</div>
        <div class="kpi-value val-white">{audit.imp_lh} / {audit.imp_mu}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Shin (Insider z)</div>
        <div class="kpi-value val-neutral">{audit.shin_z:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. DICTAMEN DE RIESGO
badge_cls = "badge-approved" if audit.is_valid else "badge-denied"
card_cls = "status-approved" if audit.is_valid else "status-denied"
badge_text = "EJECUCIÓN PERMITIDA" if audit.is_valid else "ORDEN DENEGADA"

warnings_html = "".join([f'<div class="warning-box">⚠️ {w}</div>' for w in audit.warnings])

st.markdown(f"""
<div class="status-card {card_cls}">
    <span class="status-badge {badge_cls}">{badge_text}</span>
    <h3 style="margin: 4px 0; font-size: 1.1rem; color: #F8FAFC;">{audit.status_title}</h3>
    <p style="margin: 0; font-size: 0.82rem; color: #94A3B8;">
        Auditoría realizada cruzando el modelo Dixon-Coles frente a la de-marginalización de Shin.
    </p>
    {warnings_html}
</div>
""", unsafe_allow_html=True)

# 4. GRÁFICOS OPTIMIZADOS PARA MÓVIL
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("##### 📊 Probabilidades: Modelo vs Casa")
    if HAS_PLOTLY:
        categories = ["Local (1)", "Empate (X)", "Visitante (2)"]
        fig_bar = go.Figure(data=[
            go.Bar(name='Tu Modelo', x=categories, y=[audit.m_p1*100, audit.m_px*100, audit.m_p2*100], marker_color='#10B981'),
            go.Bar(name='Mercado Shin', x=categories, y=[audit.s_p1*100, audit.s_px*100, audit.s_p2*100], marker_color='#6366F1')
        ])
        fig_bar.update_layout(
            barmode='group',
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#9CA3AF")),
            font=dict(color="#9CA3AF", size=10),
            yaxis=dict(gridcolor='#1F2937', title="")
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

with col_g2:
    st.markdown("##### 🔥 Matriz de Marcadores Exactos")
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
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            font=dict(color="#9CA3AF", size=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})

# 5. DETALLES EN PESTAÑAS
tab1, tab2 = st.tabs(["📋 Desglose Técnico", "ℹ️ Ayuda"])

with tab1:
    st.markdown(f"""
    * **Diferencial xG Local ($\Delta \lambda$):** `{f_lh - audit.imp_lh:+.2f}` goles respecto a la casa.
    * **Diferencial xG Visitante ($\Delta \mu$):** `{f_mu - audit.imp_mu:+.2f}` goles respecto a la casa.
    * **Métrica de Sesgo ($z$):** `{audit.shin_z:.4f}` ({'Baja actividad de insiders' if audit.shin_z < 0.02 else 'Flujo fuerte de dinero informado'}).
    """)

with tab2:
    st.caption("Esta interfaz utiliza CSS Grid dinámico y Plotly optimizado para que los gráficos mantengan proporciones limpias tanto en dispositivos móviles como en monitores.")
