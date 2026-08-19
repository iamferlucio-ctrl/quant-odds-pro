import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import exp, factorial
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS DE INTERFAZ INSTITUCIONAL
# ==============================================================================
st.set_page_config(
    page_title="QuantOdds Pro 360 — Multi-Market Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SERVICIO DE CONEXIÓN A THE-ODDS-API (INCLUYE SUDAMÉRICA Y EUROPA)
# ==============================================================================
class OddsAPIService:
    """Gestiona la obtención automática de partidos y cuotas en tiempo real."""
    
    LEAGUES = {
        # --- SUDAMÉRICA & CONMEBOL ---
        "Copa Libertadores": "soccer_conmebol_copa_libertadores",
        "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
        "Liga Profesional (Argentina)": "soccer_argentina_primera_division",
        "Brasileirão Série A (Brasil)": "soccer_brazil_campeonato",
        "Liga BetPlay (Colombia)": "soccer_colombia_primer_a",
        "Primera División (Chile)": "soccer_chile_campeonato",
        "Liga 1 (Perú)": "soccer_peru_liga_1",
        
        # --- EUROPA & INTERNACIONAL ---
        "UEFA Champions League": "soccer_uefa_champs_league",
        "Premier League (Inglaterra)": "soccer_epl",
        "La Liga (España)": "soccer_spain_la_liga",
        "Serie A (Italia)": "soccer_italy_serie_a",
        "Bundesliga (Alemania)": "soccer_germany_bundesliga",
    }

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
        return st.sidebar.text_input("🔑 API Key (The-Odds-API):", type="password")

    @classmethod
    def fetch_upcoming_matches(cls, sport_key: str, api_key: str) -> List[Dict]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            "apiKey": api_key,
            "regions": "eu,us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal"
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                return res.json()
            else:
                st.sidebar.error(f"Error API ({res.status_code}): Verifica tu API Key.")
                return []
        except Exception as e:
            st.sidebar.error(f"Error de conexión: {e}")
            return []

# ==============================================================================
# 1. MOTOR ESTOCÁSTICO BIVARIADO (DIXON-COLES)
# ==============================================================================
class DixonColesEngine:
    @staticmethod
    def tau_adjustment(x: int, y: int, lambda_h: float, mu_a: float, rho: float) -> float:
        if x == 0 and y == 0: return 1.0 - (lambda_h * mu_a * rho)
        elif x == 0 and y == 1: return 1.0 + (lambda_h * rho)
        elif x == 1 and y == 0: return 1.0 + (mu_a * rho)
        elif x == 1 and y == 1: return 1.0 - rho
        return 1.0

    @classmethod
    def generate_score_matrix(cls, lambda_h: float, mu_a: float, rho: float = -0.05, max_goals: int = 10) -> np.ndarray:
        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                p_x = (exp(-lambda_h) * (lambda_h ** x)) / factorial(x)
                p_y = (exp(-mu_a) * (mu_a ** y)) / factorial(y)
                adj = cls.tau_adjustment(x, y, lambda_h, mu_a, rho)
                matrix[x, y] = max(0.0, p_x * p_y * adj)
        sum_m = np.sum(matrix)
        return matrix / sum_m if sum_m > 0 else matrix

# ==============================================================================
# 2. DESMARCADO AVANZADO DE SHIN (PARÁMETRO z)
# ==============================================================================
class ShinEngine:
    @classmethod
    def deoverround_1x2(cls, odds_1: float, odds_x: float, odds_2: float) -> Tuple[float, float, float, float]:
        if odds_1 <= 1.0 or odds_x <= 1.0 or odds_2 <= 1.0:
            return 0.333, 0.333, 0.333, 0.0
        raw_probs = np.array([1.0 / odds_1, 1.0 / odds_x, 1.0 / odds_2])
        beta = np.sum(raw_probs)
        z = max(0.0001, (beta - 1.0) / 0.25)
        for _ in range(20):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (raw_probs / beta)) - z) / (2 * (1 - z))
            p = p / np.sum(p)
            new_beta = np.sum(raw_probs / (z + (1 - z) * p))
            if abs(new_beta - beta) < 1e-6: break
            z = max(0.0001, min(0.4, z + (beta - new_beta) * 0.1))
        return float(p[0]), float(p[1]), float(p[2]), float(z)

# ==============================================================================
# 3. MOTOR DE INGENIERÍA INVERSA (OPTIMIZACIÓN SCIPY L-BFGS-B)
# ==============================================================================
class ReverseEngineeringEngine:
    @classmethod
    def extract_implied_xg(cls, target_p1: float, target_px: float, target_p2: float) -> Tuple[float, float, float]:
        def loss_function(params):
            lh, ma = params
            if lh <= 0.05 or ma <= 0.05: return 999.0
            matrix = DixonColesEngine.generate_score_matrix(lh, ma, rho=-0.05, max_goals=8)
            p1 = float(np.sum(np.tril(matrix, -1)))
            px = float(np.sum(np.diag(matrix)))
            p2 = float(np.sum(np.triu(matrix, 1)))
            return (p1 - target_p1)**2 + (px - target_px)**2 + (p2 - target_p2)**2

        res = minimize(loss_function, [1.35, 1.05], bounds=[(0.05, 5.0), (0.05, 5.0)], method='L-BFGS-B')
        return round(float(res.x[0]), 2), round(float(res.x[1]), 2), float(res.fun)

# ==============================================================================
# 4. DECONVOLUCIÓN MULTIMERCADO
# ==============================================================================
class MultiMarketDeconvolution:
    def __init__(self, score_matrix: np.ndarray):
        self.M = score_matrix
        self.max_g = score_matrix.shape[0]

    def get_1x2(self) -> Tuple[float, float, float]:
        return float(np.sum(np.tril(self.M, -1))), float(np.sum(np.diag(self.M))), float(np.sum(np.triu(self.M, 1)))

    def get_totals(self, line: float = 2.5) -> Tuple[float, float]:
        over_prob = sum(self.M[x, y] for x in range(self.max_g) for y in range(self.max_g) if (x + y) > line)
        return over_prob, 1.0 - over_prob

    def get_btts(self) -> Tuple[float, float]:
        btts_yes = float(np.sum(self.M[1:, 1:]))
        return btts_yes, 1.0 - btts_yes

    def get_asian_handicap(self, line: float) -> float:
        prob_win = 0.0
        for x in range(self.max_g):
            for y in range(self.max_g):
                diff = x - y
                if line == 0.0:
                    if diff > 0: prob_win += self.M[x, y]
                    elif diff == 0: prob_win += self.M[x, y] * 0.5
                elif line == -0.25:
                    if diff > 0: prob_win += self.M[x, y]
                    elif diff == 0: prob_win += self.M[x, y] * 0.5
                elif line == -0.5:
                    if diff > 0: prob_win += self.M[x, y]
                elif line == -0.75:
                    if diff > 1: prob_win += self.M[x, y]
                    elif diff == 1: prob_win += self.M[x, y] * 0.75
                elif line == -1.0:
                    if diff > 1: prob_win += self.M[x, y]
                    elif diff == 1: prob_win += self.M[x, y] * 0.5
                elif line < 0:
                    if (diff + line) > 0: prob_win += self.M[x, y]
                elif line > 0:
                    if (diff + line) > 0: prob_win += self.M[x, y]
                    elif (diff + line) == 0: prob_win += self.M[x, y] * 0.5
        return min(1.0, max(0.0, prob_win))

# ==============================================================================
# 5. AUDITOR CUANTITATIVO Y FILTRO DE TRAMPAS
# ==============================================================================
@dataclass
class AuditResult:
    is_valid: bool
    status_title: str
    ev_1x2_pct: float
    ev_ah_pct: float
    ev_totals_pct: float
    shin_z: float
    implied_lambda: float
    implied_mu: float
    warnings: List[str]
    kelly_stake_pct: float

class QuantAuditor:
    @staticmethod
    def audit(
        fundamental_lambda: float, fundamental_mu: float,
        odds_1: float, odds_x: float, odds_2: float,
        ah_line: float, ah_home_odds: float,
        ou_line: float, over_odds: float,
        pinnacle_ah_odds: float, kelly_fraction: float = 0.20
    ) -> Tuple[np.ndarray, AuditResult]:
        
        matrix = DixonColesEngine.generate_score_matrix(fundamental_lambda, fundamental_mu)
        deconv = MultiMarketDeconvolution(matrix)
        
        m_p1, _, _ = deconv.get_1x2()
        m_over, _ = deconv.get_totals(ou_line)
        m_ah_home = deconv.get_asian_handicap(ah_line)
        
        s_p1, s_px, s_p2, shin_z = ShinEngine.deoverround_1x2(odds_1, odds_x, odds_2)
        imp_lambda, imp_mu, _ = ReverseEngineeringEngine.extract_implied_xg(s_p1, s_px, s_p2)
        
        ev_1 = (m_p1 * odds_1) - 1.0
        ev_ah = (m_ah_home * ah_home_odds) - 1.0
        ev_totals = (m_over * over_odds) - 1.0
        
        warnings = []
        is_valid = True
        
        if ev_1 > 0.035 and ev_ah < -0.015:
            warnings.append(
                f"🚨 TRAMPA DE LIQUIDEZ (+EV Falso en 1X2): El 1X2 marca +{ev_1*100:.1f}%, "
                f"pero el mercado ancla AH ({ah_line}) tiene EV negativo ({ev_ah*100:.1f}%). Rechazado."
            )
            is_valid = False

        if abs(imp_lambda - fundamental_lambda) > 0.6 and ev_1 > 0.05:
            warnings.append(
                f"🚨 DESVIACIÓN EXTREMA DE xG: El mercado asume xG Local de {imp_lambda}, "
                f"mientras que tu modelo usa {fundamental_lambda}. Posible baja/noticia relevante."
            )
            is_valid = False

        if pinnacle_ah_odds > 1.0 and ah_home_odds > (pinnacle_ah_odds * 1.035):
            warnings.append(
                f"🚨 ALERTA DE FLUJO SHARP: La cuota ofrecida ({ah_home_odds}) está inflada "
                f"un {((ah_home_odds/pinnacle_ah_odds)-1)*100:.1f}% respecto a Pinnacle ({pinnacle_ah_odds})."
            )
            is_valid = False

        kelly_stake = 0.0
        if is_valid and ev_ah > 0.02:
            b = ah_home_odds - 1.0
            p = m_ah_home
            q = 1.0 - p
            f_kelly = (b * p - q) / b
            if f_kelly > 0:
                kelly_stake = round(float(f_kelly * kelly_fraction * 100), 2)
                status_title = "✅ OPERACIÓN APROBADA (+EV MULTIMERCADO)"
            else:
                status_title = "🟡 COHERENTE SIN VENTAJA SUFICIENTE"
        elif not is_valid:
            status_title = "🔴 ORDEN RECHAZADA (Divergencia / Trampa Detectada)"
        else:
            status_title = "🟢 PRECIO JUSTO (Sin EV significativo)"

        return matrix, AuditResult(
            is_valid=is_valid,
            status_title=status_title,
            ev_1x2_pct=round(ev_1 * 100, 2),
            ev_ah_pct=round(ev_ah * 100, 2),
            ev_totals_pct=round(ev_totals * 100, 2),
            shin_z=round(shin_z, 4),
            implied_lambda=imp_lambda,
            implied_mu=imp_mu,
            warnings=warnings,
            kelly_stake_pct=kelly_stake
        )

# ==============================================================================
# 6. DASHBOARD INTERACTIVO STREAMLIT
# ==============================================================================
st.title("⚡ QuantOdds Pro 360")
st.caption("Auto-Feeder en Tiempo Real, Ingeniería Inversa & Motor Estocástico Multimercado")
st.divider()

st.sidebar.header("📡 1. Auto-Carga de Partido")
api_key = OddsAPIService.get_api_key()

auto_odds_1, auto_odds_x, auto_odds_2 = 1.80, 3.75, 4.50
auto_over_odds = 1.85

if api_key:
    selected_league_label = st.sidebar.selectbox("Selecciona Liga / Competición:", list(OddsAPIService.LEAGUES.keys()))
    league_key = OddsAPIService.LEAGUES[selected_league_label]
    
    if st.sidebar.button("🔍 Buscar Partidos En Vivo"):
        with st.spinner("Conectando con casas de apuestas..."):
            matches = OddsAPIService.fetch_upcoming_matches(league_key, api_key)
            st.session_state["fetched_matches"] = matches

    if "fetched_matches" in st.session_state and st.session_state["fetched_matches"]:
        matches = st.session_state["fetched_matches"]
        match_options = [f"{m['home_team']} vs {m['away_team']}" for m in matches]
        selected_match_idx = st.sidebar.selectbox("Selecciona Partido:", range(len(match_options)), format_func=lambda i: match_options[i])
        
        match_data = matches[selected_match_idx]
        for bookmaker in match_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == match_data["home_team"]: auto_odds_1 = outcome["price"]
                        elif outcome["name"] == match_data["away_team"]: auto_odds_2 = outcome["price"]
                        elif outcome["name"] == "Draw": auto_odds_x = outcome["price"]
        st.sidebar.success("✅ Cuotas autocargadas correctamente.")

st.sidebar.divider()
st.sidebar.subheader("2. Parametrización xG y Cuotas")

f_lambda = st.sidebar.number_input("xG Local (λ)", value=1.75, step=0.05)
f_mu = st.sidebar.number_input("xG Visitante (μ)", value=0.95, step=0.05)

odds_1 = st.sidebar.number_input("Cuota Local (1)", value=float(auto_odds_1), step=0.01)
odds_x = st.sidebar.number_input("Cuota Empate (X)", value=float(auto_odds_x), step=0.01)
odds_2 = st.sidebar.number_input("Cuota Visitante (2)", value=float(auto_odds_2), step=0.01)

st.sidebar.subheader("3. Mercado Ancla (AH, Totales & Sharp)")
ah_line = st.sidebar.number_input("Línea Hándicap Asiático", value=-0.75, step=0.25)
ah_home_odds = st.sidebar.number_input("Cuota AH Local", value=1.95, step=0.01)
ou_line = st.sidebar.number_input("Línea Totales (O/U)", value=2.50, step=0.25)
over_odds = st.sidebar.number_input("Cuota Over", value=float(auto_over_odds), step=0.01)
pin_ah = st.sidebar.number_input("Pinnacle AH Local (Ref)", value=1.91, step=0.01)

# EJECUCIÓN DE LA AUDITORÍA
matrix, audit = QuantAuditor.audit(
    f_lambda, f_mu,
    odds_1, odds_x, odds_2,
    ah_line, ah_home_odds,
    ou_line, over_odds,
    pin_ah
)

# VISTA PRINCIPAL DE RESULTADOS
st.subheader(f"Dictamen: {audit.status_title}")

if audit.warnings:
    for w in audit.warnings:
        st.error(w)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("EV Hándicap Asiático", f"{audit.ev_ah_pct:+.2f}%")
col_m2.metric("EV Mercado 1X2", f"{audit.ev_1x2_pct:+.2f}%")
col_m3.metric("EV Totales (O/U)", f"{audit.ev_totals_pct:+.2f}%")
col_m4.metric("xG Implícito Casa (λ / μ)", f"{audit.implied_lambda} / {audit.implied_mu}")
col_m5.metric("Shin Insiders (z)", f"{audit.shin_z:.4f}")

if audit.is_valid and audit.kelly_stake_pct > 0:
    st.success(
        f"🎯 **POSICIÓN CONFIRMADA:** Apuesta a **Hándicap Asiático Local ({ah_line}) @ {ah_home_odds}**\n\n"
        f"💰 **Gestión de Capital (Kelly 1/5):** Asignar el **{audit.kelly_stake_pct}%** del Bankroll."
    )

st.divider()

# PESTAÑAS DE ANÁLISIS DETALLADO
tab_matrix, tab_reverse, tab_docs = st.tabs([
    "🔥 Matriz Bivariada de Marcadores (10x10)", 
    "🔄 Análisis de Ingeniería Inversa",
    "📚 Documentación Técnica"
])

with tab_matrix:
    st.markdown("### Probabilidades Conjuntas de Marcadores Exactos (%)")
    df_matrix = pd.DataFrame(
        np.round(matrix[:7, :7] * 100, 2),
        index=[f"Local {i}" for i in range(7)],
        columns=[f"Vis {j}" for j in range(7)]
    )
    st.dataframe(df_matrix.style.highlight_max(axis=None, color='#1F6B3A'), use_container_width=True)

with tab_reverse:
    st.markdown("### Deconvolución de Cuotas del Bookmaker")
    c_r1, c_r2 = st.columns(2)
    with c_r1:
        st.info(f"**xG Fundamental Tuya:**\n- Local (λ): `{f_lambda}`\n- Visitante (μ): `{f_mu}`")
    with c_r2:
        st.warning(f"**xG Implícito Desnudado de la Casa:**\n- Local (λ_imp): `{audit.implied_lambda}`\n- Visitante (μ_imp): `{audit.implied_mu}`")
    
    diff_lh = round(f_lambda - audit.implied_lambda, 2)
    diff_mu = round(f_mu - audit.implied_mu, 2)
    st.markdown(f"**Diferencial Estructural (Modelo - Mercado):** $\Delta \lambda = {diff_lh:+ \text{g}}$, $\Delta \mu = {diff_mu:+ \text{g}}$")

with tab_docs:
    st.markdown("""
    #### Fundamentos Cuantitativos del Sistema
    1. **Dixon-Coles:** Ajusta la distribución Poisson con el parámetro $\\tau$ para corregir marcadores de baja anotación ($0$-$0$, $1$-$0$, $0$-$1$, $1$-$1$).
    2. **Desmarcado de Shin:** Modela la presencia de apostadores informados ($z$) para calcular probabilidades puras sin el sesgo proporcional del overround.
    3. **Ingeniería Inversa SciPy:** Despeja los goles esperados $(\\lambda, \\mu)$ implícitos mediante el algoritmo de optimización no lineal `L-BFGS-B`.
    4. **Filtro Anti-Trampas:** Bloquea posturas cuando el $+EV$ del $1\\text{X}2$ no coincide con el mercado profundo de Hándicap Asiático o cuando hay divergencias de flujo Sharp.
    """)
