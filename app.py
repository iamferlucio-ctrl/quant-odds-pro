import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import exp, factorial
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Importación segura de Plotly
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="QuantOdds Pro 360 — Terminal Cuantitativa",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .report-box-approved {
        background-color: #161B22;
        border-left: 5px solid #238636;
        padding: 18px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .report-box-rejected {
        background-color: #161B22;
        border-left: 5px solid #DA3633;
        padding: 18px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONEXIÓN A THE-ODDS-API
# ==============================================================================
class OddsAPIService:
    LEAGUES = {
        "Copa Libertadores": "soccer_conmebol_copa_libertadores",
        "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
        "Liga Profesional (Argentina)": "soccer_argentina_primera_division",
        "Brasileirão Série A (Brasil)": "soccer_brazil_campeonato",
        "Liga BetPlay (Colombia)": "soccer_colombia_primer_a",
        "Primera División (Chile)": "soccer_chile_campeonato",
        "Liga 1 (Perú)": "soccer_peru_liga_1",
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
# MOTORES CUANTITATIVOS (DIXON-COLES, SHIN, INGENIERÍA INVERSA)
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
    def generate_score_matrix(cls, lambda_h: float, mu_a: float, rho: float = -0.05, max_goals: int = 8) -> np.ndarray:
        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                p_x = (exp(-lambda_h) * (lambda_h ** x)) / factorial(x)
                p_y = (exp(-mu_a) * (mu_a ** y)) / factorial(y)
                adj = cls.tau_adjustment(x, y, lambda_h, mu_a, rho)
                matrix[x, y] = max(0.0, p_x * p_y * adj)
        sum_m = np.sum(matrix)
        return matrix / sum_m if sum_m > 0 else matrix

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

class MultiMarketDeconvolution:
    def __init__(self, score_matrix: np.ndarray):
        self.M = score_matrix
        self.max_g = score_matrix.shape[0]

    def get_1x2(self) -> Tuple[float, float, float]:
        return float(np.sum(np.tril(self.M, -1))), float(np.sum(np.diag(self.M))), float(np.sum(np.triu(self.M, 1)))

    def get_totals(self, line: float = 2.5) -> Tuple[float, float]:
        over_prob = sum(self.M[x, y] for x in range(self.max_g) for y in range(self.max_g) if (x + y) > line)
        return over_prob, 1.0 - over_prob

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
# ESTRUCTURA Y AUDITORÍA
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
    model_p1: float
    model_px: float
    model_p2: float
    shin_p1: float
    shin_px: float
    shin_p2: float
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
        
        m_p1, m_px, m_p2 = deconv.get_1x2()
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
                f"🚨 TRAMPA DE LIQUIDEZ: El mercado 1X2 señala +{ev_1*100:.1f}%, "
                f"pero el mercado ancla AH ({ah_line}) está en negativo ({ev_ah*100:.1f}%). Postura denegada."
            )
            is_valid = False

        if abs(imp_lambda - fundamental_lambda) > 0.55 and ev_1 > 0.04:
            warnings.append(
                f"🚨 DESVIACIÓN ESTRUCTURAL DE xG: La casa asume xG Local de {imp_lambda}, "
                f"mientras que tu modelo usa {fundamental_lambda}. Posible baja relevante o noticia de último minuto."
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
                status_title = "✅ EJECUCIÓN APROBADA (+EV MULTIMERCADO)"
            else:
                status_title = "🟡 MERCADO COHERENTE SIN VENTAJA SUFICIENTE"
        elif not is_valid:
            status_title = "🔴 ORDEN DENEGADA (Riesgo Estructural / Trampa Detectada)"
        else:
            status_title = "🟢 MERCADO EN PRECIO JUSTO (Sin EV Apalancable)"

        return matrix, AuditResult(
            is_valid=is_valid,
            status_title=status_title,
            ev_1x2_pct=round(ev_1 * 100, 2),
            ev_ah_pct=round(ev_ah * 100, 2),
            ev_totals_pct=round(ev_totals * 100, 2),
            shin_z=round(shin_z, 4),
            implied_lambda=imp_lambda,
            implied_mu=imp_mu,
            model_p1=m_p1, model_px=m_px, model_p2=m_p2,
            shin_p1=s_p1, shin_px=s_px, shin_p2=s_p2,
            warnings=warnings,
            kelly_stake_pct=kelly_stake
        )

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
st.title("⚡ QuantOdds Pro 360")
st.caption("Terminal de Inteligencia Cuantitativa, Desmarcado de Shin & Auditoría Estocástica")
st.divider()

st.sidebar.header("📡 1. Auto-Feeder de Cuotas")
api_key = OddsAPIService.get_api_key()

auto_odds_1, auto_odds_x, auto_odds_2 = 1.80, 3.75, 4.50
auto_over_odds = 1.85
match_title = "Análisis Personalizado"

if api_key:
    selected_league_label = st.sidebar.selectbox("Selecciona Competición:", list(OddsAPIService.LEAGUES.keys()))
    league_key = OddsAPIService.LEAGUES[selected_league_label]
    
    if st.sidebar.button("🔍 Cargar Partidos"):
        with st.spinner("Conectando con servidores de cuotas..."):
            matches = OddsAPIService.fetch_upcoming_matches(league_key, api_key)
            st.session_state["fetched_matches"] = matches

    if "fetched_matches" in st.session_state and st.session_state["fetched_matches"]:
        matches = st.session_state["fetched_matches"]
        match_options = [f"{m['home_team']} vs {m['away_team']}" for m in matches]
        selected_match_idx = st.sidebar.selectbox("Selecciona Partido:", range(len(match_options)), format_func=lambda i: match_options[i])
        
        match_data = matches[selected_match_idx]
        match_title = f"{match_data['home_team']} vs {match_data['away_team']}"
        for bookmaker in match_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == match_data["home_team"]: auto_odds_1 = outcome["price"]
                        elif outcome["name"] == match_data["away_team"]: auto_odds_2 = outcome["price"]
                        elif outcome["name"] == "Draw": auto_odds_x = outcome["price"]
        st.sidebar.success("✅ Cuotas cargadas correctamente.")

st.sidebar.divider()
st.sidebar.subheader("2. Parámetros xG del Modelo")
f_lambda = st.sidebar.number_input("xG Local (λ)", value=1.75, step=0.05)
f_mu = st.sidebar.number_input("xG Visitante (μ)", value=0.95, step=0.05)

st.sidebar.subheader("3. Cuotas del Mercado")
odds_1 = st.sidebar.number_input("Cuota Local (1)", value=float(auto_odds_1), step=0.01)
odds_x = st.sidebar.number_input("Cuota Empate (X)", value=float(auto_odds_x), step=0.01)
odds_2 = st.sidebar.number_input("Cuota Visitante (2)", value=float(auto_odds_2), step=0.01)

ah_line = st.sidebar.number_input("Línea Hándicap Asiático", value=-0.75, step=0.25)
ah_home_odds = st.sidebar.number_input("Cuota AH Local", value=1.95, step=0.01)
ou_line = st.sidebar.number_input("Línea Totales (O/U)", value=2.50, step=0.25)
over_odds = st.sidebar.number_input("Cuota Over", value=float(auto_over_odds), step=0.01)
pin_ah = st.sidebar.number_input("Pinnacle AH Local (Ref)", value=1.91, step=0.01)

# COMPUTAR AUDITORÍA
matrix, audit = QuantAuditor.audit(
    f_lambda, f_mu,
    odds_1, odds_x, odds_2,
    ah_line, ah_home_odds,
    ou_line, over_odds,
    pin_ah
)

# METRICAS PRINCIPALES
st.subheader(f"🏟️ {match_title}")

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("EV Hándicap Asiático", f"{audit.ev_ah_pct:+.2f}%")
m_col2.metric("EV Mercado 1X2", f"{audit.ev_1x2_pct:+.2f}%")
m_col3.metric("EV Totales (O/U)", f"{audit.ev_totals_pct:+.2f}%")
m_col4.metric("xG Implícito Casa", f"{audit.implied_lambda} / {audit.implied_mu}")
m_col5.metric("Nivel Insider (Shin z)", f"{audit.shin_z:.4f}")

st.divider()

# ==============================================================================
# VISUALIZACIÓN GRÁFICA RBUSTA
# ==============================================================================
col_chart1, col_chart2 = st.columns([1.1, 1])

with col_chart1:
    st.markdown("### 🔥 Mapa de Calor: Marcadores Exactos (0 a 5 Goles)")
    df_heatmap = pd.DataFrame(
        np.round(matrix[:6, :6] * 100, 2),
        index=[f"Local {i}" for i in range(6)],
        columns=[f"Vis {j}" for j in range(6)]
    )
    
    if HAS_PLOTLY:
        fig_matrix = px.imshow(
            df_heatmap,
            labels=dict(x="Goles Visitante", y="Goles Local", color="Probabilidad (%)"),
            color_continuous_scale="Viridis",
            text_auto=True
        )
        fig_matrix.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_matrix, use_container_width=True)
    else:
        # Respaldo nativo con Pandas
        st.dataframe(
            df_heatmap.style.background_gradient(cmap="mako", axis=None),
            use_container_width=True
        )

with col_chart2:
    st.markdown("### 📊 Tu Modelo vs. Mercado Desmarcado (Shin)")
    df_compare = pd.DataFrame({
        "Tu Modelo (%)": [audit.model_p1 * 100, audit.model_px * 100, audit.model_p2 * 100],
        "Mercado Shin (%)": [audit.shin_p1 * 100, audit.shin_px * 100, audit.shin_p2 * 100]
    }, index=["Victoria Local (1)", "Empate (X)", "Victoria Visitante (2)"])
    
    if HAS_PLOTLY:
        df_reset = df_compare.reset_index().rename(columns={'index': 'Resultado'})
        fig_bars = px.bar(
            df_reset, 
            x="Resultado", 
            y=["Tu Modelo (%)", "Mercado Shin (%)"], 
            barmode="group",
            color_discrete_sequence=["#238636", "#8957E5"]
        )
        fig_bars.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(title=""))
        st.plotly_chart(fig_bars, use_container_width=True)
    else:
        # Respaldo nativo con Streamlit Bar Chart
        st.bar_chart(df_compare, height=380)

st.divider()

# ==============================================================================
# REPORTES DE AUDITORÍA EJECUTIVA
# ==============================================================================
st.markdown("## 📋 Reporte Ejecutivo & Dictamen de Riesgo")

box_class = "report-box-approved" if audit.is_valid else "report-box-rejected"
st.markdown(f"""
<div class="{box_class}">
    <h3 style="margin-top:0;">{audit.status_title}</h3>
    <p><b>Diagnóstico Estocástico:</b> Evaluación cruzada de expectativas ofensivas frente a la estructura de precios sin overround del bookmaker.</p>
</div>
""", unsafe_allow_html=True)

if audit.warnings:
    for w in audit.warnings:
        st.error(w)

tab_report, tab_reverse, tab_docs = st.tabs([
    "📑 Informe Cuantitativo de Operación", 
    "🔬 Deconvolución de Cuotas",
    "📚 Fundamentos Matemáticos"
])

with tab_report:
    c_rep1, c_rep2 = st.columns(2)
    
    with c_rep1:
        st.markdown("#### 1. Métrica de Mercado y Asimetría")
        st.write(f"- **Métrica Insider ($z$):** `{audit.shin_z:.4f}`")
        if audit.shin_z > 0.03:
            st.warning("⚠️ **Presencia de flujo informado:** La casa de apuestas ha ajustado márgenes por movimiento de apostadores pesados o información privada.")
        else:
            st.info("ℹ️ **Mercado balanceado:** Liquidez pública estándar sin signos de manipulación insider.")

        diff_lh = f_lambda - audit.implied_lambda
        diff_mu = f_mu - audit.implied_mu
        st.write(f"- **Diferencial xG Local ($\Delta \lambda$):** `{diff_lh:+.2f}` goles")
        st.write(f"- **Diferencial xG Visitante ($\Delta \mu$):** `{diff_mu:+.2f}` goles")

    with c_rep2:
        st.markdown("#### 2. Plan de Gestión de Capital (Kelly)")
        if audit.is_valid and audit.kelly_stake_pct > 0:
            st.success(f"""
            **ORDEN DE EJECUCIÓN:**
            * **Mercado Objetivo:** Hándicap Asiático `{ah_line}` Local
            * **Cuota Mínima Exigida:** `{1.0 / audit.model_p1:.2f}`
            * **Cuota Capturada:** `{ah_home_odds}`
            * **Esperanza Matemática (+EV):** `+{audit.ev_ah_pct}%`
            * **Stake Sugerido (Kelly 1/5):** `{audit.kelly_stake_pct}%` del Bankroll
            """)
        else:
            st.error("""
            **RECOMENDACIÓN:**
            * **Acción:** NO OPERAR / ABSTENERSE
            * **Causa:** Ausencia de esperanza matemática suficiente o divergencia crítica entre 1X2 y Hándicap Asiático.
            """)

with tab_reverse:
    st.markdown("### Ingeniería Inversa SciPy (Algoritmo L-BFGS-B)")
    st.write("Goles esperados desnudados directamente del precio implícito del libro:")
    
    c_r1, c_r2 = st.columns(2)
    c_r1.metric("xG Local Implícito Casa", f"{audit.implied_lambda} goles")
    c_r2.metric("xG Visitante Implícito Casa", f"{audit.implied_mu} goles")

with tab_docs:
    st.markdown("""
    #### Formulación Técnica
    1. **Poisson Bivariado (Dixon-Coles):** Incorpora corrección $\\tau_{x,y}(\\rho)$ para marcadores de baja anotación ($0$-$0$, $1$-$0$, $0$-$1$, $1$-$1$).
    2. **Desmarcado de Shin:** Utiliza optimización iterativa para calcular la probabilidad real despejando el sesgo de información $z$.
    3. **Filtro Anti-Trampas:** Deniega posturas si existe divergencia de EV entre el mercado 1X2 y el mercado profundo de Hándicap Asiático.
    """)
