import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import requests

# ==============================================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="Terminal Quant v14.0 - Full API & Engine",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PLOTLY_CONFIG_STATIC = {'staticPlot': True, 'responsive': True}

st.markdown("""
<style>
    .stApp { background-color: #0a0d14; color: #d1d5db; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    
    .header-card {
        background: linear-gradient(135deg, #131b2e 0%, #0d1322 100%);
        border: 1px solid #1e293b; border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 14px;
    }
    .header-title { font-size: 1.3em; font-weight: 800; color: #ffffff; }
    .header-subtitle { font-size: 0.70em; font-weight: 700; color: #64748b; text-transform: uppercase; margin-top: 2px; }

    .quant-grid { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important; margin-bottom: 14px !important; }
    .quant-card { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center; }
    .quant-card-full { grid-column: span 2 !important; background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center; }
    .quant-label { font-size: 0.65em; font-weight: 700; color: #9ca3af; text-transform: uppercase; }
    .quant-val-negative { font-size: 1.2em; font-weight: 800; color: #ef4444; }
    .quant-val-positive { font-size: 1.2em; font-weight: 800; color: #10b981; }
    .quant-val-neutral { font-size: 1.2em; font-weight: 800; color: #f59e0b; }
    .quant-val-white { font-size: 1.2em; font-weight: 800; color: #f9fafb; }

    .box-ev { background-color: rgba(6, 78, 59, 0.25); border: 1px solid #059669; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
    .box-prob { background-color: rgba(30, 58, 138, 0.25); border: 1px solid #2563eb; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
    .badge-ev { background-color: #10b981; color: #022c22; font-weight: 800; font-size: 0.65em; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; }
    .badge-prob { background-color: #3b82f6; color: #ffffff; font-weight: 800; font-size: 0.65em; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; }
    .box-title { font-size: 1.0em; font-weight: 800; color: #ffffff; margin-top: 4px; }
    .box-desc { font-size: 0.80em; color: #9ca3af; margin-top: 4px; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONEXIÓN API Y EXTRACCIÓN DE MERCADOS EN TIEMPO REAL
# ==============================================================================
LIGAS_DISPONIBLES = {
    "Copa Libertadores": "soccer_conmebol_copa_libertadores",
    "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
    "Premier League (Inglaterra)": "soccer_epl",
    "LaLiga (España)": "soccer_spain_la_liga",
    "Serie A (Italia)": "soccer_italy_serie_a",
    "Bundesliga (Alemania)": "soccer_germany_bundesliga",
    "Liga Profesional (Argentina)": "soccer_argentina_primera_division",
    "Brasileirão (Brasil)": "soccer_brazil_campeonato"
}

def obtener_partidos_odds_api(api_key, sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {'apiKey': api_key, 'regions': 'eu,us', 'markets': 'h2h,spreads,totals', 'oddsFormat': 'decimal'}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json(), None
        return None, f"Error API ({res.status_code})"
    except Exception as e:
        return None, str(e)

def extraer_mercados(partido_data):
    local = partido_data.get('home_team', 'Local')
    visita = partido_data.get('away_team', 'Visitante')
    
    c1, cx, c2 = 2.20, 3.10, 3.00
    cover, cunder = 1.95, 1.85
    ah_line, c_ah1 = -0.25, 1.90

    bookmakers = partido_data.get('bookmakers', [])
    if bookmakers:
        bm = bookmakers[0]
        for m in bm.get('markets', []):
            if m['key'] == 'h2h':
                for o in m['outcomes']:
                    if o['name'] == local: c1 = float(o['price'])
                    elif o['name'] == visita: c2 = float(o['price'])
                    elif o['name'] == 'Draw': cx = float(o['price'])
            elif m['key'] == 'totals':
                for o in m['outcomes']:
                    if o['name'] == 'Over': cover = float(o['price'])
                    elif o['name'] == 'Under': cunder = float(o['price'])
            elif m['key'] == 'spreads':
                for o in m['outcomes']:
                    if o['name'] == local:
                        ah_line = float(o.get('point', -0.25))
                        c_ah1 = float(o['price'])

    return {
        "local": local, "visitante": visita,
        "c1": c1, "cx": cx, "c2": c2,
        "cover": cover, "cunder": cunder,
        "ah_local": ah_line, "c_ah1": c_ah1
    }

# ==============================================================================
# 3. MOTOR MATEMÁTICO AVANZADO
# ==============================================================================
def estimar_shin(cuotas_1x2):
    """Desmarginalización Cuantitativa por Modelo de Shin (Convergencia Suave)"""
    c = np.array(cuotas_1x2, dtype=float)
    if np.any(c <= 1.0): return np.array([0.333, 0.333, 0.334]), 0.0
    inv_c = 1.0 / c
    sum_inv = np.sum(inv_c)
    margin = sum_inv - 1.0
    
    z = max(0.0001, margin * 0.20)
    for _ in range(20):
        p_raw = (np.sqrt(z**2 + 4 * (1 - z) * (inv_c / sum_inv)) - z) / (2 * (1 - z))
        diff = np.sum(p_raw) - 1.0
        z += diff * 0.05
        z = max(0.00001, min(0.6, z))
        
    p_final = p_raw / np.sum(p_raw)
    return p_final, z

def modelo_dixon_coles(lambda_h, mu_a, rho=-0.13, max_goles=7):
    """Simulación Bivariada Dixon & Coles con Corrección Tau"""
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p_i = stats.poisson.pmf(i, lambda_h)
            p_j = stats.poisson.pmf(j, mu_a)
            if i == 0 and j == 0: tau = 1.0 - (lambda_h * mu_a * rho)
            elif i == 0 and j == 1: tau = 1.0 + (lambda_h * rho)
            elif i == 1 and j == 0: tau = 1.0 + (mu_a * rho)
            elif i == 1 and j == 1: tau = 1.0 - rho
            else: tau = 1.0
            mat[i, j] = max(0.0, p_i * p_j * tau)
    mat = mat / np.sum(mat)
    
    p_home = np.sum(np.tril(mat, -1))
    p_draw = np.sum(np.diag(mat))
    p_away = np.sum(np.triu(mat, 1))
    p_over_25 = np.sum([mat[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    return p_home, p_draw, p_away, p_over_25, 1.0 - p_over_25, mat

def calcular_handicap_asiatico(mat_goles, linea_ah):
    """Motor de Resolución Exacta de Hándicaps Asiáticos"""
    max_goles = mat_goles.shape[0] - 1
    prob_win, prob_half_win, prob_push = 0.0, 0.0, 0.0
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            diff = i - j + linea_ah
            p = mat_goles[i, j]
            if diff > 0.25: prob_win += p
            elif diff == 0.25: prob_half_win += p
            elif diff == 0.0: prob_push += p
    return prob_win + (prob_half_win * 0.5)

def simular_auxiliares(lambda_h, mu_a):
    """Simulación Poisson para Córners y Tarjetas"""
    lambda_c_h, lambda_c_a = max(2.5, lambda_h * 3.1), max(1.8, mu_a * 2.7)
    tot_corners = lambda_c_h + lambda_c_a
    c_h, c_a = stats.poisson.pmf(np.arange(0, 21), lambda_c_h), stats.poisson.pmf(np.arange(0, 21), lambda_c_a)
    mat_c = np.outer(c_h, c_a)
    
    p_c85 = np.sum([mat_c[i, j] for i in range(21) for j in range(21) if i + j > 8.5])
    p_c95 = np.sum([mat_c[i, j] for i in range(21) for j in range(21) if i + j > 9.5])
    
    lambda_cards = max(3.0, (lambda_h + mu_a) * 1.4 + 1.2)
    p_t35 = 1.0 - stats.poisson.cdf(3, lambda_cards)
    p_t45 = 1.0 - stats.poisson.cdf(4, lambda_cards)

    return tot_corners, p_c85, p_c95, lambda_cards, p_t35, p_t45

def kelly_criterion(prob, cuota, fraction=0.25):
    """Criterio de Kelly Fraccionado Sin Redondeo Nulo"""
    b = cuota - 1.0
    if b <= 0: return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f * fraction)

# ==============================================================================
# 4. BARRA LATERAL (CONEXIÓN API + ENTRADA MANUAL)
# ==============================================================================
st.sidebar.markdown("### 🔑 API & NAVEGACIÓN")
odds_api_key = st.sidebar.text_input("The Odds API Key:", value="1a927e0f762a52540fe079d963ed2460", type="password")

liga_nombre = st.sidebar.selectbox("Torneo / Liga:", list(LIGAS_DISPONIBLES.keys()))
sport_key = LIGAS_DISPONIBLES[liga_nombre]

partidos_raw, error_api = obtener_partidos_odds_api(odds_api_key, sport_key)

if partidos_raw and len(partidos_raw) > 0:
    opciones_partidos = [f"{m.get('home_team')} vs {m.get('away_team')}" for m in partidos_raw]
    idx_partido = st.sidebar.selectbox("Seleccionar Evento:", range(len(opciones_partidos)), format_func=lambda x: opciones_partidos[x])
    p_data = extraer_mercados(partidos_raw[idx_partido])
else:
    if error_api:
        st.sidebar.warning(f"API no disponible ({error_api}). Usando valores por defecto.")
    p_data = {
        "local": "Macará", "visitante": "Santos",
        "c1": 2.20, "cx": 3.10, "c2": 3.00,
        "cover": 1.95, "cunder": 1.85,
        "ah_local": -0.25, "c_ah1": 1.90
    }

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AJUSTE DE CUOTAS")
p_local = st.sidebar.text_input("Equipo Local:", p_data['local'])
p_visita = st.sidebar.text_input("Equipo Visitante:", p_data['visitante'])

c_1 = st.sidebar.number_input(f"Cuota 1 ({p_local}):", value=float(p_data['c1']), step=0.01)
c_x = st.sidebar.number_input("Cuota X (Empate):", value=float(p_data['cx']), step=0.01)
c_2 = st.sidebar.number_input(f"Cuota 2 ({p_visita}):", value=float(p_data['c2']), step=0.01)
c_over = st.sidebar.number_input("Cuota Over 2.5 Goles:", value=float(p_data['cover']), step=0.01)
c_under = st.sidebar.number_input("Cuota Under 2.5 Goles:", value=float(p_data['cunder']), step=0.01)
ah_line = st.sidebar.number_input("Línea Hándicap Asiático:", value=float(p_data['ah_local']), step=0.25)
c_ah1 = st.sidebar.number_input("Cuota Hándicap Local:", value=float(p_data['c_ah1']), step=0.01)

# ==============================================================================
# 5. CÁLCULO GENERAL Y EVALUACIÓN DE INFRAVALORADOS
# ==============================================================================
probs_shin, insider_z = estimar_shin([c_1, c_x, c_2])
total_xg = 2.72
lambda_h = (probs_shin[0] * total_xg * 1.08) / (probs_shin[0] + probs_shin[2] + 1e-5)
mu_a = (probs_shin[2] * total_xg) / (probs_shin[0] + probs_shin[2] + 1e-5)

p_dc_1, p_dc_x, p_dc_2, p_dc_over, p_dc_under, mat_dc = modelo_dixon_coles(lambda_h, mu_a)
p_ah = calcular_handicap_asiatico(mat_dc, ah_line)
exp_corn, p_c85, p_c95, exp_cards, p_t35, p_t45 = simular_auxiliares(lambda_h, mu_a)

ev_1 = (p_dc_1 * c_1 - 1.0) * 100
ev_x = (p_dc_x * c_x - 1.0) * 100
ev_2 = (p_dc_2 * c_2 - 1.0) * 100
ev_o = (p_dc_over * c_over - 1.0) * 100
ev_u = (p_dc_under * c_under - 1.0) * 100
ev_ah = (p_ah * c_ah1 - 1.0) * 100

ev_t35 = (p_t35 * 1.45 - 1.0) * 100
ev_c85 = (p_c85 * 1.55 - 1.0) * 100

mercados_evaluados = [
    (f"Hándicap {ah_line} {p_local}", p_ah, ev_ah, c_ah1, 1/p_ah, "Medio"),
    (f"Victoria Local ({p_local})", p_dc_1, ev_1, c_1, 1/p_dc_1, "Bajo"),
    ("Empate (X)", p_dc_x, ev_x, c_x, 1/p_dc_x, "Bajo"),
    (f"Victoria Visitante ({p_visita})", p_dc_2, ev_2, c_2, 1/p_dc_2, "Bajo"),
    ("Over 2.5 Goles", p_dc_over, ev_o, c_over, 1/p_dc_over, "Medio"),
    ("Under 2.5 Goles", p_dc_under, ev_u, c_under, 1/p_dc_under, "Medio"),
    ("Over 3.5 Tarjetas", p_t35, ev_t35, 1.45, 1/p_t35, "ALTO (Infravalorado)"),
    ("Over 8.5 Córners", p_c85, ev_c85, 1.55, 1/p_c85, "ALTO (Infravalorado)")
]

pick_mayor_valor = max(mercados_evaluados, key=lambda x: x[2])
pick_mayor_prob = max(mercados_evaluados, key=lambda x: x[1])
kelly_val = kelly_criterion(pick_mayor_valor[1], pick_mayor_valor[3]) * 100

# ==============================================================================
# 6. DESPLIEGUE EN INTERFAZ
# ==============================================================================
st.markdown(f"""
<div class="header-card">
    <div class="header-title">🏟️ {p_local} vs {p_visita}</div>
    <div class="header-subtitle">Terminal de Inteligencia Cuantitativa</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="quant-grid">
    <div class="quant-card">
        <div class="quant-label">EV HÁNDICAP</div>
        <div class="{'quant-val-positive' if ev_ah > 0 else 'quant-val-negative'}">{ev_ah:+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV MERCADO 1X2</div>
        <div class="{'quant-val-positive' if max(ev_1,ev_x,ev_2) > 0 else 'quant-val-negative'}">{max(ev_1,ev_x,ev_2):+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV OVER/UNDER</div>
        <div class="{'quant-val-positive' if max(ev_o,ev_u) > 0 else 'quant-val-negative'}">{max(ev_o,ev_u):+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">XG CASA IMPL.</div>
        <div class="quant-val-white">{lambda_h:.2f} / {mu_a:.2f}</div>
    </div>
    <div class="quant-card-full">
        <div class="quant-label">SHIN (INSIDER Z)</div>
        <div class="quant-val-neutral">{insider_z:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if pick_mayor_valor[2] > 0:
    st.markdown(f"""
    <div class="box-ev">
        <span class="badge-ev">ORDEN APROBADA (+EV)</span>
        <div class="box-title">{pick_mayor_valor[0]}</div>
        <div class="box-desc">
            • Ventaja Esperada (+EV): <b>+{pick_mayor_valor[2]:.2f}%</b><br>
            • Cuota Bookie: <b>{pick_mayor_valor[3]:.2f}</b> | Cuota Justa (Fair): <b>{pick_mayor_valor[4]:.2f}</b><br>
            • Probabilidad Real: <b>{pick_mayor_valor[1]*100:.1f}%</b> | Asignación Kelly (0.25x): <b>{kelly_val:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="box-prob">
    <span class="badge-prob">MAYOR PROBABILIDAD ESTADÍSTICA</span>
    <div class="box-title">{pick_mayor_prob[0]}</div>
    <div class="box-desc">
        • Probabilidad de Acierto: <b>{pick_mayor_prob[1]*100:.1f}%</b><br>
        • Valor Esperado (+EV): <b>{pick_mayor_prob[2]:+.2f}%</b> | Cuota: <b>{pick_mayor_prob[3]:.2f}</b>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "🔍 Detección de Ineficiencias", "🎯 Córners & Tarjetas"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = go.Figure(data=[
            go.Bar(name='Modelo', x=['1', 'X', '2'], y=[p_dc_1*100, p_dc_x*100, p_dc_2*100], marker_color='#10b981'),
            go.Bar(name='Shin', x=['1', 'X', '2'], y=[probs_shin[0]*100, probs_shin[1]*100, probs_shin[2]*100], marker_color='#3b82f6')
        ])
        fig_bar.update_layout(title="Comparativa 1X2 (%)", height=200, margin=dict(l=5,r=5,t=25,b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', size=9))
        fig_bar.update_xaxes(fixedrange=True); fig_bar.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG_STATIC)
    with col2:
        x_g = np.linspace(0, 6, 100)
        fig_density = go.Figure(go.Scatter(x=x_g, y=stats.norm.pdf(x_g, loc=(lambda_h+mu_a), scale=1.0), mode='lines', fill='tozeroy', line=dict(color='#10b981')))
        fig_density.update_layout(title="Distribución Goles", height=200, margin=dict(l=5,r=5,t=25,b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', size=9))
        fig_density.update_xaxes(fixedrange=True); fig_density.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_density, use_container_width=True, config=PLOTLY_CONFIG_STATIC)

with tab2:
    tabla_data = {
        "Mercado": [m[0] for m in mercados_evaluados],
        "Prob. Modelo": [f"{m[1]*100:.1f}%" for m in mercados_evaluados],
        "Cuota Fair": [f"{m[4]:.2f}" for m in mercados_evaluados],
        "Cuota Bookie": [f"{m[3]:.2f}" for m in mercados_evaluados],
        "EV (%)": [f"{m[2]:+.2f}%" for m in mercados_evaluados],
        "Nivel Ineficiencia": [m[5] for m in mercados_evaluados]
    }
    st.dataframe(tabla_data, use_container_width=True)

with tab3:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"**Tiros de Esquina Esperados:** `{exp_corn:.1f}`")
        st.caption(f"• Over 8.5 Córners: **{p_c85*100:.1f}%**")
        st.caption(f"• Over 9.5 Córners: **{p_c95*100:.1f}%**")
    with col_c2:
        st.markdown(f"**Tarjetas Esperadas:** `{exp_cards:.1f}`")
        st.caption(f"• Over 3.5 Tarjetas: **{p_t35*100:.1f}%**")
        st.caption(f"• Over 4.5 Tarjetas: **{p_t45*100:.1f}%**")
