import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import requests

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS COMPLETOS
# ==============================================================================
st.set_page_config(
    page_title="Terminal Quant v9.0 - Full Engine",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLOTLY_CONFIG = {
    'displayModeBar': False,
    'scrollZoom': False,
    'doubleClick': False,
    'showAxisDragHandles': False
}

st.markdown("""
<style>
    .stApp { 
        background-color: #0a0d14; 
        color: #d1d5db; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    }
    
    /* Header Principal */
    .header-card {
        background: linear-gradient(135deg, #131b2e 0%, #0d1322 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header-title { font-size: 1.5em; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }
    .header-subtitle { font-size: 0.75em; font-weight: 700; color: #64748b; letter-spacing: 1.5px; margin-top: 4px; text-transform: uppercase; }

    /* Grilla de Métricas Móvil (2 Columnas Rígidas) */
    .quant-grid {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 12px !important;
        margin-bottom: 20px !important;
    }
    .quant-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .quant-card-full {
        grid-column: span 2 !important;
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .quant-label { font-size: 0.72em; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; }
    .quant-val-negative { font-size: 1.4em; font-weight: 800; color: #ef4444; margin-top: 4px; }
    .quant-val-positive { font-size: 1.4em; font-weight: 800; color: #10b981; margin-top: 4px; }
    .quant-val-neutral { font-size: 1.4em; font-weight: 800; color: #f59e0b; margin-top: 4px; }
    .quant-val-white { font-size: 1.4em; font-weight: 800; color: #f9fafb; margin-top: 4px; }

    /* Módulo de Orden Aprobada / Rechazada */
    .execution-approved {
        background-color: rgba(6, 78, 59, 0.25);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 24px;
    }
    .execution-rejected {
        background-color: rgba(127, 29, 29, 0.25);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 24px;
    }
    .badge-approved { background-color: #10b981; color: #022c22; font-weight: 800; font-size: 0.75em; padding: 4px 12px; border-radius: 20px; text-transform: uppercase; }
    .badge-rejected { background-color: #ef4444; color: #450a0a; font-weight: 800; font-size: 0.75em; padding: 4px 12px; border-radius: 20px; text-transform: uppercase; }
    .execution-title { font-size: 1.2em; font-weight: 800; color: #ffffff; margin-top: 10px; }
    .execution-desc { font-size: 0.85em; color: #9ca3af; margin-top: 6px; line-height: 1.5; }

    /* Cajas Informativas Extra */
    .info-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANCO DE DATOS DE LIGAS Y CONEXIÓN CON THE ODDS API
# ==============================================================================
LIGAS_DISPONIBLES = {
    "Copa Libertadores": "soccer_conmebol_copa_libertadores",
    "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
    "Premier League (Inglaterra)": "soccer_epl",
    "LaLiga (España)": "soccer_spain_la_liga",
    "Serie A (Italia)": "soccer_italy_serie_a",
    "Bundesliga (Alemania)": "soccer_germany_bundesliga",
    "Ligue 1 (Francia)": "soccer_france_ligue_one",
    "Liga Profesional (Argentina)": "soccer_argentina_primera_division",
    "Brasileirão (Brasil)": "soccer_brazil_campeonato",
    "MLS (EE.UU.)": "soccer_usa_mls"
}

def obtener_partidos_odds_api(api_key, sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'eu,us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal'
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json(), None
        else:
            return None, f"Error API ({res.status_code}): {res.json().get('message', 'Clave inválida o límite superado')}"
    except Exception as e:
        return None, f"Error de conexión: {str(e)}"

def extraer_mercados(partido_data):
    local = partido_data.get('home_team', 'Local')
    visita = partido_data.get('away_team', 'Visitante')
    
    c1, cx, c2 = 2.35, 3.30, 2.95
    cover, cunder = 2.05, 1.75
    ah_line, c_ah1 = -0.25, 1.95

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
        "local": local,
        "visitante": visita,
        "c1": c1, "cx": cx, "c2": c2,
        "cover": cover, "cunder": cunder,
        "ah_local": ah_line, "c_ah1": c_ah1
    }

# ==============================================================================
# 3. MOTOR MATEMÁTICO COMPLETO: SHIN, DIXON-COLES Y KELLY
# ==============================================================================
def estimar_shin(cuotas_1x2):
    """Algoritmo numérico para desmarginalizar cuotas y extraer la probabilidad limpia y el factor Z de Shin."""
    c = np.array(cuotas_1x2, dtype=float)
    if np.any(c <= 1.0): 
        return np.array([0.333, 0.333, 0.334]), 0.0
    
    inv_c = 1.0 / c
    sum_inv = np.sum(inv_c)
    margin = sum_inv - 1.0
    
    # Búsqueda iterativa del factor z de Shin
    z = max(0.0001, margin * 0.18)
    for _ in range(10):
        p_raw = (np.sqrt(z**2 + 4 * (1 - z) * (inv_c / sum_inv)) - z) / (2 * (1 - z))
        z = max(0.0001, np.sum(p_raw) - 1.0 + z)
        
    p_norm = p_raw / np.sum(p_raw)
    return p_norm, z

def modelo_dixon_coles(lambda_h, mu_a, rho=-0.13, max_goles=5):
    """Calcula la matriz completa de probabilidades bivariada (0-5 goles por equipo) con parámetro tau."""
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
            
    mat = mat / np.sum(mat) # Normalización matricial
    
    p_home = np.sum(np.tril(mat, -1))
    p_draw = np.sum(np.diag(mat))
    p_away = np.sum(np.triu(mat, 1))
    p_over = np.sum([mat[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    p_under = 1.0 - p_over
    
    return p_home, p_draw, p_away, p_over, p_under, mat

def calcular_kelly(prob, cuota, fraccion=0.25):
    """Calcula el porcentaje de banca a arriesgar mediante el Criterio de Kelly Fraccional."""
    b = cuota - 1.0
    q = 1.0 - prob
    f_kelly = (b * prob - q) / b
    return max(0.0, f_kelly * fraccion) * 100

# ==============================================================================
# 4. CONTROLES Y SELECCIÓN EN BARRA LATERAL
# ==============================================================================
st.sidebar.markdown("### 🗝️ THE ODDS API KEY")
odds_api_key = st.sidebar.text_input("API Key:", value="1a927e0f762a52540fe079d963ed2460", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 SELECCIÓN DE TORNEO Y PARTIDO")
liga_nombre = st.sidebar.selectbox("Torneo:", list(LIGAS_DISPONIBLES.keys()))
sport_key_selected = LIGAS_DISPONIBLES[liga_nombre]

partidos_raw, error_api = obtener_partidos_odds_api(odds_api_key, sport_key_selected)

if error_api:
    st.sidebar.error(error_api)
    p = {"local": "Macará", "visitante": "Santos", "c1": 2.33, "cx": 3.40, "c2": 2.90, "cover": 2.10, "cunder": 1.70, "ah_local": -0.25, "c_ah1": 1.90}
else:
    if partidos_raw:
        opciones_partidos = [f"{m.get('home_team')} vs {m.get('away_team')}" for m in partidos_raw]
        idx_p = st.sidebar.selectbox("Evento en Vivo:", range(len(opciones_partidos)), format_func=lambda x: opciones_partidos[x])
        p = extraer_mercados(partidos_raw[idx_p])
    else:
        st.sidebar.warning("No se encontraron eventos activos próximos para este torneo.")
        p = {"local": "Macará", "visitante": "Santos", "c1": 2.33, "cx": 3.40, "c2": 2.90, "cover": 2.10, "cunder": 1.70, "ah_local": -0.25, "c_ah1": 1.90}

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AJUSTE MANUAL DE CUOTAS")
c_1 = st.sidebar.number_input(f"1 ({p['local']}):", value=float(p['c1']), step=0.01)
c_x = st.sidebar.number_input("X (Empate):", value=float(p['cx']), step=0.01)
c_2 = st.sidebar.number_input(f"2 ({p['visitante']}):", value=float(p['c2']), step=0.01)
c_over = st.sidebar.number_input("Over 2.5 Goles:", value=float(p['cover']), step=0.01)
c_under = st.sidebar.number_input("Under 2.5 Goles:", value=float(p['cunder']), step=0.01)
ah_line = st.sidebar.number_input("Línea Hándicap Asiático:", value=float(p['ah_local']), step=0.25)
c_ah1 = st.sidebar.number_input("Cuota Hándicap Local:", value=float(p['c_ah1']), step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 GESTIÓN DE RISK / BANKROLL")
bankroll = st.sidebar.number_input("Bankroll Total ($):", value=1000.0, step=50.0)
kelly_frac = st.sidebar.slider("Fracción de Kelly:", 0.1, 1.0, 0.25, step=0.05)

# ==============================================================================
# 5. CÁLCULOS CUANTITATIVOS Y ANÁLISIS COMPLETO
# ==============================================================================
probs_shin, insider_z = estimar_shin([c_1, c_x, c_2])

# Reverse Engineering de xG
total_xg_est = 2.60
lambda_h = (probs_shin[0] * total_xg_est * 1.08) / (probs_shin[0] + probs_shin[2] + 1e-5)
mu_a = (probs_shin[2] * total_xg_est) / (probs_shin[0] + probs_shin[2] + 1e-5)

# Modelo Dixon-Coles
p_dc_1, p_dc_x, p_dc_2, p_dc_over, p_dc_under, mat_dc = modelo_dixon_coles(lambda_h, mu_a)

# Expected Values (+EV / -EV)
ev_1 = (p_dc_1 * c_1 - 1.0) * 100
ev_x = (p_dc_x * c_x - 1.0) * 100
ev_2 = (p_dc_2 * c_2 - 1.0) * 100
ev_1x2 = max(ev_1, ev_x, ev_2)

ev_o = (p_dc_over * c_over - 1.0) * 100
ev_u = (p_dc_under * c_under - 1.0) * 100
ev_ou = max(ev_o, ev_u)

prob_ah_cover = p_dc_1 + (p_dc_x * 0.5 if abs(ah_line) == 0.25 else 0)
ev_handicap = (prob_ah_cover * c_ah1 - 1.0) * 100

max_ev_global = max(ev_1x2, ev_ou, ev_handicap)
ejecucion_permitida = max_ev_global > 0

# Determinación de la mejor opción cuantitativa
opciones_ev = [
    ("1X2 Local", ev_1, p_dc_1, c_1),
    ("1X2 Empate", ev_x, p_dc_x, c_x),
    ("1X2 Visitante", ev_2, p_dc_2, c_2),
    ("Over 2.5 Goles", ev_o, p_dc_over, c_over),
    ("Under 2.5 Goles", ev_u, p_dc_under, c_under),
    (f"Hándicap {ah_line} Local", ev_handicap, prob_ah_cover, c_ah1)
]
mejor_apuesta = max(opciones_ev, key=lambda x: x[1])

pct_stake_kelly = calcular_kelly(mejor_apuesta[2], mejor_apuesta[3], fraccion=kelly_frac)
monto_stake = (pct_stake_kelly / 100.0) * bankroll

# ==============================================================================
# 6. INTERFAZ Y RENDERIZADO VISUAL
# ==============================================================================

# Header Principal
st.markdown(f"""
<div class="header-card">
    <div class="header-title">🏟️ {p['local']} vs {p['visitante']}</div>
    <div class="header-subtitle">Terminal de Inteligencia Cuantitativa</div>
</div>
""", unsafe_allow_html=True)

# Grilla de 5 Métricas Clave (2x2 + Card Ancha)
st.markdown(f"""
<div class="quant-grid">
    <div class="quant-card">
        <div class="quant-label">EV HÁNDICAP</div>
        <div class="{'quant-val-positive' if ev_handicap > 0 else 'quant-val-negative'}">{ev_handicap:+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV MERCADO 1X2</div>
        <div class="{'quant-val-positive' if ev_1x2 > 0 else 'quant-val-negative'}">{ev_1x2:+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV OVER/UNDER</div>
        <div class="{'quant-val-positive' if ev_ou > 0 else 'quant-val-negative'}">{ev_ou:+.1f}%</div>
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

# Módulo de Decisión y Ejecución de Órdenes
if ejecucion_permitida:
    st.markdown(f"""
    <div class="execution-approved">
        <span class="badge-approved">EJECUCIÓN PERMITIDA</span>
        <div class="execution-title">Orden Aprobada (+EV): {mejor_apuesta[0]}</div>
        <div class="execution-desc">
            Auditoría cuantitativa completada mediante desmarginalización de Shin y simulación bivariada de Dixon-Coles.<br>
            • Ventaja Esperada (+EV): <b>+{mejor_apuesta[1]:.2f}%</b><br>
            • Cuota Recomendada: <b>{mejor_apuesta[3]:.2f}</b> (Probabilidad Real: <b>{mejor_apuesta[2]*100:.1f}%</b>)<br>
            • Asignación de Banca (Kelly {kelly_frac}x): <b>{pct_stake_kelly:.2f}% (${monto_stake:.2f})</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="execution-rejected">
        <span class="badge-rejected">EJECUCIÓN RECHAZADA</span>
        <div class="execution-title">Orden Bloqueada (-EV)</div>
        <div class="execution-desc">
            El cruce entre el modelo Dixon-Coles y las cuotas de-marginadas de Shin no arroja ventaja matemática sobre ningún mercado disponible.<br>
            • Máximo EV detectado: <b>{max_ev_global:+.2f}%</b><br>
            • Recomendación: <b>Abstenerse de operar / Conservar Capital</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Gráficos y Tablas Complementarias
tab1, tab2, tab3 = st.tabs(["📊 Probabilidades & Densidad", "🎯 Matriz de Marcadores Exactos", "📋 Desglose Detallado"])

with tab1:
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        fig_bar = go.Figure(data=[
            go.Bar(name='Modelo (Dixon-Coles)', x=['1', 'X', '2'], y=[p_dc_1*100, p_dc_x*100, p_dc_2*100], marker_color='#10b981'),
            go.Bar(name='Mercado (Shin)', x=['1', 'X', '2'], y=[probs_shin[0]*100, probs_shin[1]*100, probs_shin[2]*100], marker_color='#3b82f6')
        ])
        fig_bar.update_layout(
            title="Comparativa 1X2 (%)",
            barmode='group',
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af', size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    with col_f2:
        x_g = np.linspace(0, 6, 100)
        y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
        fig_density = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#10b981')))
        fig_density.add_vline(x=2.5, line_dash="dash", line_color="#ef4444", annotation_text="Línea 2.5", annotation_position="top right")
        fig_density.update_layout(
            title="Distribución de Goles Esperados",
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af', size=11)
        )
        st.plotly_chart(fig_density, use_container_width=True, config=PLOTLY_CONFIG)

with tab2:
    fig_mat = px.imshow(
        mat_dc * 100,
        labels=dict(x=f"Goles {p['visitante']}", y=f"Goles {p['local']}", color="Prob %"),
        x=[0, 1, 2, 3, 4, 5],
        y=[0, 1, 2, 3, 4, 5],
        color_continuous_scale="Viridis",
        text_auto=".1f"
    )
    fig_mat.update_layout(
        title="Matriz Bivariada de Marcadores Exactos (%)",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af')
    )
    st.plotly_chart(fig_mat, use_container_width=True, config=PLOTLY_CONFIG)

with tab3:
    st.markdown("#### Tabla Desglosada de Expected Value (+EV)")
    data_tabla = {
        "Mercado": [op[0] for op in opciones_ev],
        "Cuota Mercado": [f"{op[3]:.2f}" for op in opciones_ev],
        "Prob. Modelo": [f"{op[2]*100:.1f}%" for op in opciones_ev],
        "Expected Value (EV)": [f"{op[1]:+.2f}%" for op in opciones_ev]
    }
    st.dataframe(data_tabla, use_container_width=True)
