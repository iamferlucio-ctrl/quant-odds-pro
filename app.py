import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import requests

# ==============================================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="Terminal Quant v12.0 - Full Engine",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Bloqueo total de zoom y desplazamiento táctil en gráficos para móviles
PLOTLY_CONFIG_STATIC = {
    'staticPlot': True,
    'responsive': True
}

st.markdown("""
<style>
    .stApp { background-color: #0a0d14; color: #d1d5db; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    
    .header-card {
        background: linear-gradient(135deg, #131b2e 0%, #0d1322 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 16px;
    }
    .header-title { font-size: 1.4em; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }
    .header-subtitle { font-size: 0.72em; font-weight: 700; color: #64748b; letter-spacing: 1.5px; margin-top: 4px; text-transform: uppercase; }

    /* Grilla de Métricas Móvil */
    .quant-grid {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
        margin-bottom: 16px !important;
    }
    .quant-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .quant-card-full {
        grid-column: span 2 !important;
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .quant-label { font-size: 0.70em; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; }
    .quant-val-negative { font-size: 1.35em; font-weight: 800; color: #ef4444; margin-top: 2px; }
    .quant-val-positive { font-size: 1.35em; font-weight: 800; color: #10b981; margin-top: 2px; }
    .quant-val-neutral { font-size: 1.35em; font-weight: 800; color: #f59e0b; margin-top: 2px; }
    .quant-val-white { font-size: 1.35em; font-weight: 800; color: #f9fafb; margin-top: 2px; }

    /* Módulos de Decisiones */
    .box-ev {
        background-color: rgba(6, 78, 59, 0.25);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .box-prob {
        background-color: rgba(30, 58, 138, 0.25);
        border: 1px solid #2563eb;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 16px;
    }
    .badge-ev { background-color: #10b981; color: #022c22; font-weight: 800; font-size: 0.70em; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }
    .badge-prob { background-color: #3b82f6; color: #1e3a8a; font-weight: 800; font-size: 0.70em; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }
    .box-title { font-size: 1.1em; font-weight: 800; color: #ffffff; margin-top: 6px; }
    .box-desc { font-size: 0.82em; color: #9ca3af; margin-top: 4px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONEXIÓN API Y EXTRACCIÓN DE MERCADOS
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
    
    c1, cx, c2 = 2.15, 3.25, 3.40
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
# 3. MOTOR MATEMÁTICO COMPLETO (SHIN, DIXON-COLES, ASIÁTICOS, CÓRNERS, TARJETAS)
# ==============================================================================

def estimar_shin(cuotas_1x2):
    """Desmarginalización Cuantitativa por Modelo de Shin (1992, 1993)"""
    c = np.array(cuotas_1x2, dtype=float)
    if np.any(c <= 1.0): return np.array([0.333, 0.333, 0.334]), 0.0
    inv_c = 1.0 / c
    sum_inv = np.sum(inv_c)
    margin = sum_inv - 1.0
    
    # Solver Numérico para Insider Trading Z
    z = max(0.0001, margin * 0.15)
    for _ in range(10):
        p_raw = (np.sqrt(z**2 + 4 * (1 - z) * (inv_c / sum_inv)) - z) / (2 * (1 - z))
        diff = np.sum(p_raw) - 1.0
        z += diff * 0.1
        z = max(0.00001, min(0.4, z))
        
    p_final = p_raw / np.sum(p_raw)
    return p_final, z

def modelo_dixon_coles(lambda_h, mu_a, rho=-0.13, max_goles=7):
    """Simulación Bivariada Dixon & Coles con Ajuste de Corrección Estocástica Tau"""
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p_i = stats.poisson.pmf(i, lambda_h)
            p_j = stats.poisson.pmf(j, mu_a)
            
            # Matriz de Ajuste Tau para Marcadores Bajas (0-0, 1-0, 0-1, 1-1)
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
    p_under_25 = 1.0 - p_over_25
    
    return p_home, p_draw, p_away, p_over_25, p_under_25, mat

def calcular_handicap_asiatico(mat_goles, linea_ah):
    """Motor de Resolución Exacta de Hándicaps Asiáticos (+0.25, -0.25, -0.5, -0.75, etc.)"""
    max_goles = mat_goles.shape[0] - 1
    prob_win = 0.0
    prob_half_win = 0.0
    prob_push = 0.0
    prob_half_loss = 0.0
    prob_loss = 0.0
    
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            diff = i - j + linea_ah
            p = mat_goles[i, j]
            
            if diff > 0.25: prob_win += p
            elif diff == 0.25: prob_half_win += p
            elif diff == 0.0: prob_push += p
            elif diff == -0.25: prob_half_loss += p
            else: prob_loss += p
            
    # Probabilidad efectiva ponderada para valor esperado
    prob_efectiva = prob_win + (prob_half_win * 0.5) + (prob_push * 0.0) # push es retorno neutral
    return prob_efectiva, prob_win, prob_half_win, prob_push, prob_half_loss, prob_loss

def simular_corners_y_tarjetas(lambda_h, mu_a):
    """Motor Poisson Completo para Córners y Tarjetas por Frecuencia de Intensidad"""
    # Proyección Córners (Local/Visita)
    lambda_corners_h = max(2.5, lambda_h * 3.1)
    lambda_corners_a = max(1.8, mu_a * 2.7)
    total_corners_exp = lambda_corners_h + lambda_corners_a
    
    # Matriz Córners (0 a 20)
    c_h = stats.poisson.pmf(np.arange(0, 21), lambda_corners_h)
    c_a = stats.poisson.pmf(np.arange(0, 21), lambda_corners_a)
    mat_corners = np.outer(c_h, c_a)
    
    p_c_over_85 = np.sum([mat_corners[i, j] for i in range(21) for j in range(21) if i + j > 8.5])
    p_c_over_95 = np.sum([mat_corners[i, j] for i in range(21) for j in range(21) if i + j > 9.5])
    p_c_over_105 = np.sum([mat_corners[i, j] for i in range(21) for j in range(21) if i + j > 10.5])

    # Proyección Tarjetas
    lambda_cards = max(3.0, (lambda_h + mu_a) * 1.4 + 1.2)
    p_cards_over_35 = 1.0 - stats.poisson.cdf(3, lambda_cards)
    p_cards_over_45 = 1.0 - stats.poisson.cdf(4, lambda_cards)
    p_cards_over_55 = 1.0 - stats.poisson.cdf(5, lambda_cards)

    return {
        "exp_corners": total_corners_exp, "over_85_c": p_c_over_85, "over_95_c": p_c_over_95, "over_105_c": p_c_over_105,
        "exp_cards": lambda_cards, "over_35_t": p_cards_over_35, "over_45_t": p_cards_over_45, "over_55_t": p_cards_over_55
    }

def kelly_criterion(prob, cuota, fraction=0.25):
    """Criterio de Kelly Fraccionado para Manejo de Capital"""
    b = cuota - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f * fraction)

# ==============================================================================
# 4. BARRA LATERAL
# ==============================================================================
st.sidebar.markdown("### 🔑 API KEY")
odds_api_key = st.sidebar.text_input("The Odds API Key:", value="1a927e0f762a52540fe079d963ed2460", type="password")

liga_nombre = st.sidebar.selectbox("Torneo:", list(LIGAS_DISPONIBLES.keys()))
sport_key = LIGAS_DISPONIBLES[liga_nombre]

partidos_raw, error_api = obtener_partidos_odds_api(odds_api_key, sport_key)
if partidos_raw:
    opciones = [f"{m.get('home_team')} vs {m.get('away_team')}" for m in partidos_raw]
    idx = st.sidebar.selectbox("Evento:", range(len(opciones)), format_func=lambda x: opciones[x])
    p = extraer_mercados(partidos_raw[idx])
else:
    p = {"local": "LDU Quito", "visitante": "Mirassol", "c1": 1.95, "cx": 3.30, "c2": 3.80, "cover": 1.95, "cunder": 1.85, "ah_local": -0.25, "c_ah1": 1.95}

st.sidebar.markdown("---")
c_1 = st.sidebar.number_input(f"Cuota 1 ({p['local']}):", value=float(p['c1']), step=0.01)
c_x = st.sidebar.number_input("Cuota X (Empate):", value=float(p['cx']), step=0.01)
c_2 = st.sidebar.number_input(f"Cuota 2 ({p['visitante']}):", value=float(p['c2']), step=0.01)
c_over = st.sidebar.number_input("Cuota Over 2.5 Goles:", value=float(p['cover']), step=0.01)
c_under = st.sidebar.number_input("Cuota Under 2.5 Goles:", value=float(p['cunder']), step=0.01)
ah_line = st.sidebar.number_input("Línea Hándicap Asiático:", value=float(p['ah_local']), step=0.25)
c_ah1 = st.sidebar.number_input("Cuota Hándicap Local:", value=float(p['c_ah1']), step=0.01)

# ==============================================================================
# 5. EJECUCIÓN CÁLCULOS MATEMÁTICOS
# ==============================================================================
probs_shin, insider_z = estimar_shin([c_1, c_x, c_2])
total_xg = 2.60
lambda_h = (probs_shin[0] * total_xg * 1.08) / (probs_shin[0] + probs_shin[2] + 1e-5)
mu_a = (probs_shin[2] * total_xg) / (probs_shin[0] + probs_shin[2] + 1e-5)

p_dc_1, p_dc_x, p_dc_2, p_dc_over, p_dc_under, mat_dc = modelo_dixon_coles(lambda_h, mu_a)
p_ah_efectiva, p_ah_w, p_ah_hw, p_ah_push, p_ah_hl, p_ah_l = calcular_handicap_asiatico(mat_dc, ah_line)
aux = simular_corners_y_tarjetas(lambda_h, mu_a)

# Evaluación EV
ev_1 = (p_dc_1 * c_1 - 1.0) * 100
ev_x = (p_dc_x * c_x - 1.0) * 100
ev_2 = (p_dc_2 * c_2 - 1.0) * 100
ev_1x2 = max(ev_1, ev_x, ev_2)

ev_o = (p_dc_over * c_over - 1.0) * 100
ev_u = (p_dc_under * c_under - 1.0) * 100
ev_ou = max(ev_o, ev_u)

ev_handicap = (p_ah_efectiva * c_ah1 - 1.0) * 100
kelly_ah = kelly_criterion(p_ah_efectiva, c_ah1) * 100

# Catálogo completo de mercados evaluados
todos_los_mercados = [
    (f"Hándicap {ah_line} {p['local']}", p_ah_efectiva, ev_handicap, c_ah1),
    (f"Victoria Local ({p['local']})", p_dc_1, ev_1, c_1),
    ("Empate (X)", p_dc_x, ev_x, c_x),
    (f"Victoria Visitante ({p['visitante']})", p_dc_2, ev_2, c_2),
    ("Over 2.5 Goles", p_dc_over, ev_o, c_over),
    ("Under 2.5 Goles", p_dc_under, ev_u, c_under),
    ("Over 8.5 Córners", aux['over_85_c'], (aux['over_85_c'] * 1.55 - 1.0) * 100, 1.55),
    ("Over 9.5 Córners", aux['over_95_c'], (aux['over_95_c'] * 1.90 - 1.0) * 100, 1.90),
    ("Over 10.5 Córners", aux['over_105_c'], (aux['over_105_c'] * 2.30 - 1.0) * 100, 2.30),
    ("Over 3.5 Tarjetas", aux['over_35_t'], (aux['over_35_t'] * 1.45 - 1.0) * 100, 1.45),
    ("Over 4.5 Tarjetas", aux['over_45_t'], (aux['over_45_t'] * 1.85 - 1.0) * 100, 1.85),
    ("Over 5.5 Tarjetas", aux['over_55_t'], (aux['over_55_t'] * 2.40 - 1.0) * 100, 2.40)
]

pick_mayor_valor = max(todos_los_mercados, key=lambda x: x[2])
pick_mayor_prob = max(todos_los_mercados, key=lambda x: x[1])

# ==============================================================================
# 6. INTERFAZ GRÁFICA
# ==============================================================================
st.markdown(f"""
<div class="header-card">
    <div class="header-title">🏟️ {p['local']} vs {p['visitante']}</div>
    <div class="header-subtitle">Terminal de Inteligencia Cuantitativa</div>
</div>
""", unsafe_allow_html=True)

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

# Módulo 1: Orden (+EV)
if pick_mayor_valor[2] > 0:
    st.markdown(f"""
    <div class="box-ev">
        <span class="badge-ev">ORDEN APROBADA (+EV)</span>
        <div class="box-title">{pick_mayor_valor[0]}</div>
        <div class="box-desc">
            • Ventaja Esperada (+EV): <b>+{pick_mayor_valor[2]:.2f}%</b><br>
            • Cuota Recomendada: <b>{pick_mayor_valor[3]:.2f}</b> (Probabilidad Real: <b>{pick_mayor_valor[1]*100:.1f}%</b>)<br>
            • Asignación Kelly (0.25x): <b>{kelly_ah:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="box-ev" style="border-color: #dc2626; background-color: rgba(127,29,29,0.25);">
        <span class="badge-ev" style="background-color: #ef4444; color: #fff;">SIN VALOR DE APUESTA (-EV)</span>
        <div class="box-title">Mercados Bloqueados</div>
        <div class="box-desc">Las cuotas actuales de las casas de apuestas no ofrecen margen de ganancia matemática.</div>
    </div>
    """, unsafe_allow_html=True)

# Módulo 2: Elección por Mayor Probabilidad
st.markdown(f"""
<div class="box-prob">
    <span class="badge-prob">MAYOR PROBABILIDAD ESTADÍSTICA</span>
    <div class="box-title">{pick_mayor_prob[0]}</div>
    <div class="box-desc">
        • Probabilidad de Acierto Estimada: <b>{pick_mayor_prob[1]*100:.1f}%</b><br>
        • Valor Esperado (+EV) Asociado: <b>{pick_mayor_prob[2]:+.2f}%</b> | Cuota: <b>{pick_mayor_prob[3]:.2f}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Pestañas de Visualización Estática y Tablas Ampliadas
tab1, tab2, tab3 = st.tabs(["📊 Probabilidades", "🎯 Córners & Tarjetas", "📋 Evaluación Todos los Mercados"])

with tab1:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig_bar = go.Figure(data=[
            go.Bar(name='Modelo (Dixon-Coles)', x=['1', 'X', '2'], y=[p_dc_1*100, p_dc_x*100, p_dc_2*100], marker_color='#10b981'),
            go.Bar(name='Mercado (Shin)', x=['1', 'X', '2'], y=[probs_shin[0]*100, probs_shin[1]*100, probs_shin[2]*100], marker_color='#3b82f6')
        ])
        fig_bar.update_layout(
            title="Comparativa 1X2 (%)",
            barmode='group', height=220,
            margin=dict(l=5, r=5, t=30, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af', size=10)
        )
        fig_bar.update_xaxes(fixedrange=True)
        fig_bar.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG_STATIC)

    with col_f2:
        x_g = np.linspace(0, 6, 100)
        y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
        fig_density = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#10b981')))
        fig_density.add_vline(x=2.5, line_dash="dash", line_color="#ef4444")
        fig_density.update_layout(
            title="Distribución de Goles Esperados", height=220,
            margin=dict(l=5, r=5, t=30, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af', size=10)
        )
        fig_density.update_xaxes(fixedrange=True)
        fig_density.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_density, use_container_width=True, config=PLOTLY_CONFIG_STATIC)

with tab2:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"**Tiros de Esquina Esperados:** `{aux['exp_corners']:.1f}`")
        st.caption(f"• Over 8.5 Córners: **{aux['over_85_c']*100:.1f}%**")
        st.caption(f"• Over 9.5 Córners: **{aux['over_95_c']*100:.1f}%**")
        st.caption(f"• Over 10.5 Córners: **{aux['over_105_c']*100:.1f}%**")
    with col_c2:
        st.markdown(f"**Tarjetas Esperadas:** `{aux['exp_cards']:.1f}`")
        st.caption(f"• Over 3.5 Tarjetas: **{aux['over_35_t']*100:.1f}%**")
        st.caption(f"• Over 4.5 Tarjetas: **{aux['over_45_t']*100:.1f}%**")
        st.caption(f"• Over 5.5 Tarjetas: **{aux['over_55_t']*100:.1f}%**")

with tab3:
    data_tabla = {
        "Mercado Evaluado": [m[0] for m in todos_los_mercados],
        "Probabilidad Real": [f"{m[1]*100:.1f}%" for m in todos_los_mercados],
        "Valor Esperado (+EV)": [f"{m[2]:+.2f}%" for m in todos_los_mercados],
        "Cuota Referencia": [f"{m[3]:.2f}" for m in todos_los_mercados]
    }
    st.dataframe(data_tabla, use_container_width=True)
