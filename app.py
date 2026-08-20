import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import requests

# ==============================================================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Terminal Quant v13.0 - Value Detector",
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
# 2. MOTOR DE CÁLCULO MEJORADO
# ==============================================================================
def estimar_shin(cuotas_1x2):
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
    lambda_c_h, lambda_c_a = max(2.5, lambda_h * 3.1), max(1.8, mu_a * 2.7)
    tot_corners = lambda_c_h + lambda_c_a
    c_h, c_a = stats.poisson.pmf(np.arange(0, 21), lambda_c_h), stats.poisson.pmf(np.arange(0, 21), lambda_c_a)
    mat_c = np.outer(c_h, c_a)
    
    p_c_85 = np.sum([mat_c[i, j] for i in range(21) for j in range(21) if i + j > 8.5])
    p_c_95 = np.sum([mat_c[i, j] for i in range(21) for j in range(21) if i + j > 9.5])
    
    lambda_cards = max(3.0, (lambda_h + mu_a) * 1.4 + 1.2)
    p_t_35 = 1.0 - stats.poisson.cdf(3, lambda_cards)
    p_t_45 = 1.0 - stats.poisson.cdf(4, lambda_cards)

    return tot_corners, p_c_85, p_c_95, lambda_cards, p_t_35, p_t_45

def kelly_criterion(prob, cuota, fraction=0.25):
    b = cuota - 1.0
    if b <= 0: return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f * fraction)

# ==============================================================================
# 3. ENTRADA DE DATOS (BARRA LATERAL)
# ==============================================================================
st.sidebar.markdown("### ⚙️ PARÁMETROS DEL EVENTO")
p_local = st.sidebar.text_input("Local:", "Macará")
p_visita = st.sidebar.text_input("Visitante:", "Santos")

c_1 = st.sidebar.number_input(f"Cuota 1 ({p_local}):", value=2.20, step=0.01)
c_x = st.sidebar.number_input("Cuota X (Empate):", value=3.10, step=0.01)
c_2 = st.sidebar.number_input(f"Cuota 2 ({p_visita}):", value=3.00, step=0.01)
c_over = st.sidebar.number_input("Cuota Over 2.5 Goles:", value=1.95, step=0.01)
c_under = st.sidebar.number_input("Cuota Under 2.5 Goles:", value=1.85, step=0.01)
ah_line = st.sidebar.number_input("Línea Hándicap Asiático:", value=-0.25, step=0.25)
c_ah1 = st.sidebar.number_input("Cuota Hándicap Local:", value=1.90, step=0.01)

# ==============================================================================
# 4. EJECUCIÓN MATEMÁTICA Y IDENTIFICACIÓN DE INFRAVALORADOS
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

# Catálogo ampliado con Cálculo de Discrepancia (\Delta Prob) y Cuota Fair
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
# 5. DESPLIEGUE EN PANTALLA
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

# Módulo +EV Corregido
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

# Pestañas
tab1, tab2 = st.tabs(["📊 Gráficos", "🔍 Detección de Ineficiencias"])

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
