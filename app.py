import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# ==============================================================================
# 1. CONFIGURACIÓN Y ESTILOS INSTITUCIONALES
# ==============================================================================
st.set_page_config(
    page_title="Terminal Cuantitativo & Extractor IA v3.5",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #c9d1d9; font-family: monospace; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
    .reasoning-box { background-color: #0d1117; border-left: 4px solid #1f6feb; padding: 14px; font-size: 0.88em; color: #8b949e; }
    .badge-high { background-color: rgba(248,81,73,0.15); color: #ff7b72; border: 1px solid rgba(248,81,73,0.4); padding: 3px 8px; border-radius: 10px; font-size: 0.75em; font-weight: bold; }
    .badge-clean { background-color: rgba(46,160,67,0.15); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); padding: 3px 8px; border-radius: 10px; font-size: 0.75em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MOTOR DE EXTRACCIÓN CON IA (GEMINI API)
# ==============================================================================
def extraer_partidos_jornada(liga, api_key):
    """Utiliza Gemini para obtener o estructurar los partidos y cuotas clave de la jornada."""
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Actúa como un proveedor de datos de fútbol. Proporciona una lista JSON con 3 a 5 partidos destacados o de la jornada actual para la competición: '{liga}'.
        Devuelve EXCLUSIVAMENTE un objeto JSON válido con la siguiente estructura (sin bloques de código markdown extra):
        {{
            "partidos": [
                {{
                    "local": "Nombre Local",
                    "visitante": "Nombre Visitante",
                    "cuota_1": 2.33,
                    "cuota_x": 3.40,
                    "cuota_2": 2.90,
                    "cuota_over": 2.10,
                    "cuota_under": 1.70,
                    "apertura_over": 1.80,
                    "pct_publico_over": 74
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_clean)
        return data.get("partidos", [])
    except Exception as e:
        st.sidebar.error(f"Error extrayendo jornada: {str(e)}")
        return None

# ==============================================================================
# 3. MOTOR CUANTITATIVO (SHIN, POISSON & EV)
# ==============================================================================
def desmarginado_shin(cuotas):
    inv_c = np.array([1.0 / q if q > 1.0 else 0.0 for q in cuotas])
    if np.any(inv_c == 0.0): return np.zeros_like(cuotas), 0.0
    overround = np.sum(inv_c) - 1.0
    p_raw = inv_c / (1.0 + overround)
    return p_raw / np.sum(p_raw), overround

def estimar_matriz_poisson(lambda_h, mu_a, max_goles=6):
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            mat[i, j] = stats.poisson.pmf(i, lambda_h) * stats.poisson.pmf(j, mu_a)
    p_over = np.sum([mat[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    return p_over, 1.0 - p_over, mat

# ==============================================================================
# 4. BARRA LATERAL (GESTIÓN DINÁMICA DE JORNADA)
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & EXTRACCIÓN DINÁMICA")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")

liga_activa = st.sidebar.selectbox("Seleccionar Liga:", ["LigaPro Ecuador", "Premier League", "Copa Libertadores", "Serie A Brasil"])

# Botón de Auto-Extracción mediante API Key
partidos_cargados = []
if st.sidebar.button("📡 Cargar Partidos de la Jornada (IA)", use_container_width=True):
    with st.sidebar.spinner("Extrayendo fixture y cuotas..."):
        partidos_cargados = extraer_partidos_jornada(liga_activa, gemini_key)
        if partidos_cargados:
            st.session_state['partidos'] = partidos_cargados
            st.sidebar.success(f"¡{len(partidos_cargados)} partidos cargados!")

# Selección del partido de la jornada
partidos_disponibles = st.session_state.get('partidos', [
    {"local": "Macará", "visitante": "Santos", "cuota_1": 2.33, "cuota_x": 3.40, "cuota_2": 2.90, "cuota_over": 2.10, "cuota_under": 1.70, "apertura_over": 1.80, "pct_publico_over": 74}
])

opciones_partidos = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_sel_idx = st.sidebar.selectbox("Seleccionar Evento de la Jornada:", range(len(opciones_partidos)), format_func=lambda x: opciones_partidos[x])
p_data = partidos_disponibles[partido_sel_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AJUSTE DE CUOTAS Y MERCADO")
c_1 = st.sidebar.number_input(f"Cuota {p_data['local']}:", value=float(p_data['cuota_1']))
c_x = st.sidebar.number_input("Cuota Empate:", value=float(p_data['cuota_x']))
c_2 = st.sidebar.number_input(f"Cuota {p_data['visitante']}:", value=float(p_data['cuota_2']))
c_over = st.sidebar.number_input("Cuota Más 2.5:", value=float(p_data['cuota_over']))
c_under = st.sidebar.number_input("Cuota Menos 2.5:", value=float(p_data['cuota_under']))

# ==============================================================================
# 5. CÁLCULOS Y PANEL PRINCIPAL
# ==============================================================================
probs_1x2, ovr_1x2 = desmarginado_shin([c_1, c_x, c_2])
probs_ou, ovr_ou = desmarginado_shin([c_over, c_under])

# Expectativa de goles
total_xg = 2.40 if "Ecuador" in liga_activa else 2.70
lambda_h = (probs_1x2[0] * total_xg * 1.1) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
mu_a = (probs_1x2[2] * total_xg) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
p_over_mod, p_under_mod, _ = estimar_matriz_poisson(lambda_h, mu_a)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

st.markdown(f"<h2>QG | TERMINAL DE OBJETIVOS CUÁNTICOS <span style='font-size:0.5em; color:#3fb950;'>v3.5 HÍBRIDA</span></h2>", unsafe_allow_html=True)
st.markdown(f"**Evento Seleccionado:** `{p_data['local']} vs {p_data['visitante']}` | **Liga:** `{liga_activa}`")

# Gráficos de Espectro y Cinética
col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("**DENSIDAD ESPECTRAL DE GOLES**")
    x_g = np.linspace(0, 6, 100)
    y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
    fig = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#1f6feb')))
    fig.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    st.plotly_chart(fig, use_container_width=True)

with col_g2:
    st.markdown("**CINÉTICA EN JUEGO (DECAIMIENTO)**")
    fig2 = go.Figure(go.Scatter(x=[0, 15, 30, 45, 60, 75, 90], y=[lambda_h+mu_a, (lambda_h+mu_a)*0.8, (lambda_h+mu_a)*0.6, (lambda_h+mu_a)*0.4, (lambda_h+mu_a)*0.2, (lambda_h+mu_a)*0.1, 0], mode='lines+markers', line=dict(color='#3fb950')))
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    st.plotly_chart(fig2, use_container_width=True)

# Cinta EV
st.markdown(f"""
<div class="metric-card">
    <div style="display:flex; justify-content:space-around; text-align:center;">
        <div><small>EV OVER 2.5</small><br><b style="color:{'#3fb950' if ev_over > 0 else '#ff7b72'};">{ev_over*100:+.2f}%</b></div>
        <div><small>EV UNDER 2.5</small><br><b style="color:{'#3fb950' if ev_under > 0 else '#ff7b72'};">{ev_under*100:+.2f}%</b></div>
        <div><small>EXPECTATIVA xG</small><br><b style="color:#58a6ff;">{(lambda_h + mu_a):.2f} Goles</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Razonamiento Numérico
st.markdown(f"""
<div class="reasoning-box">
1. Extracción de fixture de <b>{liga_activa}</b> ejecutada para {p_data['local']} vs {p_data['visitante']}.<br>
2. Expectativa xG calculada: Local <b>&lambda;={lambda_h:.2f}</b>, Visitante <b>&mu;={mu_a:.2f}</b>.<br>
3. Probabilidad Modelo Over 2.5: <b>{p_over_mod*100:.1f}%</b> | Probabilidad Mercado Limpia (Shin): <b>{probs_ou[0]*100:.1f}%</b>.<br>
4. Estado del Valor: <b>{'VENTAJA +EV DETECTADA' if max(ev_over, ev_under) > 0 else 'NEUTRALIDAD / SIN EV'}</b>.
</div>
""", unsafe_allow_html=True)
