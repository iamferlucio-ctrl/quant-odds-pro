import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# Configuración Plotly para deshabilitar zoom y barra flotante en móviles
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'scrollZoom': False,
    'doubleClick': 'reset'
}

st.set_page_config(
    page_title="Terminal Cuantitativo v3.6",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #c9d1d9; font-family: monospace; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
    .reasoning-box { background-color: #0d1117; border-left: 4px solid #1f6feb; padding: 14px; font-size: 0.88em; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MOTOR DE EXTRACCIÓN CON SANITIZACIÓN Y CONTROL DE ERRORES
# ==============================================================================
def extraer_partidos_jornada(liga, api_key):
    key_limpia = api_key.strip() if api_key else ""
    if not key_limpia:
        st.sidebar.warning("⚠️ Ingrese una Gemini API Key válida.")
        return None
    
    try:
        genai.configure(api_key=key_limpia)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Proporciona un objeto JSON puro con partidos de la jornada actual para: '{liga}'.
        Estructura requerida:
        {{
            "partidos": [
                {{
                    "local": "Macará",
                    "visitante": "Santos",
                    "cuota_1": 2.33,
                    "cuota_x": 3.40,
                    "cuota_2": 2.90,
                    "cuota_over": 2.10,
                    "cuota_under": 1.70
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_clean)
        return data.get("partidos", [])
    except Exception as e:
        if "API_KEY_INVALID" in str(e) or "400" in str(e):
            st.sidebar.error("❌ API Key no válida. Verifique la clave en Google AI Studio.")
        else:
            st.sidebar.error(f"❌ Error de conexión: {str(e)}")
        return None

# ==============================================================================
# 2. MOTOR CUANTITATIVO (SHIN & POISSON)
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
# 3. INTERFAZ Y BARRA LATERAL
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & LIGA")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")
liga_activa = st.sidebar.selectbox("Seleccionar Liga:", ["Copa Libertadores", "LigaPro Ecuador", "Premier League", "Serie A Brasil"])

if st.sidebar.button("📡 Cargar Partidos de la Jornada (IA)", use_container_width=True):
    with st.sidebar.spinner("Consultando fixture..."):
        partidos = extraer_partidos_jornada(liga_activa, gemini_key)
        if partidos:
            st.session_state['partidos'] = partidos
            st.sidebar.success(f"¡{len(partidos)} partidos cargados!")

partidos_disponibles = st.session_state.get('partidos', [
    {"local": "Macará", "visitante": "Santos", "cuota_1": 2.33, "cuota_x": 3.40, "cuota_2": 2.90, "cuota_over": 2.10, "cuota_under": 1.70}
])

opciones = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_idx = st.sidebar.selectbox("Evento de la Jornada:", range(len(opciones)), format_func=lambda x: opciones[x])
p_data = partidos_disponibles[partido_idx]

st.sidebar.markdown("---")
c_1 = st.sidebar.number_input(f"Cuota {p_data['local']}:", value=float(p_data['cuota_1']))
c_x = st.sidebar.number_input("Cuota Empate:", value=float(p_data['cuota_x']))
c_2 = st.sidebar.number_input(f"Cuota {p_data['visitante']}:", value=float(p_data['cuota_2']))
c_over = st.sidebar.number_input("Cuota Más 2.5:", value=float(p_data['cuota_over']))
c_under = st.sidebar.number_input("Cuota Menos 2.5:", value=float(p_data['cuota_under']))

# ==============================================================================
# 4. DASHBOARD CON EJES FIJOS (FIXED RANGE)
# ==============================================================================
probs_1x2, ovr_1x2 = desmarginado_shin([c_1, c_x, c_2])
probs_ou, ovr_ou = desmarginado_shin([c_over, c_under])

total_xg = 2.85 if "Libertadores" in liga_activa else 2.40
lambda_h = (probs_1x2[0] * total_xg * 1.1) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
mu_a = (probs_1x2[2] * total_xg) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
p_over_mod, p_under_mod, _ = estimar_matriz_poisson(lambda_h, mu_a)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

st.markdown(f"<h2>QG | TERMINAL DE OBJETIVOS CUÁNTICOS</h2>", unsafe_allow_html=True)
st.markdown(f"**Evento Seleccionado:** `{p_data['local']} vs {p_data['visitante']}` | **Liga:** `{liga_activa}`")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**DENSIDAD ESPECTRAL DE GOLES**")
    x_g = np.linspace(0, 6, 100)
    y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
    
    fig1 = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#1f6feb')))
    fig1.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72")
    
    # Bloqueo estricto del zoom para interfaz móvil limpia
    fig1.update_xaxes(range=[0, 6], fixedrange=True)
    fig1.update_yaxes(fixedrange=True)
    fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

with col_g2:
    st.markdown("**CINÉTICA EN JUEGO (DECAIMIENTO)**")
    fig2 = go.Figure(go.Scatter(x=[0, 15, 30, 45, 60, 75, 90], y=[lambda_h+mu_a, (lambda_h+mu_a)*0.8, (lambda_h+mu_a)*0.6, (lambda_h+mu_a)*0.4, (lambda_h+mu_a)*0.2, (lambda_h+mu_a)*0.1, 0], mode='lines+markers', line=dict(color='#3fb950')))
    fig2.update_xaxes(fixedrange=True)
    fig2.update_yaxes(fixedrange=True)
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown(f"""
<div class="metric-card">
    <div style="display:flex; justify-shadow:space-around; text-align:center;">
        <div style="flex:1;"><small>EV OVER 2.5</small><br><b style="color:{'#3fb950' if ev_over > 0 else '#ff7b72'};">{ev_over*100:+.2f}%</b></div>
        <div style="flex:1;"><small>EV UNDER 2.5</small><br><b style="color:{'#3fb950' if ev_under > 0 else '#ff7b72'};">{ev_under*100:+.2f}%</b></div>
        <div style="flex:1;"><small>EXPECTATIVA xG</small><br><b style="color:#58a6ff;">{(lambda_h + mu_a):.2f} Goles</b></div>
    </div>
</div>
""", unsafe_allow_html=True)
