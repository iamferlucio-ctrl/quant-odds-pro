import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# Configuración Plotly: Bloqueo de zoom flotante para móviles
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'scrollZoom': False,
    'doubleClick': 'reset'
}

st.set_page_config(
    page_title="Terminal Cuantitativo & Extractor IA v3.7",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
    .reasoning-box { background-color: #0d1117; border-left: 4px solid #1f6feb; padding: 14px; font-size: 0.88em; color: #8b949e; }
    .badge-clean { background-color: rgba(46,160,67,0.15); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); padding: 3px 8px; border-radius: 10px; font-size: 0.75em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CATÁLOGO COMPLETO DE LIGAS Y BANCO DE DATOS DE RESPALDO (FALLBACK)
# ==============================================================================
BASE_DATOS_LIGAS = {
    "LigaPro Ecuador": {
        "xg_base": 2.35,
        "partidos_default": [
            {"local": "Macará", "visitante": "Santos", "cuota_1": 2.33, "cuota_x": 3.40, "cuota_2": 2.90, "cuota_over": 2.10, "cuota_under": 1.70},
            {"local": "LDU Quito", "visitante": "Barcelona SC", "cuota_1": 1.95, "cuota_x": 3.30, "cuota_2": 3.80, "cuota_over": 1.85, "cuota_under": 1.95},
            {"local": "Independiente del Valle", "visitante": "Emelec", "cuota_1": 1.70, "cuota_x": 3.60, "cuota_2": 4.80, "cuota_over": 1.75, "cuota_under": 2.05}
        ]
    },
    "Copa Libertadores": {
        "xg_base": 2.85,
        "partidos_default": [
            {"local": "Flamengo", "visitante": "River Plate", "cuota_1": 2.05, "cuota_x": 3.25, "cuota_2": 3.60, "cuota_over": 1.90, "cuota_under": 1.90},
            {"local": "Palmeiras", "visitante": "LDU Quito", "cuota_1": 1.50, "cuota_x": 4.00, "cuota_2": 6.50, "cuota_over": 1.70, "cuota_under": 2.10}
        ]
    },
    "Premier League": {
        "xg_base": 2.82,
        "partidos_default": [
            {"local": "Arsenal", "visitante": "Manchester City", "cuota_1": 2.50, "cuota_x": 3.40, "cuota_2": 2.75, "cuota_over": 1.80, "cuota_under": 2.00},
            {"local": "Liverpool", "visitante": "Chelsea", "cuota_1": 1.85, "cuota_x": 3.75, "cuota_2": 3.90, "cuota_over": 1.60, "cuota_under": 2.30}
        ]
    },
    "Serie A Brasil": {
        "xg_base": 2.40,
        "partidos_default": [
            {"local": "Botafogo", "visitante": "Sao Paulo", "cuota_1": 2.10, "cuota_x": 3.10, "cuota_2": 3.50, "cuota_over": 2.05, "cuota_under": 1.75}
        ]
    },
    "LaLiga España": {
        "xg_base": 2.55,
        "partidos_default": [
            {"local": "Real Madrid", "visitante": "Barcelona", "cuota_1": 2.15, "cuota_x": 3.50, "cuota_2": 3.10, "cuota_over": 1.65, "cuota_under": 2.20}
        ]
    },
    "UEFA Champions League": {
        "xg_base": 2.90,
        "partidos_default": [
            {"local": "Real Madrid", "visitante": "Bayern Múnich", "cuota_1": 2.20, "cuota_x": 3.60, "cuota_2": 3.00, "cuota_over": 1.60, "cuota_under": 2.30}
        ]
    }
}

# ==============================================================================
# 2. EXTRACCIÓN DINÁMICA CON SANITIZACIÓN DE CLAVE
# ==============================================================================
def extraer_partidos_ia(liga, api_key_input):
    key_limpia = api_key_input.strip().strip('"').strip("'") if api_key_input else ""
    if not key_limpia:
        st.sidebar.warning("⚠️ Ingresa una Gemini API Key válida para consultar en vivo.")
        return None
    
    try:
        genai.configure(api_key=key_limpia)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Devuelve EXCLUSIVAMENTE un objeto JSON puro con los partidos/cuotas principales de la jornada para: '{liga}'.
        No agregues explicaciones ni marcas de formato markdown.
        Estructura:
        {{
            "partidos": [
                {{
                    "local": "Equipo A",
                    "visitante": "Equipo B",
                    "cuota_1": 2.10,
                    "cuota_x": 3.20,
                    "cuota_2": 3.50,
                    "cuota_over": 1.90,
                    "cuota_under": 1.90
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_clean)
        return data.get("partidos", [])
    except Exception as e:
        st.sidebar.error("❌ Llave API rechazada por Google. Se mantendrán los partidos precargados de la liga.")
        return None

# ==============================================================================
# 3. MOTOR CUANTITATIVO (SHIN & POISSON)
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
# 4. BARRA LATERAL E INTERFAZ
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & SELECCIÓN DE LIGA")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")

liga_activa = st.sidebar.selectbox("Seleccionar Liga Target:", list(BASE_DATOS_LIGAS.keys()))
config_liga_actual = BASE_DATOS_LIGAS[liga_activa]

# Botón de consulta IA
if st.sidebar.button("📡 Cargar Jornada en Vivo (IA)", use_container_width=True):
    with st.sidebar.spinner("Buscando partidos..."):
        partidos_ia = extraer_partidos_ia(liga_activa, gemini_key)
        if partidos_ia:
            st.session_state['partidos_activos'] = partidos_ia
            st.sidebar.success(f"¡{len(partidos_ia)} partidos actualizados por IA!")

# Cargar partidos (IA o Respaldo)
partidos_disponibles = st.session_state.get('partidos_activos', config_liga_actual["partidos_default"])

# Si se cambia de liga y no hay caché, cargar los predeterminados de la liga seleccionada
if not st.session_state.get('partidos_activos'):
    partidos_disponibles = config_liga_actual["partidos_default"]

opciones = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_idx = st.sidebar.selectbox("Evento de la Jornada:", range(len(opciones)), format_func=lambda x: opciones[x])
p_data = partidos_disponibles[partido_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AJUSTE DE CUOTAS")
c_1 = st.sidebar.number_input(f"Cuota {p_data['local']}:", value=float(p_data['cuota_1']), step=0.01)
c_x = st.sidebar.number_input("Cuota Empate:", value=float(p_data['cuota_x']), step=0.01)
c_2 = st.sidebar.number_input(f"Cuota {p_data['visitante']}:", value=float(p_data['cuota_2']), step=0.01)
c_over = st.sidebar.number_input("Cuota Más 2.5:", value=float(p_data['cuota_over']), step=0.01)
c_under = st.sidebar.number_input("Cuota Menos 2.5:", value=float(p_data['cuota_under']), step=0.01)

# ==============================================================================
# 5. DASHBOARD CUANTITATIVO CON GRAFICACIÓN ROBUSTA
# ==============================================================================
probs_1x2, ovr_1x2 = desmarginado_shin([c_1, c_x, c_2])
probs_ou, ovr_ou = desmarginado_shin([c_over, c_under])

total_xg = config_liga_actual["xg_base"]
lambda_h = (probs_1x2[0] * total_xg * 1.1) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
mu_a = (probs_1x2[2] * total_xg) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
p_over_mod, p_under_mod, _ = estimar_matriz_poisson(lambda_h, mu_a)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

st.markdown("<h2>QG | TERMINAL DE OBJETIVOS CUÁNTICOS</h2>", unsafe_allow_html=True)
st.markdown(f"**Evento Activo:** `{p_data['local']} vs {p_data['visitante']}` | **Competición:** `{liga_activa}`")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**DENSIDAD ESPECTRAL DE GOLES**")
    x_g = np.linspace(0, 6, 100)
    y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
    
    fig1 = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#1f6feb')))
    fig1.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72")
    fig1.update_xaxes(range=[0, 6], fixedrange=True)
    fig1.update_yaxes(fixedrange=True)
    fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

with col_g2:
    st.markdown("**CINÉTICA EN JUEGO (DECAIMIENTO TEMPORAL)**")
    fig2 = go.Figure(go.Scatter(x=[0, 15, 30, 45, 60, 75, 90], y=[lambda_h+mu_a, (lambda_h+mu_a)*0.8, (lambda_h+mu_a)*0.6, (lambda_h+mu_a)*0.4, (lambda_h+mu_a)*0.2, (lambda_h+mu_a)*0.1, 0], mode='lines+markers', line=dict(color='#3fb950')))
    fig2.update_xaxes(fixedrange=True)
    fig2.update_yaxes(fixedrange=True)
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=180)
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

# Cinta de Métricas EV
st.markdown(f"""
<div class="metric-card">
    <div style="display:flex; justify-content:space-around; text-align:center;">
        <div style="flex:1;"><small>EV OVER 2.5</small><br><b style="color:{'#3fb950' if ev_over > 0 else '#ff7b72'}; font-size:1.1em;">{ev_over*100:+.2f}%</b></div>
        <div style="flex:1;"><small>EV UNDER 2.5</small><br><b style="color:{'#3fb950' if ev_under > 0 else '#ff7b72'}; font-size:1.1em;">{ev_under*100:+.2f}%</b></div>
        <div style="flex:1;"><small>EXPECTATIVA xG LIGA</small><br><b style="color:#58a6ff; font-size:1.1em;">{(lambda_h + mu_a):.2f} Goles</b></div>
    </div>
</div>
""", unsafe_allow_html=True)
