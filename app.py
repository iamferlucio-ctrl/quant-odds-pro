import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# ==============================================================================
# 1. CONFIGURACIÓN MÓVIL Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Terminal Cuantitativo v5.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PLOTLY_CONFIG = {
    'displayModeBar': False,
    'scrollZoom': False,
    'doubleClick': False,
    'showAxisDragHandles': False
}

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #c9d1d9; font-family: -apple-system, sans-serif; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .reasoning-box { background-color: #0d1117; border-left: 3px solid #1f6feb; padding: 10px; font-size: 0.85em; color: #8b949e; border-radius: 0 6px 6px 0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CATÁLOGO GLOBAL DE LIGAS Y DATOS BASE
# ==============================================================================
LIGAS_PREDEFINIDAS = [
    "LigaPro Ecuador",
    "Copa Libertadores",
    "Copa Sudamericana",
    "Premier League (Inglaterra)",
    "LaLiga (España)",
    "Serie A (Italia)",
    "Bundesliga (Alemania)",
    "Ligue 1 (Francia)",
    "Serie A (Brasil)",
    "Primera División (Argentina)",
    "UEFA Champions League",
    "✍️ Otra Liga / Competición..."
]

PARTIDOS_DEFECTO = [
    {"local": "Macará", "visitante": "Santos", "cuota_1": 2.33, "cuota_x": 3.40, "cuota_2": 2.90, "cuota_over": 2.10, "cuota_under": 1.70},
    {"local": "LDU Quito", "visitante": "Barcelona SC", "cuota_1": 1.95, "cuota_x": 3.30, "cuota_2": 3.80, "cuota_over": 1.85, "cuota_under": 1.95}
]

# ==============================================================================
# 3. EXTRACCIÓN DINÁMICA UNIVERSAL (CUALQUIER LIGA VÍA GEMINI)
# ==============================================================================
def extraer_partidos_universal(nombre_liga, api_key_input):
    key_limpia = api_key_input.strip().strip('"').strip("'") if api_key_input else ""
    if not key_limpia:
        return None, "Ingresa tu Gemini API Key para extraer datos en vivo."
    
    try:
        genai.configure(api_key=key_limpia)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Actúa como un proveedor de datos de fútbol. Proporciona los partidos de la jornada o mas recientes para la liga/competición: '{nombre_liga}'.
        Devuelve EXCLUSIVAMENTE un objeto JSON puro con este formato exacto:
        {{
            "partidos": [
                {{
                    "local": "Nombre Equipo A",
                    "visitante": "Nombre Equipo B",
                    "cuota_1": 2.10,
                    "cuota_x": 3.20,
                    "cuota_2": 3.40,
                    "cuota_over": 1.90,
                    "cuota_under": 1.90
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        text_clean = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_clean)
        return data.get("partidos", []), None
    except Exception as e:
        return None, f"Error extrayendo '{nombre_liga}': {str(e)}"

# ==============================================================================
# 4. CÁLCULOS MATEMÁTICOS (SHIN & POISSON)
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
    return p_over, 1.0 - p_over

# ==============================================================================
# 5. CONTROLES Y SELECCIÓN EN BARRA LATERAL
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & LIGA")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")

liga_sel = st.sidebar.selectbox("Seleccionar Liga:", LIGAS_PREDEFINIDAS)

if liga_sel == "✍️ Otra Liga / Competición...":
    liga_consulta = st.sidebar.text_input("Escribe el nombre de la liga:", value="MLS Estados Unidos")
else:
    liga_consulta = liga_sel

if st.sidebar.button("📡 Cargar Jornada con IA", use_container_width=True):
    with st.sidebar.spinner(f"Extrayendo partidos de {liga_consulta}..."):
        partidos_ia, err = extraer_partidos_universal(liga_consulta, gemini_key)
        if partidos_ia:
            st.session_state['partidos_activos'] = partidos_ia
            st.sidebar.success(f"¡{len(partidos_ia)} partidos cargados!")
        else:
            st.sidebar.error(err)

partidos_disponibles = st.session_state.get('partidos_activos', PARTIDOS_DEFECTO)
opciones = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_idx = st.sidebar.selectbox("Seleccionar Partido:", range(len(opciones)), format_func=lambda x: opciones[x])
p_data = partidos_disponibles[partido_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ CUOTAS DEL MERCADO")
c_1 = st.sidebar.number_input(f"Cuota {p_data['local']}:", value=float(p_data['cuota_1']), step=0.01)
c_x = st.sidebar.number_input("Cuota Empate:", value=float(p_data['cuota_x']), step=0.01)
c_2 = st.sidebar.number_input(f"Cuota {p_data['visitante']}:", value=float(p_data['cuota_2']), step=0.01)
c_over = st.sidebar.number_input("Cuota Más 2.5:", value=float(p_data['cuota_over']), step=0.01)
c_under = st.sidebar.number_input("Cuota Menos 2.5:", value=float(p_data['cuota_under']), step=0.01)

# ==============================================================================
# 6. PANEL PRINCIPAL
# ==============================================================================
probs_1x2, _ = desmarginado_shin([c_1, c_x, c_2])
probs_ou, _ = desmarginado_shin([c_over, c_under])

total_xg = 2.65
lambda_h = (probs_1x2[0] * total_xg * 1.1) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
mu_a = (probs_1x2[2] * total_xg) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
p_over_mod, p_under_mod = estimar_matriz_poisson(lambda_h, mu_a)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

st.markdown(f"### ⚡ Terminal QG | `{p_data['local']} vs {p_data['visitante']}`")
st.caption(f"Competición: **{liga_consulta}** | Expectativa total: **{(lambda_h + mu_a):.2f} xG**")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;">
        <small>EV OVER 2.5</small><br>
        <strong style="font-size:1.4em; color:{'#3fb950' if ev_over > 0 else '#ff7b72'};">{ev_over*100:+.2f}%</strong>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;">
        <small>EV UNDER 2.5</small><br>
        <strong style="font-size:1.4em; color:{'#3fb950' if ev_under > 0 else '#ff7b72'};">{ev_under*100:+.2f}%</strong>
    </div>
    """, unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("<small><b>DENSIDAD DE GOLES</b></small>", unsafe_allow_html=True)
    x_g = np.linspace(0, 6, 100)
    y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
    fig1 = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#1f6feb')))
    fig1.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72")
    fig1.update_xaxes(range=[0, 6], fixedrange=True, showgrid=False)
    fig1.update_yaxes(fixedrange=True, showgrid=False)
    fig1.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=150)
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

with col_g2:
    st.markdown("<small><b>CINÉTICA TEMPORAL</b></small>", unsafe_allow_html=True)
    fig2 = go.Figure(go.Scatter(x=[0, 15, 30, 45, 60, 75, 90], y=[lambda_h+mu_a, (lambda_h+mu_a)*0.8, (lambda_h+mu_a)*0.6, (lambda_h+mu_a)*0.4, (lambda_h+mu_a)*0.2, (lambda_h+mu_a)*0.1, 0], mode='lines+markers', line=dict(color='#3fb950')))
    fig2.update_xaxes(fixedrange=True, showgrid=False)
    fig2.update_yaxes(fixedrange=True, showgrid=False)
    fig2.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=150)
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown(f"""
<div class="reasoning-box">
• <b>Modelo:</b> Over 2.5 ({p_over_mod*100:.1f}%) | Under 2.5 ({p_under_mod*100:.1f}%).<br>
• <b>Mercado (Shin):</b> Over 2.5 ({probs_ou[0]*100:.1f}%) | Under 2.5 ({probs_ou[1]*100:.1f}%).<br>
• <b>Resultado:</b> {'VENTAJA DETECTADA' if max(ev_over, ev_under) > 0 else 'NEUTRALIDAD / SIN VALOR EV'}.
</div>
""", unsafe_allow_html=True)
