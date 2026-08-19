import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# ==============================================================================
# 1. CONFIGURACIÓN Y ESTILOS MÓVILES AVANZADOS (FLEXBOX RESPONSIVE)
# ==============================================================================
st.set_page_config(
    page_title="Terminal QG Pro v6.0",
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
    .stApp { background-color: #0b0f19; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    
    /* Flexbox horizontal forzado en celulares */
    .mobile-flex-container {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        gap: 8px !important;
        margin-bottom: 12px !important;
    }
    .mobile-card {
        flex: 1 !important;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .mobile-card small { font-size: 0.72em; color: #8b949e; font-weight: bold; }
    .mobile-card div { font-size: 1.25em; font-weight: 800; margin-top: 2px; }
    
    /* Cajas de resumen y tablas */
    .reasoning-box { background-color: #0d1117; border-left: 4px solid #1f6feb; padding: 10px 12px; font-size: 0.83em; color: #8b949e; border-radius: 0 6px 6px 0; margin-top: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 6px; padding: 6px 12px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CATÁLOGO GLOBAL DE LIGAS
# ==============================================================================
CATALOGO_LIGAS = {
    "LigaPro Ecuador": {
        "xg_base": 2.35,
        "partidos": [
            {"local": "Macará", "visitante": "Santos", "cuota_1": 2.33, "cuota_x": 3.40, "cuota_2": 2.90, "cuota_over": 2.10, "cuota_under": 1.70},
            {"local": "LDU Quito", "visitante": "Barcelona SC", "cuota_1": 1.95, "cuota_x": 3.30, "cuota_2": 3.80, "cuota_over": 1.85, "cuota_under": 1.95},
            {"local": "IDV", "visitante": "Emelec", "cuota_1": 1.70, "cuota_x": 3.60, "cuota_2": 4.80, "cuota_over": 1.75, "cuota_under": 2.05}
        ]
    },
    "Copa Libertadores": {
        "xg_base": 2.85,
        "partidos": [
            {"local": "Flamengo", "visitante": "River Plate", "cuota_1": 2.05, "cuota_x": 3.25, "cuota_2": 3.60, "cuota_over": 1.90, "cuota_under": 1.90},
            {"local": "Palmeiras", "visitante": "LDU Quito", "cuota_1": 1.50, "cuota_x": 4.00, "cuota_2": 6.50, "cuota_over": 1.70, "cuota_under": 2.10}
        ]
    },
    "Copa Sudamericana": {
        "xg_base": 2.50,
        "partidos": [
            {"local": "Athletico PR", "visitante": "Racing Club", "cuota_1": 2.15, "cuota_x": 3.20, "cuota_2": 3.40, "cuota_over": 2.00, "cuota_under": 1.80}
        ]
    },
    "Premier League": {
        "xg_base": 2.82,
        "partidos": [
            {"local": "Arsenal", "visitante": "Man City", "cuota_1": 2.50, "cuota_x": 3.40, "cuota_2": 2.75, "cuota_over": 1.80, "cuota_under": 2.00}
        ]
    },
    "LaLiga España": {
        "xg_base": 2.55,
        "partidos": [
            {"local": "Real Madrid", "visitante": "Barcelona", "cuota_1": 2.15, "cuota_x": 3.50, "cuota_2": 3.10, "cuota_over": 1.65, "cuota_under": 2.20}
        ]
    }
}

# ==============================================================================
# 3. MOTOR MATEMÁTICO AVANZADO (POISSON + SHIN)
# ==============================================================================
def desmarginado_shin(cuotas):
    inv_c = np.array([1.0 / q if q > 1.0 else 0.0 for q in cuotas])
    if np.any(inv_c == 0.0): return np.zeros_like(cuotas), 0.0
    overround = np.sum(inv_c) - 1.0
    p_raw = inv_c / (1.0 + overround)
    return p_raw / np.sum(p_raw), overround

def calcular_matriz_poisson(lambda_h, mu_a, max_goles=5):
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            mat[i, j] = stats.poisson.pmf(i, lambda_h) * stats.poisson.pmf(j, mu_a)
    
    p_over = np.sum([mat[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    return p_over, 1.0 - p_over, mat

def extraer_partidos_ia(nombre_liga, api_key_input):
    key_limpia = api_key_input.strip().strip('"').strip("'") if api_key_input else ""
    if not key_limpia: return None, "Ingresa tu Gemini API Key en el menú lateral."
    try:
        genai.configure(api_key=key_limpia)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Proporciona partidos y cuotas recientes para '{nombre_liga}'. Devuelve SOLO un JSON puro:
        {{"partidos": [{{"local": "A", "visitante": "B", "cuota_1": 2.1, "cuota_x": 3.2, "cuota_2": 3.5, "cuota_over": 1.9, "cuota_under": 1.9}}]}}
        """
        res = model.generate_content(prompt)
        data = json.loads(res.text.replace("```json", "").replace("```", "").strip())
        return data.get("partidos", []), None
    except Exception as e:
        return None, f"Error Gemini: {str(e)}"

# ==============================================================================
# 4. BARRA LATERAL (CONTROLES)
# ==============================================================================
st.sidebar.markdown("### ⚙️ CONFIGURACIÓN Y DATOS")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")

liga_sel = st.sidebar.selectbox("Seleccionar Liga:", list(CATALOGO_LIGAS.keys()))
config_liga = CATALOGO_LIGAS[liga_sel]

if st.sidebar.button("📡 Cargar Jornada con IA", use_container_width=True):
    partidos_ia, err = extraer_partidos_ia(liga_sel, gemini_key)
    if partidos_ia:
        st.session_state['partidos_activos'] = partidos_ia
        st.sidebar.success(f"{len(partidos_ia)} partidos cargados.")
    else:
        st.sidebar.error(err)

partidos_disponibles = st.session_state.get('partidos_activos', config_liga["partidos"])
if not st.session_state.get('partidos_activos'):
    partidos_disponibles = config_liga["partidos"]

opciones = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_idx = st.sidebar.selectbox("Partido Target:", range(len(opciones)), format_func=lambda x: opciones[x])
p_data = partidos_disponibles[partido_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 CUOTAS DEL MERCADO")
c_1 = st.sidebar.number_input(f"Cuota {p_data['local']}:", value=float(p_data['cuota_1']), step=0.01)
c_x = st.sidebar.number_input("Cuota Empate:", value=float(p_data['cuota_x']), step=0.01)
c_2 = st.sidebar.number_input(f"Cuota {p_data['visitante']}:", value=float(p_data['cuota_2']), step=0.01)
c_over = st.sidebar.number_input("Cuota Over 2.5:", value=float(p_data['cuota_over']), step=0.01)
c_under = st.sidebar.number_input("Cuota Under 2.5:", value=float(p_data['cuota_under']), step=0.01)

# ==============================================================================
# 5. CÁLCULOS Y PROCESAMIENTO
# ==============================================================================
probs_1x2, ovr_1x2 = desmarginado_shin([c_1, c_x, c_2])
probs_ou, ovr_ou = desmarginado_shin([c_over, c_under])

xg_liga = config_liga.get("xg_base", 2.60)
lambda_h = (probs_1x2[0] * xg_liga * 1.1) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
mu_a = (probs_1x2[2] * xg_liga) / (probs_1x2[0] + probs_1x2[2] + 1e-5)
p_over_mod, p_under_mod, matriz_poisson = calcular_matriz_poisson(lambda_h, mu_a)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

# ==============================================================================
# 6. INTERFAZ MÓVIL PRINCIPAL (PESTAÑAS & CARDS PARALELAS)
# ==============================================================================
st.markdown(f"### ⚡ Terminal QG | <span style='color:#3fb950;'>{p_data['local']} vs {p_data['visitante']}</span>", unsafe_allow_html=True)
st.caption(f"Torneo: **{liga_sel}** | Expectativa total: **{(lambda_h + mu_a):.2f} xG**")

# Fila superior de tarjetas pareadas (NO se apilan verticalmente en celular)
st.markdown(f"""
<div class="mobile-flex-container">
    <div class="mobile-card">
        <small>EV OVER 2.5</small>
        <div style="color:{'#3fb950' if ev_over > 0 else '#ff7b72'};">{ev_over*100:+.2f}%</div>
    </div>
    <div class="mobile-card">
        <small>EV UNDER 2.5</small>
        <div style="color:{'#3fb950' if ev_under > 0 else '#ff7b72'};">{ev_under*100:+.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pestañas para dividir la pantalla del celular y evitar scroll infinito
tab_dash, tab_poisson, tab_kelly = st.tabs(["📊 Dashboard", "🧮 Matriz Marcadores", "💵 Gestión Banca"])

with tab_dash:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("<small><b>DENSIDAD DE GOLES</b></small>", unsafe_allow_html=True)
        x_g = np.linspace(0, 6, 100)
        y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
        fig1 = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#1f6feb')))
        fig1.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72")
        fig1.update_xaxes(range=[0, 6], fixedrange=True, showgrid=False)
        fig1.update_yaxes(fixedrange=True, showgrid=False)
        fig1.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=140)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

    with col_g2:
        st.markdown("<small><b>CINÉTICA TEMPORAL</b></small>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Scatter(x=[0, 15, 30, 45, 60, 75, 90], y=[lambda_h+mu_a, (lambda_h+mu_a)*0.8, (lambda_h+mu_a)*0.6, (lambda_h+mu_a)*0.4, (lambda_h+mu_a)*0.2, (lambda_h+mu_a)*0.1, 0], mode='lines+markers', line=dict(color='#3fb950')))
        fig2.update_xaxes(fixedrange=True, showgrid=False)
        fig2.update_yaxes(fixedrange=True, showgrid=False)
        fig2.update_layout(margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=140)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(f"""
    <div class="reasoning-box">
    • <b>Modelo:</b> Over ({p_over_mod*100:.1f}%) | Under ({p_under_mod*100:.1f}%).<br>
    • <b>Mercado (Shin):</b> Over ({probs_ou[0]*100:.1f}%) | Under ({probs_ou[1]*100:.1f}%).<br>
    • <b>Margen Casa:</b> {ovr_ou*100:.2f}% | <b>Vértice:</b> {'VENTAJA DETECTADA' if max(ev_over, ev_under) > 0 else 'SIN VALOR'}.
    </div>
    """, unsafe_allow_html=True)

with tab_poisson:
    st.markdown("<small><b>PROBABILIDAD DE MARCADORES EXACTOS (%)</b></small>", unsafe_allow_html=True)
    matriz_pct = np.round(matriz_poisson[:4, :4] * 100, 1)
    
    # Renderizado de tabla compacta para celular
    html_tabla = "<table style='width:100%; text-align:center; font-size:0.8em; border-collapse:collapse; margin-top:8px;'>"
    html_tabla += f"<tr style='background-color:#161b22;'><th>L / V</th><th>0</th><th>1</th><th>2</th><th>3</th></tr>"
    for i in range(4):
        html_tabla += f"<tr><td style='background-color:#161b22; font-weight:bold;'>{i}</td>"
        for j in range(4):
            val = matriz_pct[i, j]
            bg = f"rgba(31, 111, 235, {min(val/15.0, 0.8)})"
            html_tabla += f"<style>td {{ border: 1px solid #30363d; padding: 6px; }}</style><td style='background:{bg};'>{val}%</td>"
        html_tabla += "</tr>"
    html_tabla += "</table>"
    st.markdown(html_tabla, unsafe_allow_html=True)

with tab_kelly:
    st.markdown("<small><b>CRITERIO DE KELLY FRACCIONADO (1/4 KELLY)</b></small>", unsafe_allow_html=True)
    banca_total = st.number_input("Tu Banca Total ($):", value=100.0, step=10.0)
    
    # Cálculo Kelly
    best_ev = max(ev_over, ev_under)
    best_cuota = c_over if ev_over > ev_under else c_under
    best_prob = p_over_mod if ev_over > ev_under else p_under_mod
    best_market = "OVER 2.5" if ev_over > ev_under else "UNDER 2.5"
    
    if best_ev > 0:
        b = best_cuota - 1.0
        f_kelly = (b * best_prob - (1.0 - best_prob)) / b
        f_quarter = max(0.0, f_kelly * 0.25)
        monto_apuesta = banca_total * f_quarter
        
        st.success(f"🎯 **Recomendación ({best_market}):** Apostar **${monto_apuesta:.2f}** ({f_quarter*100:.2f}% de tu banca).")
    else:
        st.info("⚠️ No hay apuestas con Valor Esperado Positivo (+EV) para este partido.")
