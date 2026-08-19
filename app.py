import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai
import json

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS (UI FIDELIDAD TOTAL AL DISEÑO ORIGINAL)
# ==============================================================================
st.set_page_config(
    page_title="Terminal Quant v7.0",
    page_icon="🏟️",
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
    .stApp { background-color: #0a0d14; color: #d1d5db; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    
    /* Header Principal */
    .header-card {
        background: linear-gradient(135deg, #131b2e 0%, #0d1322 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 16px;
    }
    .header-title { font-size: 1.35em; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }
    .header-subtitle { font-size: 0.72em; font-weight: 700; color: #64748b; letter-spacing: 1.5px; margin-top: 4px; text-transform: uppercase; }

    /* Grilla de Métricas Móvil (2 Columnas Rígidas en Pantalla Táctil) */
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
    .quant-val-green { font-size: 1.35em; font-weight: 800; color: #ef4444; margin-top: 2px; } /* Ajustado según estética */
    .quant-val-positive { font-size: 1.35em; font-weight: 800; color: #10b981; margin-top: 2px; }
    .quant-val-neutral { font-size: 1.35em; font-weight: 800; color: #f59e0b; margin-top: 2px; }
    .quant-val-white { font-size: 1.35em; font-weight: 800; color: #f9fafb; margin-top: 2px; }

    /* Módulo de Orden Aprobada / Rechazada */
    .execution-approved {
        background-color: rgba(6, 78, 59, 0.25);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .execution-rejected {
        background-color: rgba(127, 29, 29, 0.25);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .badge-approved { background-color: #10b981; color: #022c22; font-weight: 800; font-size: 0.72em; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }
    .badge-rejected { background-color: #ef4444; color: #450a0a; font-weight: 800; font-size: 0.72em; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }
    .execution-title { font-size: 1.15em; font-weight: 800; color: #ffffff; margin-top: 8px; }
    .execution-desc { font-size: 0.80em; color: #9ca3af; margin-top: 4px; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANCO GLOBAL DE LIGAS Y BUSCADOR AVANZADO
# ==============================================================================
CATALOGO_COMPLETO = {
    "Ecuador - LigaPro": [
        {"local": "Macará", "visitante": "Santos", "c1": 2.33, "cx": 3.40, "c2": 2.90, "cover": 2.10, "cunder": 1.70, "ah_local": -0.25, "c_ah1": 2.02, "c_ah2": 1.82},
        {"local": "LDU Quito", "visitante": "Barcelona SC", "c1": 1.95, "cx": 3.30, "c2": 3.80, "cover": 1.85, "cunder": 1.95, "ah_local": -0.50, "c_ah1": 1.95, "c_ah2": 1.90},
        {"local": "IDV", "visitante": "Emelec", "c1": 1.70, "cx": 3.60, "c2": 4.80, "cover": 1.75, "cunder": 2.05, "ah_local": -0.75, "c_ah1": 1.90, "c_ah2": 1.95}
    ],
    "Conmebol - Libertadores": [
        {"local": "Flamengo", "visitante": "River Plate", "c1": 2.05, "cx": 3.25, "c2": 3.60, "cover": 1.90, "cunder": 1.90, "ah_local": -0.50, "c_ah1": 2.05, "c_ah2": 1.80},
        {"local": "Palmeiras", "visitante": "LDU Quito", "c1": 1.50, "cx": 4.00, "c2": 6.50, "cover": 1.70, "cunder": 2.10, "ah_local": -1.00, "c_ah1": 1.85, "c_ah2": 2.00}
    ],
    "Conmebol - Sudamericana": [
        {"local": "Athletico PR", "visitante": "Racing Club", "c1": 2.15, "cx": 3.20, "c2": 3.40, "cover": 2.00, "cunder": 1.80, "ah_local": -0.25, "c_ah1": 1.85, "c_ah2": 2.00}
    ],
    "Inglaterra - Premier League": [
        {"local": "Arsenal", "visitante": "Manchester City", "c1": 2.50, "cx": 3.40, "c2": 2.75, "cover": 1.80, "cunder": 2.00, "ah_local": 0.00, "c_ah1": 1.88, "c_ah2": 1.98}
    ],
    "España - LaLiga": [
        {"local": "Real Madrid", "visitante": "Barcelona", "c1": 2.15, "cx": 3.50, "c2": 3.10, "cover": 1.65, "cunder": 2.20, "ah_local": -0.25, "c_ah1": 1.90, "c_ah2": 1.95}
    ]
}

# ==============================================================================
# 3. EXTRACCIÓN DINÁMICA IA (GEMINI MULTI-MERCADO)
# ==============================================================================
def buscar_partidos_ia_avanzado(query_busqueda, api_key):
    key_clean = api_key.strip().strip('"').strip("'") if api_key else ""
    if not key_clean:
        return None, "Ingresa tu API Key en la barra lateral para buscar en vivo."
    
    try:
        genai.configure(api_key=key_clean)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Actúa como un feed de datos cuantitativos deportivos. Proporciona partidos y cuotas para la consulta: '{query_busqueda}'.
        Devuelve EXCLUSIVAMENTE un JSON con esta estructura exacta:
        {{
            "partidos": [
                {{
                    "local": "Equipo A",
                    "visitante": "Equipo B",
                    "c1": 2.10, "cx": 3.20, "c2": 3.40,
                    "cover": 1.90, "cunder": 1.90,
                    "ah_local": -0.25, "c_ah1": 1.90, "c_ah2": 1.95
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("partidos", []), None
    except Exception as e:
        return None, f"Error en búsqueda: {str(e)}"

# ==============================================================================
# 4. MOTOR MATEMÁTICO: DE-MARGINALIZACIÓN DE SHIN & DIXON-COLES
# ==============================================================================
def estimar_shin(cuotas_1x2):
    """Calcula probabilidades reales desmarginadas y el factor Z (Insider Trading)"""
    c = np.array(cuotas_1x2, dtype=float)
    if np.any(c <= 1.0): return np.array([0.33, 0.33, 0.33]), 0.0
    
    inv_c = 1.0 / c
    sum_inv = np.sum(inv_c)
    margin = sum_inv - 1.0
    
    # Estimación numérica del parámetro z de Shin
    z = max(0.0001, margin * 0.15) # Insider factor Z aproximado
    p_raw = (np.sqrt(z**2 + 4 * (1 - z) * (inv_c / sum_inv)) - z) / (2 * (1 - z))
    p_norm = p_raw / np.sum(p_raw)
    return p_norm, z

def modelo_dixon_coles(lambda_h, mu_a, rho=-0.13, max_goles=5):
    """Matriz Bivariada de Poisson con corrección de dependencia para marcadores bajos"""
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p_i = stats.poisson.pmf(i, lambda_h)
            p_j = stats.poisson.pmf(j, mu_a)
            
            # Factor de ajuste Dixon-Coles (tau)
            if i == 0 and j == 0: tau = 1.0 - (lambda_h * mu_a * rho)
            elif i == 0 and j == 1: tau = 1.0 + (lambda_h * rho)
            elif i == 1 and j == 0: tau = 1.0 + (mu_a * rho)
            elif i == 1 and j == 1: tau = 1.0 - rho
            else: tau = 1.0
            
            mat[i, j] = max(0.0, p_i * p_j * tau)
            
    mat = mat / np.sum(mat) # Normalización
    
    p_home = np.sum(np.tril(mat, -1))
    p_draw = np.sum(np.diag(mat))
    p_away = np.sum(np.triu(mat, 1))
    p_over = np.sum([mat[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    
    return p_home, p_draw, p_away, p_over, 1.0 - p_over, mat

# ==============================================================================
# 5. CONTROLES Y SELECCIÓN (BARRA LATERAL / BÚSQUEDA)
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & BUSCADOR")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 BUSCADOR DE PARTIDOS EN VIVO")
query_custom = st.sidebar.text_input("Buscar partido o liga (ej: 'Barcelona vs Real Madrid'):")

if st.sidebar.button("🔎 Ejecutar Búsqueda IA", use_container_width=True):
    if query_custom:
        with st.sidebar.spinner("Buscando mercados cuantitativos..."):
            partidos_encontrados, err = buscar_partidos_ia_avanzado(query_custom, gemini_key)
            if partidos_encontrados:
                st.session_state['partidos_activos'] = partidos_encontrados
                st.sidebar.success(f"¡{len(partidos_encontrados)} partidos cargados!")
            else:
                st.sidebar.error(err)

liga_sel = st.sidebar.selectbox("O selecciona del Catálogo:", list(CATALOGO_COMPLETO.keys()))

if 'partidos_activos' not in st.session_state or not query_custom:
    partidos_disponibles = CATALOGO_COMPLETO[liga_sel]
else:
    partidos_disponibles = st.session_state['partidos_activos']

opciones_str = [f"{p['local']} vs {p['visitante']}" for p in partidos_disponibles]
partido_idx = st.sidebar.selectbox("Evento Activo:", range(len(opciones_str)), format_func=lambda x: opciones_str[x])
p = partidos_disponibles[partido_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AJUSTE DE CUOTAS")
c_1 = st.sidebar.number_input(f"1 ({p['local']}):", value=float(p['c1']), step=0.01)
c_x = st.sidebar.number_input("X (Empate):", value=float(p['cx']), step=0.01)
c_2 = st.sidebar.number_input(f"2 ({p['visitante']}):", value=float(p['c2']), step=0.01)
c_over = st.sidebar.number_input("Over 2.5 Goles:", value=float(p['cover']), step=0.01)
c_under = st.sidebar.number_input("Under 2.5 Goles:", value=float(p['cunder']), step=0.01)
ah_line = st.sidebar.number_input("Línea Hándicap Asiático:", value=float(p.get('ah_local', -0.25)), step=0.25)
c_ah1 = st.sidebar.number_input("Cuota Hándicap Local:", value=float(p.get('c_ah1', 1.90)), step=0.01)

# ==============================================================================
# 6. PROCESAMIENTO CUANTITATIVO Y MÉTRICAS
# ==============================================================================
probs_shin, insider_z = estimar_shin([c_1, c_x, c_2])

# xG Implícitos vía Poisson Inverso
total_xg_est = 2.60
lambda_h = (probs_shin[0] * total_xg_est * 1.08) / (probs_shin[0] + probs_shin[2] + 1e-5)
mu_a = (probs_shin[2] * total_xg_est) / (probs_shin[0] + probs_shin[2] + 1e-5)

# Modelo Dixon-Coles
p_dc_1, p_dc_x, p_dc_2, p_dc_over, p_dc_under, mat_dc = modelo_dixon_coles(lambda_h, mu_a)

# Cálculo de EV (Expected Values)
ev_1x2 = max((p_dc_1 * c_1 - 1.0), (p_dc_x * c_x - 1.0), (p_dc_2 * c_2 - 1.0)) * 100
ev_ou = max((p_dc_over * c_over - 1.0), (p_dc_under * c_under - 1.0)) * 100

# EV Hándicap (Aproximación por modelo)
prob_ah_cover = p_dc_1 + (p_dc_x * 0.5 if abs(ah_line) == 0.25 else 0)
ev_handicap = (prob_ah_cover * c_ah1 - 1.0) * 100

max_ev_global = max(ev_1x2, ev_ou, ev_handicap)
ejecucion_permitida = max_ev_global > 0

# ==============================================================================
# 7. INTERFAZ GRÁFICA (EXACTA A LA CAPTURA DE PANTALLA)
# ==============================================================================

# Header Principal
st.markdown(f"""
<div class="header-card">
    <div class="header-title">🏟️ {p['local']} vs {p['visitante']}</div>
    <div class="header-subtitle">Terminal de Inteligencia Cuantitativa</div>
</div>
""", unsafe_allow_html=True)

# Grilla de 5 Métricas Clave (Layout 2x2 + 1 Card Ancha en Móvil)
st.markdown(f"""
<div class="quant-grid">
    <div class="quant-card">
        <div class="quant-label">EV HÁNDICAP</div>
        <div class="{'quant-val-positive' if ev_handicap > 0 else 'quant-val-green'}">{ev_handicap:+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV MERCADO 1X2</div>
        <div class="{'quant-val-positive' if ev_1x2 > 0 else 'quant-val-green'}">{ev_1x2:+.1f}%</div>
    </div>
    <div class="quant-card">
        <div class="quant-label">EV OVER/UNDER</div>
        <div class="{'quant-val-positive' if ev_ou > 0 else 'quant-val-green'}">{ev_ou:+.1f}%</div>
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

# Módulo de Decisiones de Ejecución
if ejecucion_permitida:
    st.markdown(f"""
    <div class="execution-approved">
        <span class="badge-approved">EJECUCIÓN PERMITIDA</span>
        <div class="execution-title">Orden Aprobada (+EV)</div>
        <div class="execution-desc">Auditoría realizada cruzando el modelo Dixon-Coles frente a la de-marginalización de Shin. Se detectó una ventaja cuantitativa exploitable de <b>+{max_ev_global:.2f}% EV</b>.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="execution-rejected">
        <span class="badge-rejected">EJECUCIÓN RECHAZADA</span>
        <div class="execution-title">Orden Bloqueada (-EV)</div>
        <div class="execution-desc">Auditoría realizada cruzando el modelo Dixon-Coles frente a la de-marginalización de Shin. El mercado no ofrece margen positivo sobre las cuotas actuales.</div>
    </div>
    """, unsafe_allow_html=True)

# Sección de Probabilidades Comparativas
st.markdown("### 📊 Probabilidades: Modelo vs Casa")

col_fig1, col_fig2 = st.columns(2)

with col_fig1:
    fig_bar = go.Figure(data=[
        go.Bar(name='Modelo (Dixon-Coles)', x=['1', 'X', '2'], y=[p_dc_1*100, p_dc_x*100, p_dc_2*100], marker_color='#10b981'),
        go.Bar(name='Mercado (Shin)', x=['1', 'X', '2'], y=[probs_shin[0]*100, probs_shin[1]*100, probs_shin[2]*100], marker_color='#3b82f6')
    ])
    fig_bar.update_layout(
        barmode='group',
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_bar.update_xaxes(fixedrange=True)
    fig_bar.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

with col_fig2:
    # Densidad de Goles
    x_g = np.linspace(0, 6, 100)
    y_g = stats.norm.pdf(x_g, loc=(lambda_h + mu_a), scale=1.0)
    fig_density = go.Figure(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', line=dict(color='#10b981')))
    fig_density.add_vline(x=2.5, line_dash="dash", line_color="#ef4444")
    fig_density.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af', size=11)
    )
    fig_density.update_xaxes(fixedrange=True)
    fig_density.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_density, use_container_width=True, config=PLOTLY_CONFIG)
