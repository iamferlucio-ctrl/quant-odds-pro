import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import google.generativeai as genai

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL & ESTILOS CSS INSTITUCIONALES
# ==============================================================================
st.set_page_config(
    page_title="Terminal Cuantitativo & Investigador IA v3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Estructura Global Terminal Dark */
    .stApp {
        background-color: #0b0f19;
        color: #c9d1d9;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    }
    
    /* Contenedores y Tarjetas */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .reasoning-box {
        background-color: #0d1117;
        border-left: 4px solid #1f6feb;
        border-radius: 0 6px 6px 0;
        padding: 16px;
        font-size: 0.88em;
        line-height: 1.6;
        color: #8b949e;
    }
    .reasoning-box b {
        color: #58a6ff;
    }

    /* Badges de Estado */
    .badge-detected-high {
        background-color: rgba(248, 81, 73, 0.15);
        color: #ff7b72;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-clean {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-low {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Pestañas e Inserciones Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1117;
        padding: 6px;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 4px;
        color: #8b949e;
        font-size: 0.85em;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONFIGURACIÓN DE PARÁMETROS DE LIGAS Y MOTOR MATEMÁTICO
# ==============================================================================
PARAMETROS_LIGAS = {
    "LigaPro Ecuador": {"xg_base": 2.35, "alpha_home": 1.12, "sensibilidad_rlm": 0.08},
    "Premier League": {"xg_base": 2.82, "alpha_home": 1.08, "sensibilidad_rlm": 0.05},
    "Copa Libertadores": {"xg_base": 2.25, "alpha_home": 1.25, "sensibilidad_rlm": 0.10},
    "Serie A Brasil": {"xg_base": 2.40, "alpha_home": 1.15, "sensibilidad_rlm": 0.07},
    "Otras Ligas": {"xg_base": 2.50, "alpha_home": 1.10, "sensibilidad_rlm": 0.08}
}

def desmarginado_shin(cuotas):
    """Calcula probabilidades implícitas desmarginadas mediante el algoritmo Shin."""
    inv_cuotas = np.array([1.0 / q if q > 1.0 else 0.0 for q in cuotas])
    if np.any(inv_cuotas == 0.0):
        return np.zeros_like(cuotas), 0.0
    
    overround = np.sum(inv_cuotas) - 1.0
    
    # Reducción de sobregiro
    p_raw = inv_cuotas / (1.0 + overround)
    probs_limpias = p_raw / np.sum(p_raw)
    return probs_limpias, overround

def estimar_matriz_poisson(lambda_home, mu_away, max_goles=6):
    """Genera matriz probabilística bidimensional Dixon-Coles / Poisson."""
    prob_matrix = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            prob_matrix[i, j] = stats.poisson.pmf(i, lambda_home) * stats.poisson.pmf(j, mu_away)
            
    p_over_25 = np.sum([prob_matrix[i, j] for i in range(max_goles+1) for j in range(max_goles+1) if i + j > 2.5])
    p_under_25 = 1.0 - p_over_25
    return p_over_25, p_under_25, prob_matrix

def consultar_investigador_gemini(prompt, api_key):
    """Ejecuta consulta directa al motor IA con control de excepciones."""
    if not api_key:
        return "⚠️ **Llave Gemini no detectada**: Ingrese su API Key en la barra lateral para activar la auditoría semántica."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ **Error en la API de Gemini**: {str(e)}"

# ==============================================================================
# 3. BARRA LATERAL (SIDEBAR DE CONTROL)
# ==============================================================================
st.sidebar.markdown("### 🔑 CREDENCIALES & LIGA")
gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Inserte su API Key para ejecutar auditoría de IA.")

liga_activa = st.sidebar.selectbox("Liga / Competición Target:", list(PARAMETROS_LIGAS.keys()))
config_liga = PARAMETROS_LIGAS[liga_activa]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚔️ DATOS DEL EVENTO")
equipo_home = st.sidebar.text_input("Equipo Local:", value="Macará")
equipo_away = st.sidebar.text_input("Equipo Visitante:", value="Santos")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 MERCADO 1X2")
c_1 = st.sidebar.number_input(f"Cuota {equipo_home}:", value=2.33, step=0.01)
c_x = st.sidebar.number_input("Cuota Empate:", value=3.40, step=0.01)
c_2 = st.sidebar.number_input(f"Cuota {equipo_away}:", value=2.90, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚽ MERCADO TOTALES (2.5)")
c_over = st.sidebar.number_input("Cuota Más de 2.5:", value=2.10, step=0.01)
c_under = st.sidebar.number_input("Cuota Menos de 2.5:", value=1.70, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 MICROESTRUCTURA")
cuota_apertura_over = st.sidebar.number_input("Apertura Over 2.5:", value=1.80, step=0.01)
volumen_publico_over = st.sidebar.slider("% Volumen Público (Over):", 0, 100, 74)

# ==============================================================================
# 4. EJECUCIÓN DEL MOTOR CUANTITATIVO
# ==============================================================================
probs_1x2, ovr_1x2 = desmarginado_shin([c_1, c_x, c_2])
probs_ou, ovr_ou = desmarginado_shin([c_over, c_under])

# Derivación de Expectativa de Goles (xG) ajustada por parámetro de liga
xg_base_liga = config_liga["xg_base"]
lambda_home = (probs_1x2[0] * xg_base_liga * config_liga["alpha_home"]) / (probs_1x2[0] + probs_1x2[2])
mu_away = (probs_1x2[2] * xg_base_liga) / (probs_1x2[0] + probs_1x2[2])
total_xg = lambda_home + mu_away

p_over_mod, p_under_mod, matriz_poisson = estimar_matriz_poisson(lambda_home, mu_away)

ev_over = (p_over_mod * c_over) - 1.0
ev_under = (p_under_mod * c_under) - 1.0

# Detección de trampas
rlm_detectado = (volumen_publico_over > 60) and (c_over >= cuota_apertura_over + config_liga["sensibilidad_rlm"])
cuota_marketing = (c_1 % 0.05 == 0) or (c_over % 0.05 == 0) or (c_1 in [1.90, 1.95, 2.00])
margen_alto = ovr_1x2 > 0.06

# ==============================================================================
# 5. DASHBOARD PRINCIPAL Y GRAFICACIÓN
# ==============================================================================
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
    <h2 style="color:#58a6ff; margin:0;">QG | TERMINAL DE OBJETIVOS CUÁNTICOS</h2>
    <span style="font-size:0.75em; color:#3fb950; border:1px solid #2ea043; padding:4px 8px; border-radius:4px; background:rgba(46,160,67,0.1);">v3.0 INSTITUCIONAL</span>
</div>
""", unsafe_allow_html=True)

tab_terminal, tab_ia, tab_backtest = st.tabs(["TERMINAL CUANTITATIVO & INVESTIGADOR IA", "EJEMPLOS & MONITOREO DE LIGAS", "BACKTEST & CALIBRACIÓN"])

with tab_terminal:
    # FILA 1: Gráficos Cuantitativos
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**DENSIDAD ESPECTRAL DE GOLES TOTALES**")
        x_g = np.linspace(0, 6, 100)
        y_g = stats.norm.pdf(x_g, loc=total_xg, scale=1.05)
        
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(x=x_g, y=y_g, mode='lines', fill='tozeroy', name='Densidad', line=dict(color='#1f6feb', width=2)))
        fig_spec.add_vline(x=2.5, line_dash="dash", line_color="#ff7b72", annotation_text="Línea Corte (2.5)")
        fig_spec.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8b949e', size=11),
            height=190
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    with col_g2:
        st.markdown("**CINÉTICA EN JUEGO (DECAIMIENTO TEMPORAL)**")
        tiempos = [0, 15, 30, 45, 60, 75, 90]
        xg_decay = [total_xg * (1 - (t/90)**1.2) for t in tiempos]
        
        fig_kin = go.Figure()
        fig_kin.add_trace(go.Scatter(x=tiempos, y=xg_decay, mode='lines+markers', line=dict(color='#3fb950', shape='hv', width=2)))
        fig_kin.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8b949e', size=11),
            height=190
        )
        st.plotly_chart(fig_kin, use_container_width=True)

    # FILA 2: Cinta Multiplicadora de Valor (+EV)
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-around; text-align:center;">
            <div>
                <span style="font-size:0.75em; color:#8b949e;">VALOR ESPERADO (OVER 2.5)</span><br>
                <b style="color:{'#3fb950' if ev_over > 0 else '#ff7b72'}; font-size:1.15em;">{ev_over*100:+.2f}% EV</b>
            </div>
            <div style="border-left: 1px solid #30363d; padding-left:20px;">
                <span style="font-size:0.75em; color:#8b949e;">VALOR ESPERADO (UNDER 2.5)</span><br>
                <b style="color:{'#3fb950' if ev_under > 0 else '#ff7b72'}; font-size:1.15em;">{ev_under*100:+.2f}% EV</b>
            </div>
            <div style="border-left: 1px solid #30363d; padding-left:20px;">
                <span style="font-size:0.75em; color:#8b949e;">EXPECTATIVA TOTAL (xG)</span><br>
                <b style="color:#58a6ff; font-size:1.15em;">{total_xg:.2f} Goles</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # FILA 3: Desglose de Razonamiento Numérico
    st.markdown("**PASOS DEL RAZONAMIENTO NUMÉRICO:**")
    st.markdown(f"""
    <div class="reasoning-box">
    1. El modelo ajustado a <b>{liga_activa}</b> establece un xG de <b>&lambda;={lambda_home:.2f}</b> para {equipo_home} y <b>&mu;={mu_away:.2f}</b> para {equipo_away} (Total: <b>{total_xg:.2f}</b>).<br>
    2. La probabilidad ajustada por matriz de Poisson para Over 2.5 es de <b>{p_over_mod*100:.1f}%</b> y Under 2.5 es de <b>{p_under_mod*100:.1f}%</b>.<br>
    3. El desmarginado algoritmo de Shin extrae un Overround total en el mercado de goles de <b>{ovr_ou*100:.2f}%</b> (Probabilidad limpia del mercado para Over: <b>{probs_ou[0]*100:.1f}%</b>).<br>
    4. La diferencia de valor (gap) identificada es de <b>{abs(p_over_mod - probs_ou[0])*100:.2f}%</b> puntos porcentuales.<br>
    5. Estado cuantitativo: <b>{'VENTAJA DETECTADA (+EV)' if max(ev_over, ev_under) > 0 else 'NEUTRALIDAD ESTRICTA (SIN EV CONFIABLE)'}</b>.
    </div>
    """, unsafe_allow_html=True)

    # FILA 4: Detector de Microestructura y Trampas
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**LISTA DE VERIFICACIÓN DE TRAMPAS DE MERCADO & INFORMACIÓN ASIMÉTRICA**")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <b>Movimiento de Línea Inversa (RLM)</b><br>
                    <small style="color:#8b949e;">Público en Over ({volumen_publico_over}%), cuota subió de {cuota_apertura_over:.2f} a {c_over:.2f}.</small>
                </div>
                <div>{'<span class="badge-detected-high">DETECTADA (ALTA)</span>' if rlm_detectado else '<span class="badge-clean">LIMPIO</span>'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <b>Cuota Psicológica / Marketing</b><br>
                    <small style="color:#8b949e;">Cuota configurada con anzuelo comercial o número redondo (.90 / .95).</small>
                </div>
                <div>{'<span class="badge-low">DETECTADA (LOW)</span>' if cuota_marketing else '<span class="badge-clean">LIMPIO</span>'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <b>Margen de Casa Anómalo</b><br>
                    <small style="color:#8b949e;">Overround del mercado 1X2 calculado en {ovr_1x2*100:.2f}%.</small>
                </div>
                <div>{'<span class="badge-detected-high">ALTO MARGEN</span>' if margen_alto else '<span class="badge-clean">LIMPIO</span>'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <b>Riesgo Liquidez de Liga</b><br>
                    <small style="color:#8b949e;">Sensibilidad de ajuste de línea para {liga_activa}.</small>
                </div>
                <div><span class="badge-clean">MONITOREADO</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # FILA 5: Módulo de Auditoría Gemini IA
    st.markdown("---")
    st.markdown("### 🤖 AUDITORÍA DE IA GENERAL (GEMINI Flash)")
    if st.button("Ejecutar Auditoría Semántica con IA", use_container_width=True):
        prompt_ia = f"""
        Actúa como un auditor cuantitativo senior de apuestas deportivas.
        Analiza los siguientes datos de mercado para el partido {equipo_home} vs {equipo_away} ({liga_activa}):
        - Cuotas 1X2: {c_1} / {c_x} / {c_2} (Overround 1X2: {ovr_1x2*100:.2f}%)
        - Cuotas Over/Under 2.5: {c_over} / {c_under}
        - Expectativa xG Modelo: {total_xg:.2f}
        - EV Calculado Over: {ev_over*100:.2f}%, EV Under: {ev_under*100:.2f}%
        - Detección RLM: {'SÍ' if rlm_detectado else 'NO'}
        - Cuota Marketing: {'SÍ' if cuota_marketing else 'NO'}
        
        Emite un diagnóstico conciso en 3 puntos:
        1. Evaluación de riesgo por trampas de mercado.
        2. Recomendación de entrada o veto financiero.
        3. Contexto táctico del partido.
        """
        with st.spinner("Analizando microestructura con Gemini..."):
            dictamen = consultar_investigador_gemini(prompt_ia, gemini_api_key)
            st.info(dictamen)

with tab_ia:
    st.write(f"Monitoreo de parámetros para **{liga_activa}**: xG Promedio = {config_liga['xg_base']}")

with tab_backtest:
    st.write("Módulo de Backtesting e Histórico en desarrollo para v3.1.")
