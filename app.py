import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="CuantiBet Pro - Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0B0F17; }
    div[data-testid="stMetricValue"] { 
        font-family: 'JetBrains Mono', monospace; 
        font-weight: 700; 
        font-size: 1.4rem !important;
    }
    .stAlert { padding: 10px 15px; border-radius: 8px; }
    div[data-testid="stBlock"] { gap: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# BASE DE DATOS DE PARTIDOS Y CUOTAS POR LIGA (FEED API)
# ==============================================================================
LIGAS_DATA = {
    "Copa Sudamericana": [
        {
            "match": "Montevideo City Torque vs CA Tigre BA",
            "home": "Montevideo City Torque", "away": "CA Tigre BA",
            "xg_home": 0.89, "xg_away": 1.06,
            "odd_1": 3.40, "odd_x": 3.10, "odd_2": 2.25,
            "corners": 9.1, "cards": 4.0
        },
        {
            "match": "LDU Quito vs Lanús",
            "home": "LDU Quito", "away": "Lanús",
            "xg_home": 1.65, "xg_away": 0.85,
            "odd_1": 1.95, "odd_x": 3.30, "odd_2": 4.10,
            "corners": 10.2, "cards": 5.2
        }
    ],
    "Copa Libertadores": [
        {
            "match": "Flamengo vs Palmeiras",
            "home": "Flamengo", "away": "Palmeiras",
            "xg_home": 1.55, "xg_away": 1.10,
            "odd_1": 2.10, "odd_x": 3.25, "odd_2": 3.60,
            "corners": 10.0, "cards": 6.0
        }
    ],
    "LigaPro Ecuador": [
        {
            "match": "Independiente del Valle vs Emelec",
            "home": "Independiente del Valle", "away": "Emelec",
            "xg_home": 1.70, "xg_away": 0.90,
            "odd_1": 1.80, "odd_x": 3.40, "odd_2": 4.80,
            "corners": 9.0, "cards": 5.0
        }
    ]
}

# ==============================================================================
# MOTOR MATEMÁTICO COMPLETO (SHIN + POISSON + SEGUNDARIOS)
# ==============================================================================

def desmarginar_shin(odds, max_iter=100, tol=1e-6):
    odds = np.array(odds, dtype=float)
    if np.any(odds <= 1.0):
        return np.array([1/3, 1/3, 1/3]), 0.0
    
    implied = 1.0 / odds
    beta = np.sum(implied)
    
    if abs(beta - 1.0) < 1e-5:
        return implied, 0.0
    
    z = 0.0
    for _ in range(max_iter):
        f = np.sum(np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta)) - 2.0
        if abs(f) < tol:
            break
        f_prime = np.sum((z - 2 * (implied**2) / beta) / np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta))
        if f_prime == 0:
            break
        z = z - f / f_prime
        z = max(0.0, min(0.99, z))
        
    p_shin = (np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta) - z) / (2 * (1 - z))
    p_shin = p_shin / np.sum(p_shin)
    return p_shin, z

def calcular_mercados(lambda_home, mu_away, exp_corners, exp_cards):
    max_g = 7
    p_home = stats.poisson.pmf(np.arange(max_g), lambda_home)
    p_away = stats.poisson.pmf(np.arange(max_g), mu_away)
    matrix = np.outer(p_home, p_away)
    
    p1 = float(np.sum(np.tril(matrix, -1)))
    px = float(np.sum(np.diag(matrix)))
    p2 = float(np.sum(np.triu(matrix, 1)))
    
    btts_yes = float(np.sum(matrix[1:, 1:]))
    btts_no = 1.0 - btts_yes
    
    dc_1x = p1 + px
    dc_x2 = px + p2
    
    corners_o85 = 1.0 - stats.poisson.cdf(8, exp_corners)
    corners_u105 = stats.poisson.cdf(10, exp_corners)
    
    cards_o35 = 1.0 - stats.poisson.cdf(3, exp_cards)
    cards_u55 = stats.poisson.cdf(5, exp_cards)
    
    # Ranking para detectar opción infravalorada
    candidatos = [
        ("1X (Doble Oportunidad)", dc_1x, f"{1/dc_1x:.2f}"),
        ("X2 (Doble Oportunidad)", dc_x2, f"{1/dc_x2:.2f}"),
        ("BTTS - No", btts_no, f"{1/btts_no:.2f}"),
        ("Under 10.5 Córners", corners_u105, f"{1/corners_u105:.2f}"),
        ("Under 5.5 Tarjetas", cards_u55, f"{1/cards_u55:.2f}")
    ]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "matrix": matrix, "p1": p1, "px": px, "p2": p2,
        "btts_yes": btts_yes, "btts_no": btts_no,
        "corners_o85": corners_o85, "corners_u105": corners_u105,
        "cards_o35": cards_o35, "cards_u55": cards_u55,
        "top_infravalorado": candidatos[0]
    }

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
st.sidebar.title("⚙️ Configuración & API")
api_key = st.sidebar.text_input("🔑 API Key / Token", value="••••••••••••", type="password")

torneo = st.sidebar.selectbox("🏆 Campeonato / Liga", list(LIGAS_DATA.keys()))
partidos_disponibles = LIGAS_DATA[torneo]
nombres_partidos = [p["match"] for p in partidos_disponibles]
partido_seleccionado_str = st.sidebar.selectbox("📅 Partido de la Jornada", nombres_partidos)

match_default = next(p for p in partidos_disponibles if p["match"] == partido_seleccionado_str)

st.sidebar.markdown("---")
st.sidebar.subheader("⚔️ Equipos y Pronóstico")
home_team = st.sidebar.text_input("Equipo Local", match_default["home"])
away_team = st.sidebar.text_input("Equipo Visitante", match_default["away"])

st.sidebar.subheader("⚽ Proyección xG")
col_xg1, col_xg2 = st.sidebar.columns(2)
xg_home = col_xg1.number_input("xG Local", value=match_default["xg_home"], step=0.05, format="%.2f")
xg_away = col_xg2.number_input("xG Visitante", value=match_default["xg_away"], step=0.05, format="%.2f")

st.sidebar.subheader("📊 Cuotas del Mercado (1X2)")
col_o1, col_o2, col_o3 = st.sidebar.columns(3)
odd_1 = col_o1.number_input("Cuota 1", value=match_default["odd_1"], step=0.05)
odd_x = col_o2.number_input("Cuota X", value=match_default["odd_x"], step=0.05)
odd_2 = col_o3.number_input("Cuota 2", value=match_default["odd_2"], step=0.05)

st.sidebar.subheader("🚩 Promedios Auxiliares")
col_ax1, col_ax2 = st.sidebar.columns(2)
exp_corners = col_ax1.number_input("Córners Exp.", value=match_default["corners"], step=0.1)
exp_cards = col_ax2.number_input("Tarjetas Exp.", value=match_default["cards"], step=0.1)

bankroll = st.sidebar.number_input("Bankroll ($)", value=1000.0, step=50.0)

# ==============================================================================
# CÁLCULOS
# ==============================================================================
odds_list = [odd_1, odd_x, odd_2]
p_shin, z_insider = desmarginar_shin(odds_list)
sec = calcular_mercados(xg_home, xg_away, exp_corners, exp_cards)

m_p1, m_px, m_p2 = sec['p1'], sec['px'], sec['p2']
evs = [(m_p1 * odd_1) - 1, (m_px * odd_x) - 1, (m_p2 * odd_2) - 1]
best_market_idx = np.argmax(evs)
max_ev = evs[best_market_idx]

b_odd = odds_list[best_market_idx] - 1
p_win = [m_p1, m_px, m_p2][best_market_idx]
q_loss = 1.0 - p_win

kelly_full = (b_odd * p_win - q_loss) / b_odd if b_odd > 0 else 0
kelly_quarter = max(0.0, kelly_full * 0.25)
suggested_stake = bankroll * kelly_quarter

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
st.title(f"🏟️ {home_team} vs {away_team}")
st.caption("Evaluación Cuantitativa y Filtrado Anti-Trampas de Mercado")

st.markdown("#### 🎯 Panel de Mercados Probables & Opciones Infravaloradas")

# 4 Columnas completas en la sección superior
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

btts_label = "NO" if sec['btts_no'] > sec['btts_yes'] else "SÍ"
btts_prob = max(sec['btts_no'], sec['btts_yes']) * 100

with col_s1:
    with st.container(border=True):
        st.caption("⚽ AMBOS ANOTAN (BTTS)")
        st.subheader(f"{btts_label} ({btts_prob:.1f}%)")
        st.caption(f"Cuota Justa: @{100/btts_prob:.2f}")

corn_label = "Under 10.5" if sec['corners_u105'] > 0.50 else "Over 8.5"
corn_prob = sec['corners_u105'] * 100 if corn_label == "Under 10.5" else sec['corners_o85'] * 100

with col_s2:
    with st.container(border=True):
        st.caption("🚩 TIROS DE ESQUINA")
        st.subheader(f"{corn_label}")
        st.caption(f"Prob: {corn_prob:.1f}% | Cuota: @{100/corn_prob:.2f}")

card_label = "Under 5.5" if sec['cards_u55'] > 0.50 else "Over 3.5"
card_prob = sec['cards_u55'] * 100 if card_label == "Under 5.5" else sec['cards_o35'] * 100

with col_s3:
    with st.container(border=True):
        st.caption("🟨 TARJETAS TOTALES")
        st.subheader(f"{card_label}")
        st.caption(f"Prob: {card_prob:.1f}% | Cuota: @{100/card_prob:.2f}")

top_nombre, top_prob, top_cuota = sec['top_infravalorado']
with col_s4:
    with st.container(border=True):
        st.caption("💎 ALTA COBERTURA")
        st.subheader(f"{top_nombre}")
        st.caption(f"Prob: {top_prob*100:.1f}% | Cuota: @{top_cuota}")

st.markdown("---")

# Métricas cuantitativas
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("EV HÁNDICAP", f"{((m_p1 - m_p2)*100):.1f}%")
col_m2.metric("EV MERCADO 1X2", f"{max_ev*100:+.1f}%")
col_m3.metric("xG IMPLÍCITOS", f"{xg_home:.2f} / {xg_away:.2f}")
col_m4.metric("SHIN (INSIDER Z)", f"{z_insider:.4f}")

# Filtro Anti-Trampas
UMBRAL_EV = 0.025
if max_ev < UMBRAL_EV:
    st.warning(f"⚠️ **ABSTENERSE / BLOQUEO DE SEGURIDAD**: No existe ventaja financiera (+EV de {max_ev*100:.1f}% es inferior al umbral del 2.5%). Execution cancelada para proteger el capital.")
    suggested_stake = 0.0
else:
    st.success(f"🚀 **VENTAJA DETECTADA**: Esperanza Matemática (+EV) del {max_ev*100:.1f}%.")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
nombres_mkt = [home_team, "Empate (X)", away_team]
col_e1.metric("Selección Recomendada", nombres_mkt[best_market_idx] if max_ev >= UMBRAL_EV else "Ninguna (Bloqueado)")
col_e2.metric("Cuota Justa / Mercado", f"{1/p_win:.2f} / {odds_list[best_market_idx]}")
col_e3.metric("Esperanza (+EV)", f"{max_ev*100:+.2f}%")
col_e4.metric("Stake Sugerido (Kelly 1/4)", f"${suggested_stake:.1f} ({kelly_quarter*100:.1f}%)")

st.markdown("---")

# Visualizaciones
col_g1, col_g2 = st.columns(2)

home_short = home_team[:3].upper() if len(home_team) > 3 else home_team.upper()
away_short = away_team[:3].upper() if len(away_team) > 3 else away_team.upper()

with col_g1:
    st.markdown("##### 📊 Comparativa de Probabilidades (1X2)")
    categories = [f"{home_short}", "EMP", f"{away_short}"]
    
    fig_bar = go.Figure(data=[
        go.Bar(
            name='Tu Modelo',
            x=categories,
            y=[m_p1*100, m_px*100, m_p2*100],
            marker=dict(color='#10B981'),
            text=[f"{m_p1*100:.1f}%", f"{m_px*100:.1f}%", f"{m_p2*100:.1f}%"],
            textposition='inside',
            textfont=dict(size=11, color='#FFFFFF')
        ),
        go.Bar(
            name='Mercado Shin',
            x=categories,
            y=[p_shin[0]*100, p_shin[1]*100, p_shin[2]*100],
            marker=dict(color='#6366F1'),
            text=[f"{p_shin[0]*100:.1f}%", f"{p_shin[1]*100:.1f}%", f"{p_shin[2]*100:.1f}%"],
            textposition='inside',
            textfont=dict(size=11, color='#FFFFFF')
        )
    ])
    fig_bar.update_layout(
        barmode='group',
        height=300,
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11, color="#E2E8F0")),
        font=dict(color="#E2E8F0"),
        xaxis=dict(fixedrange=True, tickfont=dict(size=11, color="#94A3B8")),
        yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#1E293B', showticklabels=True)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

with col_g2:
    st.markdown("##### 🔥 Matriz de Marcadores Probables")
    df_m = np.round(sec["matrix"][:5, :5] * 100, 1)
    
    custom_colorscale = [
        [0.0, "#0F172A"],
        [0.25, "#1E293B"],
        [0.50, "#0369A1"],
        [0.75, "#0284C7"],
        [1.0, "#38BDF8"]
    ]
    
    fig_hm = go.Figure(data=go.Heatmap(
        z=df_m,
        x=["0", "1", "2", "3", "4"],
        y=["0", "1", "2", "3", "4"],
        colorscale=custom_colorscale,
        showscale=False,
        xgap=2, ygap=2,
        text=df_m,
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10, family='JetBrains Mono', color="#FFFFFF")
    ))

    fig_hm.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0"),
        xaxis=dict(title=dict(text=f"Goles {away_short}", font=dict(size=11, color="#94A3B8")), tickfont=dict(color="#E2E8F0")),
        yaxis=dict(title=dict(text=f"Goles {home_short}", font=dict(size=11, color="#94A3B8")), tickfont=dict(color="#E2E8F0"), autorange="reversed")
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
