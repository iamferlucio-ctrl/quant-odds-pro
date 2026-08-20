import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="CuantiBet Pro - Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. MOTOR MATEMÁTICO & MODELO SHIN
# ==============================================================================

def desmarginar_shin(odds, max_iter=100, tol=1e-6):
    """
    Desmargina cuotas reales utilizando el Modelo de Shin (1992, 1993)
    para aislar las probabilidades reales implícitas sin el sesgo de la casa.
    """
    odds = np.array(odds, dtype=float)
    if np.any(odds <= 1.0):
        return np.array([1/3, 1/3, 1/3]), 0.0
    
    implied = 1.0 / odds
    beta = np.sum(implied)
    
    if abs(beta - 1.0) < 1e-5:
        return implied, 0.0
    
    n = len(odds)
    z = 0.0
    for _ in range(max_iter):
        f = np.sum(np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta)) - 2.0
        if abs(f) < tol:
            break
        # Derivada numérica
        f_prime = np.sum((z - 2 * (implied**2) / beta) / np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta))
        if f_prime == 0:
            break
        z = z - f / f_prime
        z = max(0.0, min(0.99, z))
        
    p_shin = (np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta) - z) / (2 * (1 - z))
    p_shin = p_shin / np.sum(p_shin)
    return p_shin, z

def calcular_mercados_secundarios(lambda_home, mu_away, exp_corners=9.2, exp_cards=4.2):
    """
    Calcula probabilidades estocásticas para mercados secundarios e infravalorados.
    """
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
    dc_12 = p1 + p2
    
    lambda_ht = lambda_home * 0.44
    mu_ht = mu_away * 0.44
    over_05_ht = 1.0 - (stats.poisson.pmf(0, lambda_ht) * stats.poisson.pmf(0, mu_ht))
    
    corners_o85 = 1.0 - stats.poisson.cdf(8, exp_corners)
    corners_u105 = stats.poisson.cdf(10, exp_corners)
    
    cards_o35 = 1.0 - stats.poisson.cdf(3, exp_cards)
    cards_u55 = stats.poisson.cdf(5, exp_cards)
    
    candidatos = [
        ("Doble Oportunidad 1X", dc_1x, f"{1/dc_1x:.2f}"),
        ("Doble Oportunidad X2", dc_x2, f"{1/dc_x2:.2f}"),
        ("BTTS - Sí", btts_yes, f"{1/btts_yes:.2f}"),
        ("BTTS - No", btts_no, f"{1/btts_no:.2f}"),
        ("Over 0.5 Goles 1HT", over_05_ht, f"{1/over_05_ht:.2f}"),
        ("Over 8.5 Córners", corners_o85, f"{1/corners_o85:.2f}"),
        ("Over 3.5 Tarjetas", cards_o35, f"{1/cards_o35:.2f}")
    ]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "matrix": matrix,
        "p1": p1, "px": px, "p2": p2,
        "btts_yes": btts_yes, "btts_no": btts_no,
        "corners_o85": corners_o85, "corners_u105": corners_u105,
        "cards_o35": cards_o35, "cards_u55": cards_u55,
        "top_infravalorado": candidatos[0]
    }

# ==============================================================================
# 2. BARRA LATERAL (PARÁMETROS DE ENTRADA)
# ==============================================================================
st.sidebar.title("⚙️ Parámetros Cuantitativos")

home_team = st.sidebar.text_input("Equipo Local", "Montevideo City Torque")
away_team = st.sidebar.text_input("Equipo Visitante", "CA Tigre BA")

st.sidebar.subheader("⚽ Proyección xG")
xg_home = st.sidebar.number_input("xG Local", value=0.89, step=0.05, format="%.2f")
xg_away = st.sidebar.number_input("xG Visitante", value=1.06, step=0.05, format="%.2f")

st.sidebar.subheader("📊 Cuotas del Mercado (1X2)")
odd_1 = st.sidebar.number_input("Cuota Local (1)", value=3.40, step=0.05)
odd_x = st.sidebar.number_input("Cuota Empate (X)", value=3.10, step=0.05)
odd_2 = st.sidebar.number_input("Cuota Visitante (2)", value=2.25, step=0.05)

st.sidebar.subheader("🚩 Promedios Auxiliares")
exp_corners = st.sidebar.number_input("Expectativa Córners", value=9.1, step=0.1)
exp_cards = st.sidebar.number_input("Expectativa Tarjetas", value=4.0, step=0.1)

# Bankroll para Kelly
bankroll = st.sidebar.number_input("Bankroll Disponible ($)", value=1000.0, step=50.0)

# ==============================================================================
# 3. PROCESAMIENTO Y CÁLCULOS
# ==============================================================================
# Datos de Mercado y Shin
odds_list = [odd_1, odd_x, odd_2]
p_shin, z_insider = desmarginar_shin(odds_list)

# Modelo del Usuario
sec_data = calcular_mercados_secundarios(xg_home, xg_away, exp_corners, exp_cards)
m_p1, m_px, m_p2 = sec_data['p1'], sec_data['px'], sec_data['p2']
matrix = sec_data['matrix']

# Análisis de Valor (+EV) sobre 1X2
ev_1 = (m_p1 * odd_1) - 1
ev_x = (m_px * odd_x) - 1
ev_2 = (m_p2 * odd_2) - 1

evs = [ev_1, ev_x, ev_2]
best_market_idx = np.argmax(evs)
max_ev = evs[best_market_idx]

# Criterio Kelly (Fraccionado a 1/4)
b_odd = odds_list[best_market_idx] - 1
p_win = [m_p1, m_px, m_p2][best_market_idx]
q_loss = 1.0 - p_win
kelly_full = (b_odd * p_win - q_loss) / b_odd if b_odd > 0 else 0
kelly_quarter = max(0.0, kelly_full * 0.25)
suggested_stake = bankroll * kelly_quarter

# ==============================================================================
# 4. RENDERIZADO DE LA INTERFAZ
# ==============================================================================
st.title(f"🏟️ {home_team} vs {away_team}")
st.caption("Evaluación Cuantitativa y Filtrado Anti-Trampas de Mercado")

# --- PANEL SUPERIOR: MERCADOS SECUNDARIOS E INFRAVALORADOS ---
st.markdown("### 🎯 Panel de Mercados Probables & Opciones Infravaloradas")

st.markdown("""
    <style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .badge-value {
        background-color: #064E3B;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

btts_label = "SÍ" if sec_data['btts_yes'] > sec_data['btts_no'] else "NO"
btts_prob = max(sec_data['btts_yes'], sec_data['btts_no']) * 100
with col_s1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚽ Ambos Anotan (BTTS)</div>
            <div class="metric-value">{btts_label} ({btts_prob:.1f}%)</div>
            <div class="metric-sub">Cuota Justa: <b>@{100/btts_prob:.2f}</b></div>
        </div>
    """, unsafe_allow_html=True)

if sec_data['corners_o85'] >= 0.65:
    corn_text, corn_prob = "Over 8.5", sec_data['corners_o85'] * 100
else:
    corn_text, corn_prob = "Under 10.5", sec_data['corners_u105'] * 100
with col_s2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🚩 Tiros de Esquina</div>
            <div class="metric-value">{corn_text}</div>
            <div class="metric-sub">Prob: <b>{corn_prob:.1f}%</b> | Cuota: <b>@{100/corn_prob:.2f}</b></div>
        </div>
    """, unsafe_allow_html=True)

if sec_data['cards_o35'] >= 0.60:
    card_text, card_prob = "Over 3.5", sec_data['cards_o35'] * 100
else:
    card_text, card_prob = "Under 5.5", sec_data['cards_u55'] * 100
with col_s3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟨 Tarjetas Totales</div>
            <div class="metric-value">{card_text}</div>
            <div class="metric-sub">Prob: <b>{card_prob:.1f}%</b> | Cuota: <b>@{100/card_prob:.2f}</b></div>
        </div>
    """, unsafe_allow_html=True)

top_nombre, top_prob, top_cuota = sec_data['top_infravalorado']
with col_s4:
    st.markdown(f"""
        <div class="metric-card" style="border-color: #059669;">
            <div class="metric-title" style="color: #34D399;">💎 Selección Alta Cobertura</div>
            <div class="metric-value" style="color: #F8FAFC; font-size: 1.05rem;">{top_nombre}</div>
            <div class="metric-sub">Prob: <b>{top_prob*100:.1f}%</b> <span class="badge-value">@{top_cuota}</span></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- INDICADORES CUANTITATIVOS SECUNDARIOS ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("EV HÁNDICAP", f"{((m_p1 - m_p2)*100):.1f}%")
col_m2.metric("EV MERCADO 1X2", f"{max_ev*100:+.1f}%", delta_color="normal" if max_ev > 0 else "inverse")
col_m3.metric("xG IMPLÍCITOS", f"{xg_home:.2f} / {xg_away:.2f}")
col_m4.metric("SHIN (INSIDER Z)", f"{z_insider:.4f}")

# --- BLOQUE DE SEGURIDAD Y EJECUCIÓN ---
UMBRAL_EV = 0.025 # 2.5% mínimo
if max_ev < UMBRAL_EV:
    st.warning(f"⚠️ **ABSTENERSE / BLOQUEO DE SEGURIDAD**: No existe ventaja financiera (+EV de {max_ev*100:.1f}% es inferior al umbral mínimo del {UMBRAL_EV*100:.1f}%).")
    suggested_stake = 0.0
else:
    st.success(f"🚀 **VENTAJA ENCONTRADA**: Valor detectado del {max_ev*100:.1f}%.")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
nombres_mkt = [home_team, "Empate (X)", away_team]
col_e1.metric("Selección Recomendada", nombres_mkt[best_market_idx] if max_ev >= UMBRAL_EV else "Ninguna (Bloqueado)")
col_e2.metric("Cuota Justa / Mercado", f"{1/p_win:.2f} / {odds_list[best_market_idx]}")
col_e3.metric("Esperanza (+EV)", f"{max_ev*100:+.2f}%")
col_e4.metric("Stake Sugerido (Kelly 1/4)", f"${suggested_stake:.1f} ({kelly_quarter*100:.1f}%)")

st.markdown("---")

# ==============================================================================
# 5. VISUALIZACIONES (GRÁFICOS DE ALTO CONTRASTE)
# ==============================================================================
col_g1, col_g2 = st.columns(2)

home_short = home_team[:3].upper() if len(home_team) > 3 else home_team.upper()
away_short = away_team[:3].upper() if len(away_team) > 3 else away_team.upper()

with col_g1:
    st.markdown("##### 📊 Comparativa de Probabilidades (1X2)")
    categories = [f"{home_short} (1)", "EMP (X)", f"{away_short} (2)"]
    max_y = max(m_p1, m_px, m_p2, p_shin[0], p_shin[1], p_shin[2]) * 100
    
    fig_bar = go.Figure(data=[
        go.Bar(
            name='Tu Modelo',
            x=categories,
            y=[m_p1*100, m_px*100, m_p2*100],
            marker=dict(color='#10B981', line=dict(color='#059669', width=1)),
            text=[f"{m_p1*100:.1f}%", f"{m_px*100:.1f}%", f"{m_p2*100:.1f}%"],
            textposition='outside',
            textfont=dict(size=11, color='#F8FAFC', family='JetBrains Mono')
        ),
        go.Bar(
            name='Mercado Shin',
            x=categories,
            y=[p_shin[0]*100, p_shin[1]*100, p_shin[2]*100],
            marker=dict(color='#6366F1', line=dict(color='#4F46E5', width=1)),
            text=[f"{p_shin[0]*100:.1f}%", f"{p_shin[1]*100:.1f}%", f"{p_shin[2]*100:.1f}%"],
            textposition='outside',
            textfont=dict(size=11, color='#93C5FD', family='JetBrains Mono')
        )
    ])
    fig_bar.update_layout(
        barmode='group',
        height=310,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.1, x=1, font=dict(size=11, color="#E2E8F0")),
        font=dict(color="#E2E8F0"),
        xaxis=dict(fixedrange=True, tickfont=dict(size=11, color="#94A3B8")),
        yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#1E293B', showticklabels=False, range=[0, max_y * 1.25])
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

with col_g2:
    st.markdown("##### 🔥 Matriz de Marcadores (Tu Modelo)")
    df_m = np.round(matrix[:5, :5] * 100, 1)
    
    # Paleta de Alto Contraste Dark Slate -> Sky Blue
    custom_colorscale = [
        [0.0, "#0F172A"],
        [0.25, "#1E293B"],
        [0.50, "#0369A1"],
        [0.75, "#0284C7"],
        [1.0, "#38BDF8"]
    ]
    
    fig_hm = go.Figure(data=go.Heatmap(
        z=df_m,
        x=[f"{i}" for i in range(5)],
        y=[f"{i}" for i in range(5)],
        colorscale=custom_colorscale,
        showscale=False,
        xgap=3, ygap=3,
        text=df_m,
        texttemplate="%{text:.1f}%",
        textfont=dict(size=11, family='JetBrains Mono', color="#F8FAFC")
    ))

    fig_hm.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0"),
        xaxis=dict(title=dict(text=f"Goles {away_short}", font=dict(size=11, color="#94A3B8")), tickfont=dict(color="#E2E8F0")),
        yaxis=dict(title=dict(text=f"Goles {home_short}", font=dict(size=11, color="#94A3B8")), tickfont=dict(color="#E2E8F0"), autorange="reversed")
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
