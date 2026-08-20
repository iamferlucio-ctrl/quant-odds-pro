import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import requests

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="CuantiBet Hybrid Engine v3.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #060911; }
    
    .hero-card {
        background: linear-gradient(180deg, #0F172A 0%, #0B1120 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 1.15rem !important;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .hero-sub {
        font-size: 0.65rem;
        color: #64748B;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-weight: 700;
    }
    
    .section-title {
        font-size: 0.78rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin: 14px 0 8px 0;
    }
    
    .grid-2x2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 10px;
    }
    
    .dash-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 88px;
    }
    .dash-card-highlight {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        border: 1px solid #6366F1;
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 88px;
    }
    .dash-label {
        font-size: 0.62rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .dash-value {
        font-size: 1.0rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 3px 0 2px 0;
        line-height: 1.2;
    }
    .dash-sub {
        font-size: 0.68rem;
        font-weight: 600;
        color: #38BDF8;
    }
    
    .analysis-box {
        background: #0B132B;
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
        padding: 12px;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #CBD5E1;
        line-height: 1.4;
    }
    
    .metric-card {
        background: #0D1527;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.60rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
    }
    .metric-val-neg {
        font-size: 1.05rem;
        font-weight: 800;
        color: #EF4444;
        font-family: monospace;
    }
    .metric-val-pos {
        font-size: 1.05rem;
        font-weight: 800;
        color: #10B981;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTOR MATH (DIXON-COLES, SHIN & BINOMIAL NEGATIVA)
# ==============================================================================

def desmarginar_shin(odds, max_iter=100, tol=1e-6):
    odds = np.array(odds, dtype=float)
    if np.any(odds <= 1.0): return np.array([1/3, 1/3, 1/3]), 0.0
    implied = 1.0 / odds
    beta = np.sum(implied)
    if abs(beta - 1.0) < 1e-5: return implied, 0.0
    z = 0.0
    for _ in range(max_iter):
        f = np.sum(np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta)) - 2.0
        if abs(f) < tol: break
        f_prime = np.sum((z - 2 * (implied**2) / beta) / np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta))
        if f_prime == 0: break
        z = z - f / f_prime
        z = max(0.0, min(0.99, z))
    p_shin = (np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / beta) - z) / (2 * (1 - z))
    return p_shin / np.sum(p_shin), z

def dixon_coles_tau(x, y, lambda_h, mu_a, rho=-0.11):
    if x == 0 and y == 0: return 1.0 - (lambda_h * mu_a * rho)
    elif x == 0 and y == 1: return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 0: return 1.0 + (mu_a * rho)
    elif x == 1 and y == 1: return 1.0 - rho
    else: return 1.0

def calcular_matriz_bivariada(lambda_h, mu_a, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_i = stats.poisson.pmf(i, lambda_h)
            p_j = stats.poisson.pmf(j, mu_a)
            tau = dixon_coles_tau(i, j, lambda_h, mu_a)
            matrix[i, j] = p_i * p_j * tau
            
    matrix = matrix / np.sum(matrix)
    p1 = float(np.sum(np.tril(matrix, -1)))
    px = float(np.sum(np.diag(matrix)))
    p2 = float(np.sum(np.triu(matrix, 1)))
    return matrix, p1, px, p2

def cdf_neg_binomial(k, mean, dispersion=1.25):
    r = mean / (dispersion - 1.0) if dispersion > 1.0 else 10.0
    p = r / (r + mean)
    return stats.nbinom.cdf(k, r, p)

def calcular_mercados_alternativos(lambda_h, mu_a, corners_avg, cards_avg, p1, px, p2):
    matrix, _, _, _ = calcular_matriz_bivariada(lambda_h, mu_a)
    p_btts_no = np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0]
    p_btts_yes = 1.0 - p_btts_no
    
    line_corners = 9.5
    prob_under_corners = float(cdf_neg_binomial(line_corners, corners_avg, dispersion=1.35))
    prob_over_corners = 1.0 - prob_under_corners
    
    line_cards = 4.5
    prob_under_cards = float(cdf_neg_binomial(line_cards, cards_avg, dispersion=1.45))
    prob_over_cards = 1.0 - prob_under_cards
    
    candidatos = [
        ("Doble Oportunidad 1X", p1 + px, 1.0 / max(0.01, p1 + px)),
        ("Doble Oportunidad X2", p2 + px, 1.0 / max(0.01, p2 + px)),
        ("BTTS No", p_btts_no, 1.0 / max(0.01, p_btts_no)),
        ("Under 9.5 Córners", prob_under_corners, 1.0 / max(0.01, prob_under_corners)),
        ("Over 4.5 Tarjetas", prob_over_cards, 1.0 / max(0.01, prob_over_cards))
    ]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "btts": ("SÍ", p_btts_yes, 1/p_btts_yes) if p_btts_yes >= 0.50 else ("NO", p_btts_no, 1/p_btts_no),
        "corners": (f"Under {line_corners}", prob_under_corners, 1/prob_under_corners) if prob_under_corners >= 0.50 else (f"Over {line_corners}", prob_over_corners, 1/prob_over_corners),
        "cards": (f"Under {line_cards}", prob_under_cards, 1/prob_under_cards) if prob_under_cards >= 0.50 else (f"Over {line_cards}", prob_over_cards, 1/prob_over_cards),
        "cobertura": candidatos[0]
    }

# ==============================================================================
# CONEXIONES API (API-FOOTBALL Y THE ODDS API)
# ==============================================================================

def fetch_odds_api(api_key, sport="soccer_epl"):
    clean_key = api_key.strip()
    if not clean_key: return None, "Key vacía"
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {"apiKey": clean_key, "regions": "eu", "markets": "h2h", "dateFormat": "iso"}
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            return res.json(), None
        return None, f"Status {res.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_apifootball_data(api_key):
    clean_key = api_key.strip()
    if not clean_key: return None, "Key vacía"
    url = "https://v3.football.api-sports.io/fixtures?next=10"
    headers = {"x-apisports-key": clean_key}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            return res.json().get("response", []), None
        return None, f"Status {res.status_code}"
    except Exception as e:
        return None, str(e)

# ==============================================================================
# BARRA LATERAL HÍBRIDA
# ==============================================================================

st.sidebar.title("🎛️ Motor Híbrido Dual-API")

key_odds_api = st.sidebar.text_input("🔑 The Odds API Key", type="password", help="Aporta cuotas de mercado en tiempo real")
key_apifootball = st.sidebar.text_input("🔑 API-Football Key", type="password", help="Aporta xG y métricas de rendimiento")

# Base local de respaldo
DATABASE_BACKUP = {
    "CSD Macará vs Santos FC": {"home": "CSD Macará", "away": "Santos FC", "xg_home": 1.25, "xg_away": 1.15, "odd_1": 2.60, "odd_x": 3.10, "odd_2": 2.75, "corners": 9.2, "cards": 5.4},
    "LDU Quito vs Lanús": {"home": "LDU Quito", "away": "Lanús", "xg_home": 1.65, "xg_away": 0.85, "odd_1": 1.95, "odd_x": 3.30, "odd_2": 4.10, "corners": 10.2, "cards": 5.2},
    "Real Madrid vs Manchester City": {"home": "Real Madrid", "away": "Manchester City", "xg_home": 1.70, "xg_away": 1.55, "odd_1": 2.45, "odd_x": 3.50, "odd_2": 2.75, "corners": 10.2, "cards": 4.2}
}

match_data = None
source_status = []

# Procesar The Odds API
odds_events, err_odds = None, None
if key_odds_api:
    odds_events, err_odds = fetch_odds_api(key_odds_api)
    if not err_odds and odds_events:
        source_status.append("🟢 Odds API: Mercado 1X2 Conectado")
    else:
        source_status.append("🔴 Odds API: Fallo de Autenticación")

# Procesar API-Football
apif_events, err_apif = None, None
if key_apifootball:
    apif_events, err_apif = fetch_apifootball_data(key_apifootball)
    if not err_apif and apif_events:
        source_status.append("🟢 API-Football: Métricas xG Conectadas")
    else:
        source_status.append("🔴 API-Football: Fallo de Autenticación")

# Construcción de la lista desplegable unificada
if odds_events:
    dict_matches = {}
    for ev in odds_events:
        h, a = ev.get('home_team', 'Home'), ev.get('away_team', 'Away')
        o1, ox, o2 = 2.10, 3.20, 3.40
        if ev.get('bookmakers'):
            for m in ev['bookmakers'][0].get('markets', []):
                if m.get('key') == 'h2h':
                    outcomes = {o['name']: o['price'] for o in m.get('outcomes', [])}
                    o1, ox, o2 = outcomes.get(h, 2.10), outcomes.get('Draw', 3.20), outcomes.get(a, 3.40)
        
        dict_matches[f"{h} vs {a}"] = {
            "home": h, "away": a,
            "xg_home": 1.50, "xg_away": 1.20, # xG base por defecto si no coincide
            "odd_1": o1, "odd_x": ox, "odd_2": o2,
            "corners": 9.5, "cards": 4.5
        }
    
    selected = st.sidebar.selectbox("📅 Partidos Disponibles", list(dict_matches.keys()))
    match_data = dict_matches[selected]

else:
    selected = st.sidebar.selectbox("📅 Partidos (Modo Local/Respaldo)", list(DATABASE_BACKUP.keys()))
    match_data = DATABASE_BACKUP[selected]

for st_msg in source_status:
    st.sidebar.caption(st_msg)

# ==============================================================================
# EJECUCIÓN Y RENDERIZADO
# ==============================================================================

home_team = match_data["home"]
away_team = match_data["away"]
xg_h = match_data["xg_home"]
xg_a = match_data["xg_away"]
odd_1 = match_data["odd_1"]
odd_x = match_data["odd_x"]
odd_2 = match_data["odd_2"]

matrix, p1, px, p2 = calcular_matriz_bivariada(xg_h, xg_a)
p_shin, z_val = desmarginar_shin([odd_1, odd_x, odd_2])

ev_1 = (p1 * odd_1) - 1.0
ev_x = (px * odd_x) - 1.0
ev_2 = (p2 * odd_2) - 1.0

probs_1x2 = [p1, px, p2]
names_1x2 = [f"Victoria {home_team}", "Empate (X)", f"Victoria {away_team}"]
best_scen_idx = int(np.argmax(probs_1x2))

max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
score_str = f"{max_pos[0]} - {max_pos[1]}"
score_prob = matrix[max_pos] * 100

prob_under_25 = sum(matrix[i, j] for i in range(3) for j in range(3) if i + j <= 2) * 100
alt_markets = calcular_mercados_alternativos(xg_h, xg_a, match_data["corners"], match_data["cards"], p1, px, p2)

st.markdown(f'''
    <div class="hero-card">
        <div class="hero-title">🏟️ {home_team} vs {away_team}</div>
        <div class="hero-sub">Motor Híbrido Quant (Estadísticas xG + Cuotas Reales de Mercado)</div>
    </div>
''', unsafe_allow_html=True)

st.markdown('<div class="section-title">🎯 PANEL DE MERCADOS & CUOTAS JUSTAS TEÓRICAS</div>', unsafe_allow_html=True)

st.markdown(f'''
    <div class="grid-2x2">
        <div class="dash-card">
            <div class="dash-label">⚽ AMBOS ANOTAN (BTTS)</div>
            <div class="dash-value">{alt_markets["btts"][0]} ({alt_markets["btts"][1]*100:.1f}%)</div>
            <div class="dash-sub">Cuota Justa: @{alt_markets["btts"][2]:.2f}</div>
        </div>
        <div class="dash-card">
            <div class="dash-label">🚩 TIROS DE ESQUINA (NEG-BINOM)</div>
            <div class="dash-value">{alt_markets["corners"][0]}</div>
            <div class="dash-sub">Prob: {alt_markets["corners"][1]*100:.1f}% | Fair @{alt_markets["corners"][2]:.2f}</div>
        </div>
        <div class="dash-card">
            <div class="dash-label">🟨 TARJETAS (SOBREDISPERSIÓN)</div>
            <div class="dash-value">{alt_markets["cards"][0]}</div>
            <div class="dash-sub">Prob: {alt_markets["cards"][1]*100:.1f}% | Fair @{alt_markets["cards"][2]:.2f}</div>
        </div>
        <div class="dash-card-highlight">
            <div class="dash-label">💎 COBERTURA MÁS ROBUSTA</div>
            <div class="dash-value">{alt_markets["cobertura"][0]}</div>
            <div class="dash-sub">Éxito Estocástico: {alt_markets["cobertura"][1]*100:.1f}%</div>
        </div>
    </div>
''', unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">MARGINALIDAD SHIN (Z-INDEX)</div>
            <div class="metric-val-pos">{z_val*100:.2f}%</div>
        </div>
    ''', unsafe_allow_html=True)

with col_m2:
    best_ev = max(ev_1, ev_x, ev_2)
    ev_class = "metric-val-pos" if best_ev > 0.0 else "metric-val-neg"
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">EV MÁXIMO MERCADO 1X2</div>
            <div class="{ev_class}">{best_ev*100:+.1f}%</div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 COMPARATIVA MODELO VS SHIN Y MATRIZ BIVARIADA</div>', unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)
h_short = home_team[:3].upper() if len(home_team) >= 3 else home_team.upper()
a_short = away_team[:3].upper() if len(away_team) >= 3 else away_team.upper()

with col_g1:
    fig_bar = go.Figure(data=[
        go.Bar(name='Modelo DC', x=[h_short, "EMP", a_short], y=[p1*100, px*100, p2*100], marker_color='#38BDF8', texttemplate='%{y:.1f}%', textposition='inside'),
        go.Bar(name='Shin Implícito', x=[h_short, "EMP", a_short], y=[p_shin[0]*100, p_shin[1]*100, p_shin[2]*100], marker_color='#6366F1', texttemplate='%{y:.1f}%', textposition='inside')
    ])
    fig_bar.update_layout(
        barmode='group',
        height=250,
        margin=dict(l=10, r=10, t=15, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0", size=10),
        showlegend=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, showgrid=False)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

with col_g2:
    fig_hm = go.Figure(data=go.Heatmap(
        z=np.round(matrix[:4, :4] * 100, 1),
        x=["0", "1", "2", "3"],
        y=["0", "1", "2", "3"],
        colorscale=[[0, "#0F172A"], [1, "#0284C7"]],
        showscale=False,
        text=np.round(matrix[:4, :4] * 100, 1),
        texttemplate="%{text:.1f}%"
    ))
    fig_hm.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=15, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0", size=10),
        xaxis=dict(title="Goles Visita", fixedrange=True),
        yaxis=dict(title="Goles Local", autorange="reversed", fixedrange=True)
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
