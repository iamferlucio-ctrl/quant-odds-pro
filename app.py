import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# ==============================================================================
# 1. MOTOR EXTENDIDO DE CÁLCULO CUANTITATIVO (MERCADOS SECUNDARIOS)
# ==============================================================================

def calcular_mercados_secundarios(lambda_home, mu_away, exp_corners=9.2, exp_cards=4.2):
    """
    Calcula probabilidades estocásticas para mercados secundarios e infravalorados
    usando distribuciones de Poisson y matrices de dependencia.
    """
    # Matriz 7x7 de marcadores (hasta 6 goles por equipo)
    max_g = 7
    p_home = stats.poisson.pmf(np.arange(max_g), lambda_home)
    p_away = stats.poisson.pmf(np.arange(max_g), mu_away)
    matrix = np.outer(p_home, p_away)
    
    # 1X2 Proyecciones
    p1 = float(np.sum(np.tril(matrix, -1)))
    px = float(np.sum(np.diag(matrix)))
    p2 = float(np.sum(np.triu(matrix, 1)))
    
    # Ambos Anotan (BTTS)
    btts_yes = float(np.sum(matrix[1:, 1:]))
    btts_no = 1.0 - btts_yes
    
    # Doble Oportunidad
    dc_1x = p1 + px
    dc_x2 = px + p2
    dc_12 = p1 + p2
    
    # Goles en 1ra Mitad (Proporción histórica ~44% del xG total en 1HT)
    lambda_ht = lambda_home * 0.44
    mu_ht = mu_away * 0.44
    over_05_ht = 1.0 - (stats.poisson.pmf(0, lambda_ht) * stats.poisson.pmf(0, mu_ht))
    
    # Tiros de Esquina (Modelo Poisson sobre expectativa de córners)
    corners_o85 = 1.0 - stats.poisson.cdf(8, exp_corners)
    corners_o95 = 1.0 - stats.poisson.cdf(9, exp_corners)
    corners_u105 = stats.poisson.cdf(10, exp_corners)
    
    # Tarjetas (Modelo Poisson sobre expectativa de disciplina)
    cards_o35 = 1.0 - stats.poisson.cdf(3, exp_cards)
    cards_o45 = 1.0 - stats.poisson.cdf(4, exp_cards)
    cards_u55 = stats.poisson.cdf(5, exp_cards)
    
    # Selección del mercado secundario más eficiente ("Infravalorado")
    candidatos = [
        ("Doble Oportunidad 1X", dc_1x, f"{1/dc_1x:.2f}"),
        ("Doble Oportunidad X2", dc_x2, f"{1/dc_x2:.2f}"),
        ("BTTS - Sí", btts_yes, f"{1/btts_yes:.2f}"),
        ("BTTS - No", btts_no, f"{1/btts_no:.2f}"),
        ("Over 0.5 Goles 1HT", over_05_ht, f"{1/over_05_ht:.2f}"),
        ("Over 8.5 Córners", corners_o85, f"{1/corners_o85:.2f}"),
        ("Over 3.5 Tarjetas", cards_o35, f"{1/cards_o35:.2f}")
    ]
    # Ordenar por mayor probabilidad matemática
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "dc_1x": dc_1x,
        "dc_x2": dc_x2,
        "over_05_ht": over_05_ht,
        "corners_o85": corners_o85,
        "corners_o95": corners_o95,
        "corners_u105": corners_u105,
        "cards_o35": cards_o35,
        "cards_o45": cards_o45,
        "cards_u55": cards_u55,
        "top_infravalorado": candidatos[0]
    }

# ==============================================================================
# 2. PANEL SUPERIOR DE MERCADOS MÁS PROBABLES E INFRAVALORADOS (UI)
# ==============================================================================

def render_panel_mercados_probables(m_secundarios, home_team, away_team):
    st.markdown("### 🎯 Panel de Mercados Probables & Opciones Infravaloradas")
    
    # CSS personalizado para tarjetas oscuras de alto contraste
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
            font-size: 1.35rem;
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

    col1, col2, col3, col4 = st.columns(4)

    # CARD 1: AMBOS ANOTAN (BTTS)
    btts_label = "SÍ" if m_secundarios['btts_yes'] > m_secundarios['btts_no'] else "NO"
    btts_prob = max(m_secundarios['btts_yes'], m_secundarios['btts_no']) * 100
    cuota_justa_btts = 100 / btts_prob
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">⚽ Ambos Anotan (BTTS)</div>
                <div class="metric-value">{btts_label} ({btts_prob:.1f}%)</div>
                <div class="metric-sub">Cuota Justa: <b>@{cuota_justa_btts:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    # CARD 2: TIROS DE ESQUINA
    if m_secundarios['corners_o85'] >= 0.65:
        corn_text = "Over 8.5"
        corn_prob = m_secundarios['corners_o85'] * 100
    else:
        corn_text = "Under 10.5"
        corn_prob = m_secundarios['corners_u105'] * 100
    cuota_justa_corn = 100 / corn_prob
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🚩 Tiros de Esquina</div>
                <div class="metric-value">{corn_text}</div>
                <div class="metric-sub">Prob: <b>{corn_prob:.1f}%</b> | Cuota: <b>@{cuota_justa_corn:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    # CARD 3: DISCIPLINA / TARJETAS
    if m_secundarios['cards_o35'] >= 0.60:
        card_text = "Over 3.5"
        card_prob = m_secundarios['cards_o35'] * 100
    else:
        card_text = "Under 5.5"
        card_prob = m_secundarios['cards_u55'] * 100
    cuota_justa_cards = 100 / card_prob
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🟨 Tarjetas Totales</div>
                <div class="metric-value">{card_text}</div>
                <div class="metric-sub">Prob: <b>{card_prob:.1f}%</b> | Cuota: <b>@{cuota_justa_cards:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)

    # CARD 4: OPCIÓN MÁS INFRAVALORADA / SEGURA
    top_nombre, top_prob, top_cuota = m_secundarios['top_infravalorado']
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="border-color: #059669;">
                <div class="metric-title" style="color: #34D399;">💎 Selección de Alta Cobertura</div>
                <div class="metric-value" style="color: #F8FAFC; font-size: 1.1rem;">{top_nombre}</div>
                <div class="metric-sub">Prob: <b>{top_prob*100:.1f}%</b> <span class="badge-value">@{top_cuota}</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


# ==============================================================================
# 3. EJEMPLO DE INTEGRACIÓN EN EL FLUJO PRINCIPAL
# ==============================================================================

# Entradas simuladas de xG y Parámetros del Partido
home_team = "Montevideo City Torque"
away_team = "CA Tigre BA"
xg_home = 0.89  # Proyección xG Local
xg_away = 1.06  # Proyección xG Visitante

# Ejecución del cálculo del motor
m_secundarios = calcular_mercados_secundarios(
    lambda_home=xg_home, 
    mu_away=xg_away, 
    exp_corners=9.1, 
    exp_cards=4.0
)

# Renderizado del nuevo panel al inicio de la app
render_panel_mercados_probables(m_secundarios, home_team, away_team)
