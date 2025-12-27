import streamlit as st
import plotly.graph_objects as go

# === IMPORT YOUR BACKEND (NO LOGIC CHANGE) ===
# Adjust the import line ONLY to match your file name
from backend_engine import (
    price_surface,
    delta_surface,
    gamma_surface,
    vega_surface,
    theta_surface
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Options Surface Lab",
    layout="wide"
)

# ===============================
# GLOBAL DARK STYLE
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e6e6e6;
}
[data-testid="stSidebar"] {
    background-color: #0b0f14;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.title("Controls")

option_type = st.sidebar.selectbox(
    "Option Type",
    ["call", "put"]
)

vol_mult = st.sidebar.slider(
    "Volatility Multiplier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05
)

# ===============================
# MAIN HEADER
# ===============================
st.markdown("## Options Surface Analytics")

# ===============================
# TABS
# ===============================
tab_price, tab_greeks = st.tabs(["Price Surface", "Greeks"])

# ===============================
# PRICE SURFACE TAB
# ===============================
with tab_price:

    spot, vol, price = price_surface(option_type, vol_mult)

    fig_price = go.Figure(
        data=[
            go.Surface(
                z=price,
                x=spot,
                y=vol,
                colorscale="Viridis"
            )
        ]
    )

    fig_price.update_layout(
        scene=dict(
            xaxis_title="Spot",
            yaxis_title="Volatility",
            zaxis_title="Option Price",
            bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="#e6e6e6"),
        height=700
    )

    st.plotly_chart(fig_price, use_container_width=True)

# ===============================
# GREEKS TAB
# ===============================
with tab_greeks:

    col1, col2 = st.columns(2)

    with col1:
        spot, vol, delta = delta_surface(option_type, vol_mult)
        fig_delta = go.Figure(
            data=[go.Surface(z=delta, x=spot, y=vol, colorscale="RdBu")]
        )
        fig_delta.update_layout(
            title="Delta",
            scene=dict(bgcolor="#0e1117"),
            paper_bgcolor="#0e1117",
            font=dict(color="#e6e6e6"),
            height=450
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    with col2:
        spot, vol, gamma = gamma_surface(option_type, vol_mult)
        fig_gamma = go.Figure(
            data=[go.Surface(z=gamma, x=spot, y=vol, colorscale="Viridis")]
        )
        fig_gamma.update_layout(
            title="Gamma",
            scene=dict(bgcolor="#0e1117"),
            paper_bgcolor="#0e1117",
            font=dict(color="#e6e6e6"),
            height=450
        )
        st.plotly_chart(fig_gamma, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        spot, vol, vega = vega_surface(option_type, vol_mult)
        fig_vega = go.Figure(
            data=[go.Surface(z=vega, x=spot, y=vol, colorscale="Plasma")]
        )
        fig_vega.update_layout(
            title="Vega",
            scene=dict(bgcolor="#0e1117"),
            paper_bgcolor="#0e1117",
            font=dict(color="#e6e6e6"),
            height=450
        )
        st.plotly_chart(fig_vega, use_container_width=True)

    with col4:
        spot, vol, theta = theta_surface(option_type, vol_mult)
        fig_theta = go.Figure(
            data=[go.Surface(z=theta, x=spot, y=vol, colorscale="Cividis")]
        )
        fig_theta.update_layout(
            title="Theta",
            scene=dict(bgcolor="#0e1117"),
            paper_bgcolor="#0e1117",
            font=dict(color="#e6e6e6"),
            height=450
        )
        st.plotly_chart(fig_theta, use_container_width=True)
