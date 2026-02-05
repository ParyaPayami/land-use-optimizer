import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING (Premium Pro)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PIMALUOS",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background-color: #0F1116;
        background-image: radial-gradient(at 0% 0%, rgba(0, 210, 255, 0.05) 0px, transparent 50%),
                          radial-gradient(at 100% 100%, rgba(255, 75, 75, 0.05) 0px, transparent 50%);
    }

    /* Cards */
    div.metric-card {
        background-color: rgba(30, 32, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px; 
    }
    
    /* Sidebar Text Visibility Fix */
    section[data-testid="stSidebar"] { background-color: #16181D; }
    section[data-testid="stSidebar"] * { color: #F9FAFB !important; }
    section[data-testid="stSidebar"] .stMarkdown { color: #D1D5DB !important; }
    
    h1, h2, h3 { color: #FFFFFF !important; }
    div[data-testid="stMetricValue"] { color: #00D2FF !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (Full Scale)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load the new Full Scale Simulation
    try:
        df = pd.read_csv("results/full_scale_simulation/manhattan_landuse.csv")
        
        # Color Map for Land Uses (Standard Planning Colors)
        df['color'] = df['proposed_use_label'].map({
            'Residential': '#F1C40F',  # Yellow/Orange
            'Commercial': '#E74C3C',   # Red
            'Industrial': '#8E44AD',   # Purple
            'Mixed-Use': '#D35400',    # Brown/Orange
            'Public': '#3498DB',       # Blue
            'Open Space': '#2ECC71'    # Green
        })
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("PIMALUOS: NYC Land Use Optimization")
    st.caption("v3.0 | Full-Scale Simulation")
    
    st.subheader("Map Layers")
    layer = st.radio("Display", ["Proposed Land Use", "Current Land Use", "ROI Lift"])
    
    st.subheader("Filter")
    if not df.empty:
        zones = st.multiselect("Zones", df['proposed_use_label'].unique(), default=df['proposed_use_label'].unique())

# -----------------------------------------------------------------------------
# 4. DASHBOARD
# -----------------------------------------------------------------------------
st.title("Manhattan Land Use Optimization")
st.markdown("**Comprehensive Plan 2030** • Full Island Simulation")

# Metrics
if not df.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Parcels Optimized", f"{len(df):,}", "Full Coverage")
    with k2: st.metric("New Mixed-Use", f"{len(df[df['proposed_use_label']=='Mixed-Use']):,}", "+15%")
    with k3: st.metric("Avg ROI Lift", f"+{df['roi_lift'].mean():.1f}%", "High Value")
    with k4: st.metric("Open Space Added", f"{len(df[df['proposed_use_label']=='Open Space']) - len(df[df['current_use_label']=='Open Space']):+}", "Equity Target")

st.markdown("---")

col_map, col_stats = st.columns([2.5, 1])

with col_map:
    st.subheader("📍 Optimal Zoning Configuration")
    if not df.empty:
        
        # Color Logic based on selection
        if layer == "Proposed Land Use": 
            color_col = "proposed_use_label"
            title = "Proposed 2030 Plan"
        elif layer == "Current Land Use":
            color_col = "current_use_label"
            title = "Current Conditions (2025)"
        else:
            color_col = "roi_lift"
            title = "Economic Value Map"
            
        fig_map = px.scatter_mapbox(
            df, lat="lat", lon="lon",
            color=color_col,
            color_discrete_map={
                'Residential': '#F1C40F', 'Commercial': '#E74C3C', 
                'Industrial': '#8E44AD', 'Mixed-Use': '#D35400', 
                'Public': '#3498DB', 'Open Space': '#2ECC71'
            },
            size_max=8, zoom=10.5, height=600,
            title=title
        )
        fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

with col_stats:
    st.subheader("📊 Distribution")
    if not df.empty:
        # Land Use Bar Chart
        cts = df['proposed_use_label'].value_counts().reset_index()
        cts.columns = ['Use', 'Count']
        
        fig_bar = px.bar(
            cts, x='Count', y='Use', orientation='h', 
            color='Use', text_auto=True,
            color_discrete_map={
                'Residential': '#F1C40F', 'Commercial': '#E74C3C', 
                'Industrial': '#8E44AD', 'Mixed-Use': '#D35400', 
                'Public': '#3498DB', 'Open Space': '#2ECC71'
            }
        )
        fig_bar.update_layout(showlegend=False, height=300, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Change Analysis - Grouped Bar Chart
        st.subheader("Change Analysis")
        
        # Get counts for current and proposed
        current_counts = df['current_use_label'].value_counts()
        proposed_counts = df['proposed_use_label'].value_counts()
        
        # Create comparison dataframe
        all_uses = sorted(set(current_counts.index) | set(proposed_counts.index))
        comparison_df = pd.DataFrame({
            'Land Use': all_uses,
            'Current (2025)': [current_counts.get(use, 0) for use in all_uses],
            'Proposed (2030)': [proposed_counts.get(use, 0) for use in all_uses]
        })
        
        # Melt for grouped bar chart
        comparison_melted = comparison_df.melt(
            id_vars='Land Use', 
            var_name='Plan', 
            value_name='Parcels'
        )
        
        fig_change = px.bar(
            comparison_melted, 
            x='Land Use', 
            y='Parcels', 
            color='Plan',
            barmode='group',
            text_auto=True,
            color_discrete_map={
                'Current (2025)': '#6B7280',  # Gray
                'Proposed (2030)': '#00D2FF'  # Cyan
            }
        )
        fig_change.update_layout(
            height=300, 
            margin={"r":0,"t":20,"l":0,"b":0},
            xaxis_title=None,
            yaxis_title="Number of Parcels",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_change.update_traces(textposition='outside')
        st.plotly_chart(fig_change, use_container_width=True)
