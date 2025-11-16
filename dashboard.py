#!/usr/bin/env python3
"""
Airport Defense Battle Management System - Dashboard MVP
Visualizes drone defense scenarios with population density data.
"""

import streamlit as st
import folium
from folium import plugins
import pickle
import numpy as np
from pathlib import Path
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Page configuration
st.set_page_config(
    page_title="Airport Defense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and military aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00ff41;
        font-family: 'Courier New', monospace;
    }
    .metric-container {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #00ff41;
        margin: 10px 0;
    }
    .metric-label {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
    }
    .metric-value {
        color: #00ff41;
        font-size: 24px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .status-active {
        color: #ff4444;
    }
    .status-ready {
        color: #00ff41;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_airport_data():
    """Load preprocessed airport population data."""
    data_file = Path('airports_data.pkl')

    if not data_file.exists():
        st.error("⚠️ Data files not found! Please run 'python extract_data.py' first.")
        st.stop()

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    return data


def create_population_heatmap_data(pop_data, threshold=10):
    """
    Convert population raster data to heatmap points for Folium.

    Args:
        pop_data: Dictionary containing population data and transform
        threshold: Minimum population to display (reduces clutter)

    Returns:
        List of [lat, lon, intensity] for heatmap
    """
    data = pop_data['data']
    transform = pop_data['transform']

    heatmap_points = []

    # Get dimensions
    rows, cols = data.shape

    # Sample points (every nth point to reduce data size)
    step = max(1, min(rows, cols) // 100)  # Aim for ~100x100 points max

    for row in range(0, rows, step):
        for col in range(0, cols, step):
            population = data[row, col]

            # Only include cells above threshold
            if population > threshold:
                # Convert pixel coordinates to lat/lon
                lon, lat = transform * (col, row)

                # Add to heatmap data [lat, lon, weight]
                heatmap_points.append([lat, lon, float(population)])

    return heatmap_points


def create_folium_map(airport_data, show_heatmap=True, show_airport=True):
    """
    Create Folium map with ESRI satellite tiles and population overlay.

    Args:
        airport_data: Airport data dictionary
        show_heatmap: Whether to show population heatmap
        show_airport: Whether to show airport marker

    Returns:
        Folium map object
    """
    center_lat = airport_data['center_lat']
    center_lon = airport_data['center_lon']

    # Create map with ESRI World Imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,  # We'll add custom tiles
        prefer_canvas=True
    )

    # Add ESRI World Imagery (satellite) tiles
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='ESRI Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Add dark labels overlay for better readability
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Labels',
        overlay=True,
        control=True,
        opacity=0.5
    ).add_to(m)

    # Add population heatmap
    if show_heatmap:
        heatmap_data = create_population_heatmap_data(airport_data, threshold=10)

        if heatmap_data:
            plugins.HeatMap(
                heatmap_data,
                name='Population Density',
                min_opacity=0.3,
                max_opacity=0.8,
                radius=15,
                blur=20,
                gradient={
                    0.0: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime',
                    0.7: 'yellow',
                    0.9: 'orange',
                    1.0: 'red'
                }
            ).add_to(m)

    # Add airport marker
    if show_airport:
        folium.Marker(
            location=[center_lat, center_lon],
            popup=folium.Popup(f"<b>{airport_data['airport_name']}</b><br>{airport_data['airport_type']}", max_width=200),
            tooltip=airport_data['airport_name'],
            icon=folium.Icon(color='green', icon='plane', prefix='fa')
        ).add_to(m)

        # Add 10km radius circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=10000,  # 10km in meters
            color='#00ff41',
            fill=False,
            weight=2,
            opacity=0.5,
            popup='10km Defense Perimeter'
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def render_metrics(airport_data, threats_active=0, defenders_active=0):
    """Render metrics sidebar."""
    st.markdown("### 📊 BATTLE METRICS")

    # Population statistics
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Total Population (10km)</div>
        <div class="metric-value">{int(airport_data['total_population']):,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Peak Density</div>
        <div class="metric-value">{int(airport_data['max_density']):,} /cell</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Avg Density</div>
        <div class="metric-value">{int(airport_data['mean_density']):,} /cell</div>
    </div>
    """, unsafe_allow_html=True)

    # Threat status (placeholder for simulation)
    st.markdown("---")
    st.markdown("### ⚔️ THREAT STATUS")

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Active Threats</div>
        <div class="metric-value status-active">{threats_active}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Defenders Ready</div>
        <div class="metric-value status-ready">{defenders_active}</div>
    </div>
    """, unsafe_allow_html=True)

    # Collateral risk (placeholder)
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Collateral Risk Score</div>
        <div class="metric-value">0.00</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""

    # Header
    st.markdown("# 🛡️ AIRPORT DEFENSE SYSTEM")
    st.markdown("*Battle Management Dashboard - Population-Aware Defense*")

    # Load data
    airports_data = load_airport_data()

    # Sidebar controls
    st.sidebar.markdown("## 🎯 MISSION CONTROL")

    # Airport selection
    airport_choice = st.sidebar.selectbox(
        "Select Target Airport",
        options=['Songshan', 'Taoyuan'],
        format_func=lambda x: airports_data[x]['airport_name']
    )

    selected_airport = airports_data[airport_choice]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ VIEW OPTIONS")

    # View toggles
    show_heatmap = st.sidebar.checkbox("Population Heatmap", value=True)
    show_airport = st.sidebar.checkbox("Airport Marker", value=True)
    show_trajectories = st.sidebar.checkbox("Show Trajectories", value=False, disabled=True)
    show_infrastructure = st.sidebar.checkbox("Infrastructure", value=False, disabled=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ SIMULATION")

    threat_intensity = st.sidebar.slider(
        "Threat Intensity",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of incoming threats"
    )

    start_simulation = st.sidebar.button(
        "🚀 START SIMULATION",
        disabled=True,
        help="Simulation logic coming from partner"
    )

    if start_simulation:
        st.sidebar.success("Simulation started!")

    st.sidebar.markdown("---")

    # Airport info display
    st.sidebar.markdown("### 📍 AIRPORT INFO")
    st.sidebar.markdown(f"**{selected_airport['airport_name']}**")
    st.sidebar.markdown(f"Type: {selected_airport['airport_type']}")
    st.sidebar.markdown(f"Location: {selected_airport['center_lat']:.4f}°N, {selected_airport['center_lon']:.4f}°E")
    st.sidebar.markdown(f"Defense Radius: {selected_airport['radius_km']}km")

    # Main content area - split into map and metrics
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 🗺️ TACTICAL MAP")

        # Create and display map
        folium_map = create_folium_map(
            selected_airport,
            show_heatmap=show_heatmap,
            show_airport=show_airport
        )

        # Render map
        st_folium(
            folium_map,
            width=1000,
            height=600,
            returned_objects=[]
        )

    with col2:
        # Render metrics in right column
        render_metrics(
            selected_airport,
            threats_active=0,  # Placeholder
            defenders_active=0  # Placeholder
        )

    # Bottom status bar
    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("System Status", "🟢 READY")

    with col_b:
        st.metric("Intercept Rate", "0%")

    with col_c:
        st.metric("Engagements", "0")

    with col_d:
        st.metric("Risk Level", "LOW")

    # Footer
    st.markdown("---")
    st.markdown("*Awaiting simulation.py and allocation.py integration*")


if __name__ == '__main__':
    main()
