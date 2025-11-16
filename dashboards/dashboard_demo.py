"""
Simple Side-by-Side Demo Dashboard
Shows partner's simulation + static results map
"""

import sys
from pathlib import Path
import time

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO

from src.integration.collateral_env import CollateralDroneDefenseEnv
from src.integration.coordinate_mapper import CoordinateMapper
from collateral_calculator import CollateralCalculator
from drone_defense_env import DroneDefenseEnv
import pickle
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="Terrain-Aware Defense Demo",
    layout="wide"
)

# Dark mode
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .stMarkdown { color: #ffffff !important; }

    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background-color: #1e2130 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e2130 !important;
    }

    /* Button dark background */
    .stButton > button {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a5e !important;
    }
    .stButton > button:hover {
        background-color: #3a3a4e !important;
        border-color: #6a6a7e !important;
    }

    /* Dropdown dark background */
    .stSelectbox > div > div {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    [data-baseweb="select"] {
        background-color: #262730 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def render_simulation_frame(env):
    """Use partner's original render method."""
    # Call partner's render which updates env.fig
    if env.fig is None:
        env.fig, env.ax = plt.subplots(figsize=(8, 8))

    env.ax.clear()
    env.ax.set_xlim(0, env.map_width)
    env.ax.set_ylim(0, env.map_height)
    env.ax.set_aspect('equal')
    env.ax.set_title(f'Partner RL Simulation - Step {env.timestep}', fontsize=14)

    # Draw safe zone
    safe_zone_rect = patches.Rectangle(
        (env.safe_zone[0], env.safe_zone[1]),
        env.safe_zone[2] - env.safe_zone[0],
        env.safe_zone[3] - env.safe_zone[1],
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3
    )
    env.ax.add_patch(safe_zone_rect)

    # Draw defenders
    for defender in env.defenders:
        if not defender.alive:
            continue
        color = 'blue' if defender.defender_type == 'SAM' else 'darkorange'
        env.ax.plot(defender.x, defender.y, marker='s', color=color,
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        # Range circle
        range_circle = plt.Circle((defender.x, defender.y), defender.max_range,
                                 color=color, fill=False, linestyle='--', alpha=0.3)
        env.ax.add_patch(range_circle)

    # Draw attackers
    for attacker in env.attackers:
        if not attacker.alive:
            continue
        env.ax.plot(attacker.x, attacker.y, marker='^', color='red',
                   markersize=12, markeredgecolor='darkred', markeredgewidth=1.5)
        # Velocity vector
        if attacker.get_speed() > 0.1:
            env.ax.arrow(attacker.x, attacker.y, attacker.vx * 8, attacker.vy * 8,
                        head_width=2, head_length=3, fc='red', ec='darkred', alpha=0.5)

    plt.tight_layout()
    return env.fig


def create_terrain_map(airport_key: str, airports_data: dict):
    """Create static terrain map with population heatmap."""
    airport_data = airports_data[airport_key]
    center_lat = airport_data['center_lat']
    center_lon = airport_data['center_lon']

    # Create map with satellite imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )

    # Add population heatmap
    if 'data' in airport_data and airport_data['data'] is not None:
        pop_data = airport_data['data']
        transform = airport_data.get('transform', None)

        if transform is not None:
            heat_points = []
            rows, cols = pop_data.shape

            for i in range(0, rows, 10):
                for j in range(0, cols, 10):
                    if pop_data[i, j] > 0:
                        lon, lat = transform * (j, i)
                        weight = float(pop_data[i, j])
                        heat_points.append([lat, lon, weight])

            if heat_points:
                HeatMap(
                    heat_points,
                    min_opacity=0.2,
                    max_opacity=0.6,
                    radius=15,
                    blur=20,
                    gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
                ).add_to(m)

    # Add airport marker
    folium.Marker(
        [center_lat, center_lon],
        popup=f"<b>{airport_key} Airport</b>",
        icon=folium.Icon(color='blue', icon='plane', prefix='fa')
    ).add_to(m)

    return m


def run_demo(airport_key: str, max_steps: int = 50):
    """Run simple demo with attack simulation."""

    st.title(f"Terrain-Aware Defense Demo: {airport_key}")


    # Load airport data
    with open('airports_data.pkl', 'rb') as f:
        airports_data = pickle.load(f)

    # Initialize
    with st.spinner("Initializing simulation..."):
        # Use partner's original environment for visualization
        env = DroneDefenseEnv(
            config_path='config.yaml',
            use_attention=False,
            render_mode=None  # We'll get the figure directly
        )
        obs, info = env.reset()

        # Create terrain map
        terrain_map = create_terrain_map(airport_key, airports_data)

    # Layout - 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Live Attack Simulation")
        sim_container = st.empty()

    with col2:
        st.subheader("Terrain & Population")
        # Render static terrain map directly (not in st.empty())
        st_folium(terrain_map, width=400, height=400)

    with col3:
        st.subheader("Live Metrics")
        metrics_container = st.empty()

        st.markdown("---")

        stats_text = st.empty()

    # Run simulation
    st.sidebar.info("Running simulation...")

    for step in range(max_steps):
        # Step
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)

        # Render simulation
        fig = render_simulation_frame(env)

        # Display in streamlit
        with sim_container:
            st.pyplot(fig)
        plt.close(fig)

        # Update metrics (stacked vertically to avoid nested columns)
        with metrics_container:
            st.metric("Step", f"{step+1} / {max_steps}")
            st.metric("Active Threats", info['attackers_alive'])
            st.metric("Kills", info['attackers_destroyed'])

        # Stats
        with stats_text:
            st.markdown(f"""
            **Partner's RL Simulation:**
            - Defenders: {sum(1 for d in env.defenders if d.alive)}
            - Total Ammo: {sum(d.ammo for d in env.defenders if d.alive)}

            **Terrain Context ({airport_key}):**
            - {'Ocean safe zones available for low-risk engagements' if airport_key == 'Taoyuan' else 'Urban environment - careful targeting required'}
            """)

        # Check termination
        if terminated or truncated:
            st.sidebar.success(f"Simulation complete at step {step+1}")
            break

        # Frame rate - much faster for detailed engagement view
        time.sleep(0.05)

    # Final summary
    env.close()
    st.sidebar.success("Demo complete! Both visualizations will remain visible.")

    # Stop execution to keep visualizations on screen
    st.stop()


def main():
    """Main demo app."""

    # Initialize session state
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False

    st.sidebar.title("Demo Controls")

    # Airport selection
    airport_key = st.sidebar.selectbox(
        "Select Airport",
        options=['Taoyuan', 'Songshan'],
        index=0
    )

    # Steps
    max_steps = st.sidebar.slider(
        "Max Steps",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )

    # Start button
    st.sidebar.markdown("---")
    if st.sidebar.button("Start Demo"):
        st.session_state.demo_running = True

    if st.session_state.demo_running:
        run_demo(airport_key, max_steps)
    else:
        st.title("Terrain-Aware Defense Demonstration")

        st.markdown("""
        ### Ready for Demo

        This dashboard shows our terrain-aware collateral minimization system
        integrated with the partner's RL-based drone defense.

        **Key Features:**
        - Real-time simulation visualization
        - Terrain-aware engagement decisions
        - Ocean advantage for Taoyuan Airport
        - Live metrics and statistics

        **Select an airport and click "Start Demo" to begin**
        """)


if __name__ == '__main__':
    main()
