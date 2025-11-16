"""
Overlay Dashboard - Simulation on Satellite Map
Combines matplotlib simulation with satellite imagery background
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
from PIL import Image
import requests
from io import BytesIO
import pickle

from drone_defense_env import DroneDefenseEnv
from src.integration.coordinate_mapper import CoordinateMapper

# Page config
st.set_page_config(
    page_title="Overlay Simulation Demo",
    layout="wide"
)

# Dark mode
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #1e2130 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a5e !important;
    }
</style>
""", unsafe_allow_html=True)


def get_satellite_tile(center_lat, center_lon, zoom=13, size=800):
    """
    Fetch satellite imagery from Esri World Imagery.
    Returns image that can be used as matplotlib background.
    """
    # Calculate tile bounds for the center point
    # Using OpenStreetMap tile system
    import math

    def lat_lon_to_tile(lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x_tile, y_tile)

    x_tile, y_tile = lat_lon_to_tile(center_lat, center_lon, zoom)

    # Fetch tile from Esri
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y_tile}/{x_tile}"

    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        # Return blank if fetch fails
        return Image.new('RGB', (size, size), color='#0e1117')


def render_overlay_frame(env, airport_data, coord_mapper):
    """
    Render simulation overlaid on satellite map.
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e2130')

    # Get satellite background
    center_lat = airport_data['center_lat']
    center_lon = airport_data['center_lon']

    satellite_img = get_satellite_tile(center_lat, center_lon, zoom=12)

    # Calculate extent in lat/lon for the 20km x 20km area
    # 1 degree lat ~ 111 km, 1 degree lon ~ 111 km * cos(lat)
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

    # 10 km radius = 20 km total
    lat_extent = 10.0 / km_per_degree_lat
    lon_extent = 10.0 / km_per_degree_lon

    extent = [
        center_lon - lon_extent,  # left
        center_lon + lon_extent,  # right
        center_lat - lat_extent,  # bottom
        center_lat + lat_extent   # top
    ]

    # Show satellite image as background
    ax.imshow(satellite_img, extent=extent, aspect='auto', alpha=0.7)

    # Plot simulation entities in lat/lon coordinates
    # Convert defenders
    for defender in env.defenders:
        if not defender.alive:
            continue

        lat, lon = coord_mapper.simulation_to_real(defender.x, defender.y)

        color = 'blue' if defender.defender_type == 'SAM' else 'darkorange'
        ax.plot(lon, lat, marker='s', color=color, markersize=15,
                markeredgecolor='white', markeredgewidth=2, alpha=0.9)

        # Range circle in lat/lon
        range_km = defender.max_range * 0.1  # units to km
        range_degrees_lat = range_km / km_per_degree_lat
        range_degrees_lon = range_km / km_per_degree_lon

        circle = plt.Circle((lon, lat), range_degrees_lat,
                           color=color, fill=False, linestyle='--',
                           alpha=0.4, linewidth=2)
        ax.add_patch(circle)

    # Plot attackers
    for attacker in env.attackers:
        if not attacker.alive:
            continue

        lat, lon = coord_mapper.simulation_to_real(attacker.x, attacker.y)

        ax.plot(lon, lat, marker='^', color='red', markersize=15,
                markeredgecolor='yellow', markeredgewidth=2, alpha=0.9)

        # Velocity arrow
        if attacker.get_speed() > 0.1:
            # Convert velocity to lat/lon delta
            dlat = attacker.vy * 0.8 / km_per_degree_lat
            dlon = attacker.vx * 0.8 / km_per_degree_lon

            ax.arrow(lon, lat, dlon, dlat,
                    head_width=0.002, head_length=0.003,
                    fc='red', ec='yellow', alpha=0.7, linewidth=2)

    # Safe zone (runway)
    runway_corners = [
        (env.safe_zone[0], env.safe_zone[1]),  # bottom-left
        (env.safe_zone[2], env.safe_zone[1]),  # bottom-right
        (env.safe_zone[2], env.safe_zone[3]),  # top-right
        (env.safe_zone[0], env.safe_zone[3])   # top-left
    ]

    runway_lons = []
    runway_lats = []
    for x, y in runway_corners:
        lat, lon = coord_mapper.simulation_to_real(x, y)
        runway_lats.append(lat)
        runway_lons.append(lon)
    runway_lons.append(runway_lons[0])  # Close polygon
    runway_lats.append(runway_lats[0])

    ax.plot(runway_lons, runway_lats, 'g-', linewidth=3, alpha=0.8)
    ax.fill(runway_lons, runway_lats, color='green', alpha=0.2)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Longitude', color='white', fontsize=12)
    ax.set_ylabel('Latitude', color='white', fontsize=12)
    ax.set_title(f'Simulation Overlay - Step {env.timestep}',
                 color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0e1117')

    plt.tight_layout()
    return fig


def run_overlay_demo(airport_key: str, max_steps: int = 50):
    """Run overlay demo."""

    st.title(f"Satellite Overlay Demo: {airport_key}")

    # Load airport data
    with open('airports_data.pkl', 'rb') as f:
        airports_data = pickle.load(f)

    airport_data = airports_data[airport_key]

    # Initialize coordinate mapper
    coord_mapper = CoordinateMapper(
        center_lat=airport_data['center_lat'],
        center_lon=airport_data['center_lon'],
        radius_km=10.0
    )

    # Initialize environment
    with st.spinner("Initializing simulation..."):
        env = DroneDefenseEnv(
            config_path='config.yaml',
            use_attention=False,
            render_mode=None
        )
        obs, info = env.reset()

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Satellite Overlay")
        map_container = st.empty()

    with col2:
        st.subheader("Metrics")
        metrics_container = st.empty()

    # Run simulation
    st.sidebar.info("Running overlay simulation...")

    for step in range(max_steps):
        # Step
        obs, reward, terminated, truncated, info = env.step(0)

        # Render overlay
        fig = render_overlay_frame(env, airport_data, coord_mapper)

        with map_container:
            st.pyplot(fig)
        plt.close(fig)

        # Metrics
        with metrics_container:
            st.metric("Step", f"{step+1} / {max_steps}")
            st.metric("Active Threats", info['attackers_alive'])
            st.metric("Kills", info['attackers_destroyed'])
            st.metric("Defenders", info['defenders_alive'])

        if terminated or truncated:
            st.sidebar.success(f"Simulation complete at step {step+1}")
            break

        time.sleep(0.1)

    env.close()
    st.sidebar.success("Overlay demo complete!")
    st.stop()


def main():
    """Main app."""

    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False

    st.sidebar.title("Overlay Demo Controls")

    airport_key = st.sidebar.selectbox(
        "Select Airport",
        options=['Taoyuan', 'Songshan'],
        index=0
    )

    max_steps = st.sidebar.slider(
        "Max Steps",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )

    if st.sidebar.button("Start Overlay Demo"):
        st.session_state.demo_running = True

    if st.session_state.demo_running:
        run_overlay_demo(airport_key, max_steps)
    else:
        st.title("Satellite Overlay Demonstration")
        st.markdown("""
        ### Overlay Simulation on Satellite Map

        This demo overlays the partner's RL simulation directly onto
        satellite imagery, showing real-world context.

        Features:
        - Satellite imagery background
        - Simulation entities in real coordinates
        - Defender range circles
        - Attacker velocity vectors
        - Runway overlay

        Click **Start Overlay Demo** to begin.
        """)


if __name__ == '__main__':
    main()
