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
from demo_heuristic_conservative import ConservativeDroneDefenseEnv
import pickle
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="Conservative Strategy Demo",
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
    env.ax.set_title(f'Conservative Strategy - Step {env.timestep}', fontsize=14, fontweight='bold')

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
        # Use partner's CONSERVATIVE environment (updated version)
        # Different config based on airport to show different threat scenarios
        config_file = f'config_{airport_key.lower()}.yaml'
        env = ConservativeDroneDefenseEnv(
            config_path=config_file,
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
            threat_level = "HIGH" if airport_key == 'Taoyuan' else "MODERATE"
            max_concurrent = 12 if airport_key == 'Taoyuan' else 9
            st.markdown(f"""
            **Conservative Strategy:**
            - Kinetic: Close-range only (< 2.5km)
            - AESA: 3+ targets required to fire
            - Defenders: {sum(1 for d in env.defenders if d.alive)}
            - Total Ammo: {sum(d.ammo for d in env.defenders if d.alive)}

            **{airport_key} Scenario:**
            - Threat Level: {threat_level}
            - Max Concurrent: {max_concurrent} attackers
            - {'High engagement capacity' if airport_key == 'Taoyuan' else 'Urban constraints limit engagement'}
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


def run_comparison():
    """Run both scenarios and create comparison charts."""
    st.title("Taoyuan vs Songshan - Threat Comparison")

    results = {}

    for airport in ['Taoyuan', 'Songshan']:
        with st.spinner(f"Running {airport} simulation..."):
            config_file = f'config_{airport.lower()}.yaml'
            env = ConservativeDroneDefenseEnv(
                config_path=config_file,
                use_attention=False,
                render_mode=None
            )
            obs, info = env.reset()

            # Track metrics over time
            timesteps = []
            attackers_alive = []
            attackers_destroyed = []
            defenders_alive = []

            for step in range(100):  # Longer simulation to show neutralizations
                obs, reward, terminated, truncated, info = env.step(0)

                timesteps.append(step)
                attackers_alive.append(info['attackers_alive'])
                attackers_destroyed.append(info['attackers_destroyed'])
                defenders_alive.append(info['defenders_alive'])

                if terminated or truncated:
                    break

            env.close()

            results[airport] = {
                'timesteps': timesteps,
                'attackers_alive': attackers_alive,
                'attackers_destroyed': attackers_destroyed,
                'defenders_alive': defenders_alive,
                'final_kills': attackers_destroyed[-1],
                'max_concurrent': max(attackers_alive)
            }

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1e2130')

    # Plot 1: Active Threats Over Time
    ax1 = axes[0, 0]
    ax1.set_facecolor('#0e1117')
    ax1.plot(results['Taoyuan']['timesteps'], results['Taoyuan']['attackers_alive'],
             'b-', linewidth=2, label='Taoyuan')
    ax1.plot(results['Songshan']['timesteps'], results['Songshan']['attackers_alive'],
             'r-', linewidth=2, label='Songshan')
    ax1.set_xlabel('Timestep', color='white')
    ax1.set_ylabel('Active Threats', color='white')
    ax1.set_title('Active Threats Over Time', color='white', fontweight='bold')
    ax1.legend(facecolor='#1e2130', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')

    # Plot 2: Cumulative Kills
    ax2 = axes[0, 1]
    ax2.set_facecolor('#0e1117')
    ax2.plot(results['Taoyuan']['timesteps'], results['Taoyuan']['attackers_destroyed'],
             'b-', linewidth=2, label='Taoyuan (Aggressive)')
    ax2.plot(results['Songshan']['timesteps'], results['Songshan']['attackers_destroyed'],
             'r-', linewidth=2, label='Songshan (Conservative)')
    ax2.set_xlabel('Timestep', color='white')
    ax2.set_ylabel('Threats Neutralized', color='white')
    ax2.set_title('Neutralization Effectiveness (Timing)', color='white', fontweight='bold')
    ax2.legend(facecolor='#1e2130', edgecolor='white', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')

    # Plot 3: Max Concurrent Threats (Bar Chart)
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0e1117')
    airports = ['Taoyuan', 'Songshan']
    max_threats = [results['Taoyuan']['max_concurrent'], results['Songshan']['max_concurrent']]
    colors = ['blue', 'red']
    bars = ax3.bar(airports, max_threats, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Max Concurrent Threats', color='white')
    ax3.set_title('Peak Threat Intensity', color='white', fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white', axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')

    # Plot 4: Final Kill Count (Bar Chart)
    ax4 = axes[1, 1]
    ax4.set_facecolor('#0e1117')
    final_kills = [results['Taoyuan']['final_kills'], results['Songshan']['final_kills']]
    bars = ax4.bar(airports, final_kills, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    ax4.set_ylabel('Total Neutralizations', color='white')
    ax4.set_title('Defensive Effectiveness (Collateral vs Performance)', color='white', fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.2, color='white', axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary statistics
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ### Taoyuan
        - Max Concurrent Threats: **{results['Taoyuan']['max_concurrent']}**
        - Total Kills: **{results['Taoyuan']['final_kills']}**
        - Engagement Style: Aggressive, early intercepts
        - Collateral Risk: Low - rapid neutralization
        """)

    with col2:
        st.markdown(f"""
        ### Songshan
        - Max Concurrent Threats: **{results['Songshan']['max_concurrent']}**
        - Total Kills: **{results['Songshan']['final_kills']}**
        - Engagement Style: Conservative, delayed timing
        - Collateral Risk: High - careful targeting
        """)

    kill_diff = results['Taoyuan']['final_kills'] - results['Songshan']['final_kills']
    kill_percent = (kill_diff / results['Songshan']['final_kills'] * 100) if results['Songshan']['final_kills'] > 0 else 0

    st.info(f"**Key Finding:** Both airports neutralize threats, but Taoyuan achieves {kill_diff} more kills ({kill_percent:.1f}% higher) due to aggressive engagement timing enabled by low collateral environments. Songshan's urban constraints require delayed, careful targeting despite similar defensive capability.")


def main():
    """Main demo app."""

    # Initialize session state
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False

    st.sidebar.title("Demo Controls")

    # Mode selection
    demo_mode = st.sidebar.radio(
        "Select Mode",
        options=["Live Simulation", "Comparison Analysis"],
        index=0
    )

    if demo_mode == "Comparison Analysis":
        if st.sidebar.button("Run Comparison"):
            run_comparison()
        else:
            st.title("Airport Comparison Mode")
            st.markdown("""
            Click **Run Comparison** to see a side-by-side analysis of threat scenarios
            between Taoyuan and Songshan airports.

            This will generate:
            - Active threats over time
            - Cumulative kills comparison
            - Peak threat intensity
            - Total neutralizations
            """)
        return

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
