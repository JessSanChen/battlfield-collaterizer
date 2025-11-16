"""
Integrated Real-Time Simulation Dashboard

Combines our collateral risk visualization with live RL simulation.
Shows defenders/attackers moving on the real map with terrain-aware
engagement decisions highlighted.
"""

import sys
from pathlib import Path
import time
import pickle

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

from src.integration.collateral_env import CollateralDroneDefenseEnv
from src.integration.coordinate_mapper import CoordinateMapper
from collateral_calculator import CollateralCalculator


# Page config
st.set_page_config(
    page_title="Live Terrain-Aware Defense Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMarkdown, .stText {
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #a0a0a0 !important;
    }
    .stMetric .metric-value {
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e2130 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e2130 !important;
    }
    /* Dropdown / Selectbox styling */
    .stSelectbox > div > div {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    .stSelectbox label {
        color: #ffffff !important;
    }
    [data-baseweb="select"] {
        background-color: #262730 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    /* Button styling */
    .stButton > button {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a5e !important;
    }
    .stButton > button:hover {
        background-color: #3a3a4e !important;
        border-color: #6a6a7e !important;
    }
    /* Info/Warning boxes - brighter for visibility */
    .stAlert {
        background-color: #2a3f5f !important;
        color: #ffffff !important;
    }
    [data-testid="stAlert"] {
        background-color: #2a3f5f !important;
        color: #ffffff !important;
    }
    [data-testid="stAlert"] * {
        color: #ffffff !important;
    }
    div[data-testid="stMarkdownContainer"] > div[data-testid="stAlert"] {
        background-color: #2a3f5f !important;
    }
    /* Success/Info boxes (for Taoyuan) */
    .stSuccess, .stInfo {
        background-color: #1e4d2b !important;
        color: #ffffff !important;
    }
    .stSuccess *, .stInfo * {
        color: #ffffff !important;
    }
    div.stSuccess, div.stInfo {
        background-color: #1e4d2b !important;
    }
    div.stSuccess *, div.stInfo * {
        color: #ffffff !important;
    }
    /* Warning boxes (for Songshan) */
    .stWarning {
        background-color: #4d3e1e !important;
        color: #ffffff !important;
    }
    .stWarning * {
        color: #ffffff !important;
    }
    div.stWarning {
        background-color: #4d3e1e !important;
    }
    div.stWarning * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load airport data."""
    data_file = Path('airports_data.pkl')
    if not data_file.exists():
        st.error("airports_data.pkl not found. Run extract_data.py first.")
        st.stop()

    with open(data_file, 'rb') as f:
        return pickle.load(f)


def create_base_map(airport_key: str, airports_data: dict):
    """Create base folium map with collateral risk overlay."""
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

    # Add airport marker
    folium.Marker(
        [center_lat, center_lon],
        popup=f"<b>{airport_key} Airport</b>",
        tooltip=airport_key,
        icon=folium.Icon(color='blue', icon='plane', prefix='fa')
    ).add_to(m)

    # Add population density overlay
    if 'data' in airport_data and airport_data['data'] is not None:
        from folium.plugins import HeatMap

        pop_data = airport_data['data']
        transform = airport_data.get('transform', None)

        if transform is not None:
            # Create heat map points from population grid
            heat_points = []
            rows, cols = pop_data.shape

            # Sample every 10th point to avoid too many points
            for i in range(0, rows, 10):
                for j in range(0, cols, 10):
                    if pop_data[i, j] > 0:
                        # Convert pixel to lat/lon
                        lon, lat = transform * (j, i)
                        # Weight by population density
                        weight = float(pop_data[i, j])
                        heat_points.append([lat, lon, weight])

            # Add heatmap if we have points
            if heat_points:
                HeatMap(
                    heat_points,
                    min_opacity=0.2,
                    max_opacity=0.6,
                    radius=15,
                    blur=20,
                    gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
                ).add_to(m)

    # Add 10km radius circle
    folium.Circle(
        location=[center_lat, center_lon],
        radius=10000,  # 10km
        color='blue',
        fill=False,
        weight=2,
        opacity=0.5,
        popup="10km Defense Perimeter"
    ).add_to(m)

    return m


def get_risk_color(risk: float) -> str:
    """Get color based on collateral risk level."""
    if risk < 0.05:
        return 'green'
    elif risk < 0.15:
        return 'lightgreen'
    elif risk < 0.30:
        return 'orange'
    else:
        return 'red'


def add_simulation_entities(m: folium.Map, env, coord_mapper: CoordinateMapper):
    """Add current defenders and attackers as animated colored dots."""

    # Get current state
    defenders = env.defenders if hasattr(env, 'defenders') else []
    attackers = env.attackers if hasattr(env, 'attackers') else []

    # Add defenders as colored circles
    for i, defender in enumerate(defenders):
        if not defender.alive:
            continue

        lat, lon = coord_mapper.abstract_to_real(defender.x, defender.y)

        # Defender type
        d_type = getattr(defender, 'defender_type', 'Unknown')
        color = 'darkgreen' if d_type.upper() == 'SAM' else 'green'
        radius = 8 if d_type.upper() == 'SAM' else 6

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            weight=2,
            popup=f"<b>Defender {i}</b><br>Type: {d_type}<br>Pos: ({defender.x:.1f}, {defender.y:.1f})",
            tooltip=f"{d_type}"
        ).add_to(m)

    # Add attackers as red circles
    for i, attacker in enumerate(attackers):
        if not attacker.alive:
            continue

        lat, lon = coord_mapper.abstract_to_real(attacker.x, attacker.y)

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.9,
            weight=2,
            popup=f"<b>Attacker {i}</b><br>Pos: ({attacker.x:.1f}, {attacker.y:.1f})<br>Velocity: ({attacker.vx:.2f}, {attacker.vy:.2f})",
            tooltip=f"Threat {i}"
        ).add_to(m)

    return m


def add_engagement_visualization(m: folium.Map, env, coord_mapper: CoordinateMapper,
                                 collateral_calc: CollateralCalculator):
    """Add engagement lines with collateral risk visualization for assigned targets only."""

    defenders = env.defenders if hasattr(env, 'defenders') else []
    attackers = env.attackers if hasattr(env, 'attackers') else []

    engagement_count = 0
    safe_engagements = 0
    engaged_attackers = set()  # Track unique attackers being engaged

    # Get current assignments from environment (if available)
    assignments = getattr(env, 'current_assignments', {})

    # If no assignments available, only show engagements for closest targets within range
    for defender in defenders:
        if not defender.alive:
            continue

        # Check if defender has an assigned target
        assigned_attacker = None
        if hasattr(defender, 'target_id') and defender.target_id is not None:
            # Find the assigned attacker
            for attacker in attackers:
                if hasattr(attacker, 'id') and attacker.id == defender.target_id and attacker.alive:
                    assigned_attacker = attacker
                    break

        # If no assignment system, only draw line to closest attacker within range
        if assigned_attacker is None:
            min_dist = float('inf')
            closest_attacker = None
            defender_range = getattr(defender, 'max_range', 100)

            for attacker in attackers:
                if not attacker.alive:
                    continue
                dist = ((defender.x - attacker.x)**2 + (defender.y - attacker.y)**2)**0.5
                if dist < min_dist and dist < defender_range:
                    min_dist = dist
                    closest_attacker = attacker

            if closest_attacker is None:
                continue
            assigned_attacker = closest_attacker

        if assigned_attacker and assigned_attacker.alive:
            attacker = assigned_attacker  # Use the assigned attacker

            # Calculate intercept point (midpoint approximation)
            intercept_x = (defender.x + attacker.x) / 2.0
            intercept_y = (defender.y + attacker.y) / 2.0

            # Convert to real coordinates
            intercept_lat, intercept_lon = coord_mapper.abstract_to_real(
                intercept_x, intercept_y
            )
            def_lat, def_lon = coord_mapper.abstract_to_real(defender.x, defender.y)
            att_lat, att_lon = coord_mapper.abstract_to_real(attacker.x, attacker.y)

            # Calculate collateral risk
            effector_type = getattr(defender, 'defender_type', 'kinetic').lower()
            risk = collateral_calc.calculate_engagement_risk(
                intercept_lat, intercept_lon, effector_type
            )

            # Check if ocean engagement
            is_ocean = collateral_calc.is_ocean_safe_zone(intercept_lat, intercept_lon)

            # Color based on risk
            color = get_risk_color(risk)
            weight = 3 if risk > 0.3 else 2

            # Draw engagement line
            folium.PolyLine(
                locations=[[def_lat, def_lon], [intercept_lat, intercept_lon],
                          [att_lat, att_lon]],
                color=color,
                weight=weight,
                opacity=0.6,
                popup=f"<b>Active Engagement</b><br>"
                      f"Risk: {risk:.3f}<br>"
                      f"Zone: {'Ocean' if is_ocean else 'Land'}"
            ).add_to(m)

            # Add intercept point marker
            folium.CircleMarker(
                location=[intercept_lat, intercept_lon],
                radius=5,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"<b>Intercept Point</b><br>"
                      f"Risk: {risk:.3f}<br>"
                      f"{'SAFE OCEAN ENGAGEMENT' if is_ocean else 'Urban Area'}"
            ).add_to(m)

            # Track unique attackers being engaged
            attacker_id = id(attacker)  # Use object ID
            if attacker_id not in engaged_attackers:
                engaged_attackers.add(attacker_id)
                engagement_count += 1
                if is_ocean or risk < 0.05:
                    safe_engagements += 1

    return m, engagement_count, safe_engagements


def run_simulation(airport_key: str, airports_data: dict, max_steps: int = 50):
    """Run simulation with real-time visualization."""

    st.title(f"Live Terrain-Aware Defense Simulation: {airport_key}")

    # Create environment
    with st.spinner(f"Initializing {airport_key} defense environment..."):
        env = CollateralDroneDefenseEnv(
            airport_key=airport_key,
            use_attention=False,
            verbose=False
        )
        coord_mapper = CoordinateMapper(airport_key)
        collateral_calc = CollateralCalculator(airport_key, airports_data)

    # Reset environment
    obs, info = env.reset()

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Live Metrics")

        # Metrics placeholders (stacked vertically to avoid nested columns)
        timestep_metric = st.empty()
        kills_metric = st.empty()
        alive_metric = st.empty()
        risk_metric = st.empty()
        engagement_metric = st.empty()
        breach_metric = st.empty()

        st.markdown("---")

        st.subheader("Collateral Statistics")
        stats_container = st.empty()

        st.markdown("---")

        st.subheader("Ocean Advantage")
        ocean_container = st.empty()

        st.markdown("---")

        st.subheader("Position Debug")
        position_debug = st.empty()

    with col1:
        st.subheader("Real-Time Tactical Map")
        map_container = st.empty()

    # Control panel
    st.sidebar.title("Simulation Controls")
    st.sidebar.markdown(f"**Airport:** {airport_key}")
    st.sidebar.markdown(f"**Base Risk:** {collateral_calc.avg_risk:.3f}")

    # Run simulation
    st.sidebar.info("Running simulation...")

    total_reward = 0
    total_engagements = 0
    total_safe_engagements = 0

    for step in range(max_steps):
        # Step environment
        action = 0  # Dummy action for heuristic
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Get collateral stats
        collateral_stats = env.get_collateral_stats()

        # Create map with current state
        m = create_base_map(airport_key, airports_data)
        m = add_simulation_entities(m, env, coord_mapper)
        m, eng_count, safe_eng = add_engagement_visualization(
            m, env, coord_mapper, collateral_calc
        )

        total_engagements += eng_count
        total_safe_engagements += safe_eng

        # Update metrics
        timestep_metric.metric("Timestep", f"{step + 1} / {max_steps}")
        kills_metric.metric("Kills", info['attackers_destroyed'])
        alive_metric.metric("Active Threats", info['attackers_alive'])

        risk_metric.metric(
            "Avg Collateral Risk",
            f"{collateral_stats.get('average_risk', 0.0):.3f}",
            delta=None
        )

        engagement_metric.metric(
            "Intercepting",
            f"{eng_count} targets",
            delta=None
        )

        breach_metric.metric(
            "Safe Zone",
            "BREACHED" if info.get('safe_zone_breached', False) else "SECURE"
        )

        # Update stats
        with stats_container.container():
            st.markdown(f"""
            - **Total Reward:** {total_reward:.2f}
            - **Engagements Tracked:** {collateral_stats.get('engagements_tracked', 0)}
            - **Penalties Applied:** {collateral_stats.get('penalties_applied', 0)}
            """)

        # Ocean advantage (Taoyuan-specific)
        with ocean_container.container():
            if airport_key == 'Taoyuan':
                ocean_pct = (total_safe_engagements / total_engagements * 100
                            if total_engagements > 0 else 0)
                st.success(f"**{ocean_pct:.1f}%** engagements over safe zones")
                st.caption("Taoyuan's ocean advantage: Low collateral risk")
            else:
                st.warning("Urban environment: Higher collateral constraints")
                st.caption("All engagements require careful risk assessment")

        # Position debug - show first attacker position to verify movement
        with position_debug.container():
            if len(env.attackers) > 0:
                att = env.attackers[0]
                st.caption(f"Attacker 0: ({att.x:.1f}, {att.y:.1f}) | v=({att.vx:.2f}, {att.vy:.2f})")

        # Render map using HTML component for proper updates
        import streamlit.components.v1 as components
        map_html = m._repr_html_()

        # Clear and render with unique component
        map_container.empty()
        with map_container:
            components.html(map_html, height=600, scrolling=False)

        # Check termination
        if terminated or truncated:
            st.sidebar.success(f"Simulation complete at timestep {info['timestep']}")
            break

        # Control FPS (5 FPS = 0.2s per frame)
        time.sleep(0.2)

    # Final summary
    env.close()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Final Results")
    st.sidebar.markdown(f"""
    **Defense Outcome:**
    - Attackers Destroyed: {info['attackers_destroyed']}
    - Attackers Remaining: {info['attackers_alive']}
    - Safe Zone: {'Breached' if info.get('safe_zone_breached', False) else 'Secure'}

    **Collateral Impact:**
    - Average Risk: {collateral_stats.get('average_risk', 0.0):.3f}
    - Safe Engagements: {total_safe_engagements}/{total_engagements}
    - Total Reward: {total_reward:.2f}
    """)


def main():
    """Main dashboard application."""

    st.sidebar.title("Integrated Dashboard")
    st.sidebar.markdown("Real-time terrain-aware defense simulation")

    # Load data
    airports_data = load_data()

    # Airport selection
    st.sidebar.markdown("---")
    airport_key = st.sidebar.selectbox(
        "Select Airport",
        options=['Taoyuan', 'Songshan'],
        index=0
    )

    # Simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    max_steps = st.sidebar.slider(
        "Max Timesteps",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )

    # Start button
    st.sidebar.markdown("---")
    if st.sidebar.button("Start Simulation"):
        run_simulation(airport_key, airports_data, max_steps)
    else:
        # Show welcome screen
        st.title("Integrated Terrain-Aware Defense Dashboard")

        st.markdown("""
        ### Real-Time Simulation Visualization

        This dashboard overlays live RL-based defense simulations on real-world maps,
        demonstrating terrain-aware collateral minimization.

        **Features:**
        - Real-world satellite map with population density overlay
        - Live defender/attacker positions
        - Engagement lines colored by collateral risk
        - Real-time metrics and statistics
        - Ocean advantage visualization (Taoyuan)

        **How to Use:**
        1. Select an airport from the sidebar
        2. Adjust simulation parameters
        3. Click "Start Simulation" to begin
        4. Watch as terrain-aware decisions unfold in real-time

        **Key Insights:**
        - **Taoyuan:** Ocean to the west enables low-risk early engagements
        - **Songshan:** Urban environment requires careful risk assessment

        ---
        **Select an airport and click "Start Simulation" to begin**
        """)

        # Show comparison
        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Taoyuan Airport**

            - Base avg risk: ~0.074
            - 75% ocean safe zones
            - Early engagement over water
            - Lower collateral constraints
            """)

        with col2:
            st.warning("""
            **Songshan Airport**

            - Base avg risk: ~0.158 (2.1x higher)
            - Dense urban environment
            - Delayed engagements
            - Higher collateral considerations
            """)


if __name__ == "__main__":
    main()
