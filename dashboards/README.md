# Integrated Real-Time Simulation Dashboard

## Overview

The integrated dashboard combines our terrain-aware collateral risk visualization with live RL-based defense simulation. Watch in real-time as defenders and attackers move across the real map with engagement decisions highlighted based on collateral risk.

## Features

### 🗺️ Real-World Map Visualization
- **Base Map**: OpenStreetMap centered on airport
- **Population Overlay**: Semi-transparent red overlay showing population density
- **Defense Perimeter**: 10km radius circle around airport
- **Airport Marker**: Blue plane icon at center

### 🎯 Live Simulation Entities
- **Defenders** (Green markers):
  - Dark green = SAM batteries (crosshairs icon)
  - Light green = Kinetic drones (fighter jet icon)
  - Shows real-time positions mapped from abstract coordinates

- **Attackers** (Red markers):
  - Red warning icons
  - Tracks live positions as they approach safe zone

### 🎨 Engagement Visualization
- **Engagement Lines**: Connect defender → intercept point → attacker
  - Green: Safe (ocean/low risk < 0.05)
  - Light green: Low risk (0.05-0.15)
  - Orange: Medium risk (0.15-0.30)
  - Red: High risk (> 0.30)

- **Intercept Points**: Colored circle markers showing predicted engagement locations
  - 🌊 Taoyuan: Many green markers over ocean (SAFE ENGAGEMENT)
  - 🏙️ Songshan: More orange/red markers over urban areas

### 📊 Live Metrics Dashboard
- **Timestep counter**: Current/Max timesteps
- **Kills**: Total attackers destroyed
- **Threats**: Remaining attackers alive
- **Avg Collateral Risk**: Real-time collateral score
- **Active Engagements**: Current defender-attacker pairs
- **Safe Zone Status**: Secure ✅ / Breached ⚠️

### 🌊 Ocean Advantage Panel (Taoyuan-specific)
- Shows percentage of engagements in safe zones
- Highlights Taoyuan's geographic advantage
- Real-time comparison vs urban constraints

## Usage

### Starting the Dashboard

```bash
# From project root
source venv/bin/activate
streamlit run dashboards/dashboard_integrated.py
```

Dashboard will be available at: `http://localhost:8501`

### Running a Simulation

1. **Select Airport**:
   - Taoyuan: Ocean advantage demonstration
   - Songshan: Urban constraints demonstration

2. **Adjust Parameters**:
   - Max Timesteps: 20-100 (default: 50)
   - Runs at 5 FPS (0.2s per frame)

3. **Start Simulation**:
   - Click "▶️ Start Simulation" button
   - Watch real-time updates on map and metrics
   - Simulation runs until all attackers destroyed or max timesteps reached

4. **Observe Key Differences**:
   - **Taoyuan**: Look for green engagement lines over ocean (west of airport)
   - **Songshan**: Notice more orange/red lines over urban areas

## Technical Details

### Coordinate Mapping
- Abstract simulation space: 200×200 units
- Real-world mapping: 20km×20km (1 unit = 100m)
- CoordinateMapper converts defender/attacker positions to lat/lon
- Intercept points calculated as midpoint between defender and attacker

### Collateral Risk Calculation
For each potential engagement:
1. Calculate predicted intercept point (midpoint)
2. Convert to real lat/lon coordinates
3. Query CollateralCalculator for risk score:
   - Population density
   - Infrastructure proximity
   - Flight path conflicts
4. Color-code visualization based on risk

### Performance
- **Frame Rate**: 5 FPS (0.2s per timestep)
- **Map Updates**: Full re-render each frame
- **Metrics**: Real-time updates via st.empty() containers

## Key Insights to Look For

### Taoyuan Simulation
- Early engagement lines extending west (over ocean)
- Majority of intercept markers in green (low risk)
- Ocean advantage metric showing 60-80% safe engagements
- Defenders can engage aggressively without collateral concerns

### Songshan Simulation
- More cautious engagement patterns
- Orange/red intercept markers in urban areas
- Urban constraint warnings in sidebar
- Defenders must balance threat elimination vs civilian safety

### Side-by-Side Comparison
Run both simulations to see how **the same attack scenario** is handled differently:
- Same defender layout (8 defenders: 4 SAM, 4 kinetic)
- Same attacker spawn points
- Different engagement strategies based on terrain

## Troubleshooting

### Dashboard Won't Start
- Ensure `airports_data.pkl` exists (run `extract_data.py` first)
- Check all dependencies installed: `pip install -r requirements.txt`
- Verify Python 3.9 (required for compatibility)

### No Engagement Lines Showing
- Attackers may not be in range yet (early timesteps)
- Try running more timesteps (increase max to 100)
- Check console for errors

### Map Not Loading
- Check internet connection (OpenStreetMap requires internet)
- Verify `streamlit-folium` is installed correctly
- Try refreshing browser

## Files

- `dashboard_integrated.py`: Main dashboard application (550+ lines)
- `README.md`: This file
- Required imports:
  - `src/integration/collateral_env.py`
  - `src/integration/coordinate_mapper.py`
  - `collateral_calculator.py`
  - `airports_data.pkl`

## Next Steps

1. **Train RL Agent**: Replace heuristic with attention network for smarter allocation
2. **Add Playback**: Save simulation data and replay at different speeds
3. **Comparison Mode**: Run Taoyuan and Songshan side-by-side
4. **Historical Engagements**: Track and visualize all past engagements with trails
5. **Export Results**: Save simulation data to CSV for analysis

## Credits

Built for the 8-hour defense hackathon, integrating:
- Partner's RL-based drone defense system (gymnasium + PPO)
- Our terrain-aware collateral minimization system (real-world data)
