# Visualization MVP Specification

## Immediate Goal
Create a Streamlit dashboard that visualizes the airport defense scenario with real population data.

## Data Available
- `taiwan_population.tif`: WorldPop 100m resolution population data for all of Taiwan
- Airport coordinates:
  - Songshan: 25.0694° N, 121.5517° E
  - Taoyuan: 25.0777° N, 121.2325° E

## Required Components

### 1. Map Display
- Use Folium with ESRI World Imagery tiles (no API key needed)
- Center on selected airport
- 10km radius view around airport
- Dark theme UI to match military/defense aesthetic

### 2. Population Heatmap
- Extract 10km x 10km window from WorldPop TIF for each airport
- Overlay as semi-transparent heatmap on map
- Color gradient: blue (low) → yellow → orange → red (high density)
- Only show cells with >10 people to reduce clutter

### 3. Initial UI Layout
```
[Sidebar]                    [Main Map Area]
- Airport selector           - Folium map with satellite
- Threat intensity slider    - Population heatmap overlay
- View toggles               - Airport marker
  □ Population heatmap       
  □ Infrastructure           [Bottom Metrics Bar]
  □ Show trajectories        - Total population in view
- [Start Simulation] button  - High risk zones count
                            - Active threats counter
```

### 4. Placeholder Elements for Integration
Create empty data structures for:
- `threats_list`: Array of threat objects with lat, lon, type, velocity
- `defenders_list`: Array of defender objects with lat, lon, type, status
- `engagement_log`: List of engagement events with timestamps

These will be populated by simulation.py later.

## Implementation Steps
1. Load and cache WorldPop data for both airports on startup
2. Create Streamlit layout with sidebar and main area
3. Initialize Folium map with satellite imagery
4. Generate population heatmap from TIF data
5. Add basic controls and metrics display
6. Create update loop structure for real-time updates

## Performance Requirements
- Map should load in <2 seconds
- Support 50+ objects on map without lag
- Refresh rate of at least 5 FPS for smooth animation

## Visual Polish
- Use military-style colors (dark theme, red threats, green defenders)
- Add subtle animations for threats (blinking or pulsing)
- Show trajectories as dashed lines
- Include "danger zones" as semi-transparent circles
