# Airport Defense Battle Management System

## Project Overview
We are building a drone defense system for airports in an 8-hour hackathon. The system helps operators defend against drone attacks (small quads and Shahed-like threats) while minimizing collateral damage to civilian populations and infrastructure.

## Competition Context
- Judges: Skilled defense engineers and researchers
- Scoring: Creativity, Technical Elegance, Usability (1-10 scale)
- Company focus: Small, cheap drones for Ukraine (we're adapting for civilian airports)

## Core Concept
An intelligent battle management algorithm that:
1. Tracks incoming drone threats
2. Calculates collateral damage risk for each defensive action
3. Recommends optimal defender-to-threat allocation
4. Allows operator to execute strategy with minimal clicks

## Airports
- Songshan (TSA): Urban, city center, high population density
- Taoyuan (TPE): Rural/suburban, lower population density

## Technical Stack
- Streamlit for dashboard
- Folium with ESRI satellite tiles for maps
- WorldPop data for population density (already downloaded as .tif)
- Python for all simulation logic

## Current Phase: Visualization MVP
Focus on building the dashboard that displays:
1. Real-time map with satellite imagery
2. Population density heatmap overlay
3. Threat and defender positions
4. Engagement trajectories
5. Metrics sidebar (intercept rate, collateral score, etc.)

## File Structure
- taiwan_population.tif: WorldPop population density data
- dashboard.py: Main Streamlit application
- simulation.py: Will contain battle simulation logic (partner working on this)
- allocation.py: Will contain optimization algorithm (partner working on this)

## Key Constraints
- 8-hour time limit
- Demo must be visual and impressive
- Must show clear advantage over greedy baseline algorithm
- Must handle two different airport scenarios

## Coding Priorities
1. Get a working visualization first
2. Use synthetic/approximate data where needed
3. Focus on demo-able results over perfect accuracy
4. Make it look professional with minimal effort
