#!/usr/bin/env python3
"""
Flight Path Generator for Airport Defense System
Generates semi-realistic curved flight paths based on ICAO standards.
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict


# Airport configurations
AIRPORTS = {
    'songshan': {
        'name': 'Songshan Airport (TSA)',
        'lat': 25.0694,
        'lon': 121.5517,
        'runways': [
            {'name': '10', 'heading': 100, 'reciprocal': '28', 'reciprocal_heading': 280}
        ],
        'elevation_ft': 18,
        'urban': True,
        'noise_abatement': True
    },
    'taoyuan': {
        'name': 'Taoyuan Airport (TPE)',
        'lat': 25.0777,
        'lon': 121.2325,
        'runways': [
            {'name': '05L', 'heading': 50, 'reciprocal': '23R', 'reciprocal_heading': 230},
            {'name': '05R', 'heading': 50, 'reciprocal': '23L', 'reciprocal_heading': 230}
        ],
        'elevation_ft': 106,
        'urban': False,
        'ocean_west': True
    }
}

# ICAO standard parameters
GLIDESLOPE_ANGLE = 3.0  # degrees
APPROACH_DISTANCE_NM = 10  # nautical miles
NM_TO_KM = 1.852
NM_TO_DEG_LAT = 1.0 / 60.0  # Approximate
FT_PER_NM_3DEG = 318  # Feet of descent per nautical mile at 3° glideslope
NUM_PATH_POINTS = 40


def heading_to_offset(heading: float, distance_deg: float) -> Tuple[float, float]:
    """Convert heading and distance to lat/lon offset."""
    # Convert heading to radians (0° = North, 90° = East)
    heading_rad = np.radians(heading)

    # Calculate offsets
    lat_offset = distance_deg * np.cos(heading_rad)
    lon_offset = distance_deg * np.sin(heading_rad)

    return lat_offset, lon_offset


def generate_smooth_curve(start_point: Tuple[float, float],
                         end_point: Tuple[float, float],
                         control_points: List[Tuple[float, float]],
                         num_points: int = NUM_PATH_POINTS) -> List[List[float]]:
    """Generate a smooth curve through control points using cubic spline interpolation."""
    # Combine start, control points, and end
    all_points = [start_point] + control_points + [end_point]

    # Extract lats and lons
    lats = [p[0] for p in all_points]
    lons = [p[1] for p in all_points]

    # Create parameter t from 0 to 1
    t = np.linspace(0, 1, len(all_points))
    t_smooth = np.linspace(0, 1, num_points)

    # Create cubic splines for lat and lon
    cs_lat = CubicSpline(t, lats, bc_type='natural')
    cs_lon = CubicSpline(t, lons, bc_type='natural')

    # Evaluate splines
    smooth_lats = cs_lat(t_smooth)
    smooth_lons = cs_lon(t_smooth)

    # Return as list of [lat, lon] pairs
    return [[float(lat), float(lon)] for lat, lon in zip(smooth_lats, smooth_lons)]


def generate_altitude_profile(approach_distance_nm: float,
                              airport_elevation_ft: float,
                              num_points: int = NUM_PATH_POINTS) -> List[int]:
    """Generate altitude profile for 3-degree glideslope."""
    # Calculate starting altitude
    descent_ft = approach_distance_nm * FT_PER_NM_3DEG
    start_altitude_ft = airport_elevation_ft + descent_ft

    # Generate linear descent
    altitudes = np.linspace(start_altitude_ft, airport_elevation_ft, num_points)

    return [int(alt) for alt in altitudes]


def generate_songshan_approaches(airport_config: Dict) -> List[Dict]:
    """Generate approach paths for Songshan Airport."""
    lat = airport_config['lat']
    lon = airport_config['lon']
    elevation_ft = airport_config['elevation_ft']
    approaches = []

    # RWY 10 ILS Approach (heading 100°, from west)
    # Curved approach following Keelung River with noise abatement
    approach_distance_deg = (APPROACH_DISTANCE_NM * NM_TO_KM) / 111.0

    # Calculate final approach course
    lat_offset, lon_offset = heading_to_offset(100 + 180, approach_distance_deg)
    start_lat = lat + lat_offset
    start_lon = lon + lon_offset

    # Add curve to follow river valley
    control_points = [
        (lat + lat_offset * 0.7, lon + lon_offset * 0.7 - 0.005),  # Slight left
        (lat + lat_offset * 0.4, lon + lon_offset * 0.4 - 0.003),  # Curve back
        (lat + lat_offset * 0.15, lon + lon_offset * 0.15)  # Final approach
    ]

    smooth_path = generate_smooth_curve((start_lat, start_lon), (lat, lon), control_points)
    altitude_profile = generate_altitude_profile(APPROACH_DISTANCE_NM, elevation_ft)

    approaches.append({
        'name': 'ILS_RWY10',
        'type': 'approach',
        'runway': '10',
        'heading': 100,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.9,  # High vulnerability over urban area
        'notes': 'Curved approach following Keelung River valley'
    })

    # RWY 28 ILS Approach (heading 280°, from east)
    lat_offset, lon_offset = heading_to_offset(280 + 180, approach_distance_deg)
    start_lat = lat + lat_offset
    start_lon = lon + lon_offset

    # Less curved, more direct over water
    control_points = [
        (lat + lat_offset * 0.6, lon + lon_offset * 0.6),
        (lat + lat_offset * 0.3, lon + lon_offset * 0.3)
    ]

    smooth_path = generate_smooth_curve((start_lat, start_lon), (lat, lon), control_points)
    altitude_profile = generate_altitude_profile(APPROACH_DISTANCE_NM, elevation_ft)

    approaches.append({
        'name': 'ILS_RWY28',
        'type': 'approach',
        'runway': '28',
        'heading': 280,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.7,  # Lower vulnerability over water initially
        'notes': 'Approach over Taipei Basin with water crossing'
    })

    return approaches


def generate_songshan_departures(airport_config: Dict) -> List[Dict]:
    """Generate departure paths for Songshan Airport."""
    lat = airport_config['lat']
    lon = airport_config['lon']
    elevation_ft = airport_config['elevation_ft']
    departures = []

    # RWY 10 Departure with noise abatement turn
    departure_distance_deg = (APPROACH_DISTANCE_NM * NM_TO_KM) / 111.0

    # Initial climb on runway heading, then turn right
    lat_offset, lon_offset = heading_to_offset(100, departure_distance_deg)
    end_lat = lat + lat_offset
    end_lon = lon + lon_offset

    # Noise abatement: turn right after takeoff
    control_points = [
        (lat + lat_offset * 0.2, lon + lon_offset * 0.2),  # Initial climb
        (lat + lat_offset * 0.5, lon + lon_offset * 0.5 + 0.008),  # Start turn right
        (lat + lat_offset * 0.8, lon + lon_offset * 0.8 + 0.015)  # Continue turn
    ]

    smooth_path = generate_smooth_curve((lat, lon), (end_lat, end_lon + 0.02), control_points)
    altitude_profile = list(reversed(generate_altitude_profile(APPROACH_DISTANCE_NM, elevation_ft)))

    departures.append({
        'name': 'SID_RWY10',
        'type': 'departure',
        'runway': '10',
        'heading': 100,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.85,
        'notes': 'Noise abatement right turn after departure'
    })

    return departures


def generate_taoyuan_approaches(airport_config: Dict) -> List[Dict]:
    """Generate approach paths for Taoyuan Airport."""
    lat = airport_config['lat']
    lon = airport_config['lon']
    elevation_ft = airport_config['elevation_ft']
    approaches = []

    approach_distance_deg = (APPROACH_DISTANCE_NM * NM_TO_KM) / 111.0

    # RWY 05L Approach (heading 50°, from southwest/ocean)
    lat_offset, lon_offset = heading_to_offset(50 + 180, approach_distance_deg * 1.2)
    start_lat = lat + lat_offset
    start_lon = lon + lon_offset

    # Gentle curve from ocean
    control_points = [
        (lat + lat_offset * 0.7, lon + lon_offset * 0.7),
        (lat + lat_offset * 0.4, lon + lon_offset * 0.4),
        (lat + lat_offset * 0.15, lon + lon_offset * 0.15)
    ]

    smooth_path = generate_smooth_curve((start_lat, start_lon), (lat, lon), control_points)
    altitude_profile = generate_altitude_profile(APPROACH_DISTANCE_NM * 1.2, elevation_ft)

    approaches.append({
        'name': 'ILS_RWY05L',
        'type': 'approach',
        'runway': '05L',
        'heading': 50,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.3,  # LOW - over ocean initially
        'notes': 'Ocean approach from southwest - SAFE ZONE'
    })

    # RWY 23R Approach (heading 230°, from northeast - OCEAN APPROACH)
    # This is the key safe approach - comes from west over ocean
    lat_offset, lon_offset = heading_to_offset(230 + 180, approach_distance_deg * 1.3)
    start_lat = lat + lat_offset
    start_lon = lon + lon_offset  # This will be WEST of airport

    # Long approach over ocean
    control_points = [
        (lat + lat_offset * 0.8, lon + lon_offset * 0.8),  # Still over ocean
        (lat + lat_offset * 0.5, lon + lon_offset * 0.5),  # Approaching coast
        (lat + lat_offset * 0.2, lon + lon_offset * 0.2)   # Final approach
    ]

    smooth_path = generate_smooth_curve((start_lat, start_lon), (lat, lon), control_points)
    altitude_profile = generate_altitude_profile(APPROACH_DISTANCE_NM * 1.3, elevation_ft)

    approaches.append({
        'name': 'ILS_RWY23R_OCEAN',
        'type': 'approach',
        'runway': '23R',
        'heading': 230,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.2,  # VERY LOW - ocean approach!
        'notes': 'Primary ocean approach from west - MAXIMUM SAFE ZONE'
    })

    # RWY 23R Alternate (shorter, from land)
    lat_offset, lon_offset = heading_to_offset(230 + 180, approach_distance_deg)
    start_lat = lat + lat_offset
    start_lon = lon + lon_offset

    control_points = [
        (lat + lat_offset * 0.6, lon + lon_offset * 0.6),
        (lat + lat_offset * 0.3, lon + lon_offset * 0.3)
    ]

    smooth_path = generate_smooth_curve((start_lat, start_lon), (lat, lon), control_points)
    altitude_profile = generate_altitude_profile(APPROACH_DISTANCE_NM, elevation_ft)

    approaches.append({
        'name': 'ILS_RWY23R_LAND',
        'type': 'approach',
        'runway': '23R',
        'heading': 230,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.6,
        'notes': 'Alternate land approach - higher population risk'
    })

    return approaches


def generate_taoyuan_departures(airport_config: Dict) -> List[Dict]:
    """Generate departure paths for Taoyuan Airport."""
    lat = airport_config['lat']
    lon = airport_config['lon']
    elevation_ft = airport_config['elevation_ft']
    departures = []

    departure_distance_deg = (APPROACH_DISTANCE_NM * NM_TO_KM) / 111.0

    # RWY 05L Departure (northeast)
    lat_offset, lon_offset = heading_to_offset(50, departure_distance_deg)
    end_lat = lat + lat_offset
    end_lon = lon + lon_offset

    control_points = [
        (lat + lat_offset * 0.3, lon + lon_offset * 0.3),
        (lat + lat_offset * 0.6, lon + lon_offset * 0.6)
    ]

    smooth_path = generate_smooth_curve((lat, lon), (end_lat, end_lon), control_points)
    altitude_profile = list(reversed(generate_altitude_profile(APPROACH_DISTANCE_NM, elevation_ft)))

    departures.append({
        'name': 'SID_RWY05L',
        'type': 'departure',
        'runway': '05L',
        'heading': 50,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.5,
        'notes': 'Standard departure to northeast'
    })

    # RWY 23R Departure to OCEAN (west) - SAFE ZONE
    lat_offset, lon_offset = heading_to_offset(230, departure_distance_deg * 1.2)
    end_lat = lat + lat_offset
    end_lon = lon + lon_offset  # West = ocean

    control_points = [
        (lat + lat_offset * 0.3, lon + lon_offset * 0.3),
        (lat + lat_offset * 0.7, lon + lon_offset * 0.7)
    ]

    smooth_path = generate_smooth_curve((lat, lon), (end_lat, end_lon), control_points)
    altitude_profile = list(reversed(generate_altitude_profile(APPROACH_DISTANCE_NM * 1.2, elevation_ft)))

    departures.append({
        'name': 'SID_RWY23R_OCEAN',
        'type': 'departure',
        'runway': '23R',
        'heading': 230,
        'smooth_path': smooth_path,
        'altitude_profile': altitude_profile,
        'vulnerability_score': 0.25,  # LOW - over ocean
        'notes': 'Ocean departure to west - SAFE ZONE'
    })

    return departures


def main():
    """Generate all flight paths and save to JSON."""
    print("="*60)
    print("FLIGHT PATH GENERATOR")
    print("="*60)
    print(f"ICAO Standards: {GLIDESLOPE_ANGLE}° glideslope")
    print(f"Approach distance: {APPROACH_DISTANCE_NM} NM")
    print(f"Path resolution: {NUM_PATH_POINTS} points per path\n")

    flight_paths_data = {
        'metadata': {
            'source': 'ICAO Annex 14 - Aerodromes',
            'standard': 'PANS-OPS (Doc 8168)',
            'glideslope_angle_deg': GLIDESLOPE_ANGLE,
            'approach_distance_nm': APPROACH_DISTANCE_NM,
            'descent_rate_ft_per_nm': FT_PER_NM_3DEG,
            'path_points': NUM_PATH_POINTS,
            'generation_method': 'Cubic spline interpolation'
        },
        'airports': {}
    }

    # Generate Songshan paths
    print("Generating Songshan Airport paths...")
    songshan_config = AIRPORTS['songshan']
    songshan_approaches = generate_songshan_approaches(songshan_config)
    songshan_departures = generate_songshan_departures(songshan_config)

    flight_paths_data['airports']['songshan'] = {
        'name': songshan_config['name'],
        'location': {
            'lat': songshan_config['lat'],
            'lon': songshan_config['lon']
        },
        'elevation_ft': songshan_config['elevation_ft'],
        'characteristics': {
            'urban': True,
            'noise_abatement': True,
            'safe_zones': 'Limited - dense urban environment'
        },
        'approaches': songshan_approaches,
        'departures': songshan_departures
    }

    print(f"  Approaches: {len(songshan_approaches)}")
    print(f"  Departures: {len(songshan_departures)}")

    # Generate Taoyuan paths
    print("\nGenerating Taoyuan Airport paths...")
    taoyuan_config = AIRPORTS['taoyuan']
    taoyuan_approaches = generate_taoyuan_approaches(taoyuan_config)
    taoyuan_departures = generate_taoyuan_departures(taoyuan_config)

    flight_paths_data['airports']['taoyuan'] = {
        'name': taoyuan_config['name'],
        'location': {
            'lat': taoyuan_config['lat'],
            'lon': taoyuan_config['lon']
        },
        'elevation_ft': taoyuan_config['elevation_ft'],
        'characteristics': {
            'urban': False,
            'ocean_west': True,
            'safe_zones': 'Extensive - ocean approaches available'
        },
        'approaches': taoyuan_approaches,
        'departures': taoyuan_departures
    }

    print(f"  Approaches: {len(taoyuan_approaches)}")
    print(f"  Departures: {len(taoyuan_departures)}")

    # Highlight ocean approaches
    ocean_approaches = [a for a in taoyuan_approaches if 'OCEAN' in a['name']]
    print(f"  Ocean safe zone approaches: {len(ocean_approaches)}")

    # Save to JSON
    output_path = Path('flight_paths_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flight_paths_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Total paths generated: {len(songshan_approaches) + len(songshan_departures) + len(taoyuan_approaches) + len(taoyuan_departures)}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("VULNERABILITY SUMMARY")
    print(f"{'='*60}")

    print("\nSongshan (Urban):")
    for path in songshan_approaches + songshan_departures:
        print(f"  {path['name']:20s} - Risk: {path['vulnerability_score']:.1f}")

    print("\nTaoyuan (Coastal):")
    for path in taoyuan_approaches + taoyuan_departures:
        indicator = " ⭐ SAFE ZONE" if path['vulnerability_score'] < 0.4 else ""
        print(f"  {path['name']:20s} - Risk: {path['vulnerability_score']:.1f}{indicator}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
