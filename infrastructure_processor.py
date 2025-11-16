#!/usr/bin/env python3
"""
Infrastructure Processor for Airport Defense System
Processes GeoJSON data to identify critical infrastructure and safe zones.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


# Risk radii for different infrastructure types (in meters)
RISK_RADII = {
    'hospital': 500,
    'school': 300,
    'fire_station': 200,
    'police': 200,
    'power_plant': 400,
    'power_substation': 300,
    'fuel': 600,  # High explosion risk
    'terminal': 400,  # High casualty risk
    'tower': 300,
    'hangar': 200,
    'station': 300,  # Railway station
}

# Priority scores for infrastructure (higher = more critical)
PRIORITY_SCORES = {
    'hospital': 10,
    'school': 9,
    'terminal': 8,
    'power_plant': 8,
    'fuel': 7,
    'tower': 6,
    'fire_station': 6,
    'police': 5,
    'power_substation': 5,
    'station': 5,
    'hangar': 4,
}

# Airport coordinates for ocean safe zone calculation
TAOYUAN_LON = 121.2325
OCEAN_SAFE_ZONE_LON_THRESHOLD = 121.20  # West of this is ocean


def extract_coordinates(geometry: Dict) -> Tuple[float, float]:
    """Extract lat/lon from GeoJSON geometry."""
    if geometry['type'] == 'Point':
        lon, lat = geometry['coordinates']
        return lat, lon
    elif geometry['type'] == 'Polygon':
        # Use centroid of polygon
        coords = geometry['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return np.mean(lats), np.mean(lons)
    elif geometry['type'] == 'LineString':
        # Use midpoint of line
        coords = geometry['coordinates']
        mid_idx = len(coords) // 2
        lon, lat = coords[mid_idx]
        return lat, lon
    elif geometry['type'] == 'MultiPolygon':
        # Use centroid of first polygon
        coords = geometry['coordinates'][0][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return np.mean(lats), np.mean(lons)
    else:
        return None, None


def classify_infrastructure(properties: Dict) -> Tuple[str, int, int]:
    """
    Classify infrastructure feature and return (type, risk_radius, priority).
    Returns (None, 0, 0) if not a relevant infrastructure type.
    """
    # Check amenity field
    if 'amenity' in properties:
        amenity = properties['amenity']
        if amenity == 'hospital':
            return 'hospital', RISK_RADII['hospital'], PRIORITY_SCORES['hospital']
        elif amenity == 'school':
            return 'school', RISK_RADII['school'], PRIORITY_SCORES['school']
        elif amenity == 'police':
            return 'police', RISK_RADII['police'], PRIORITY_SCORES['police']
        elif amenity == 'fire_station':
            return 'fire_station', RISK_RADII['fire_station'], PRIORITY_SCORES['fire_station']

    # Check aeroway field
    if 'aeroway' in properties:
        aeroway = properties['aeroway']
        if aeroway == 'terminal':
            return 'terminal', RISK_RADII['terminal'], PRIORITY_SCORES['terminal']
        elif aeroway == 'tower':
            return 'tower', RISK_RADII['tower'], PRIORITY_SCORES['tower']
        elif aeroway == 'hangar':
            return 'hangar', RISK_RADII['hangar'], PRIORITY_SCORES['hangar']
        elif aeroway == 'fuel':
            return 'fuel', RISK_RADII['fuel'], PRIORITY_SCORES['fuel']

    # Check power field
    if 'power' in properties:
        power = properties['power']
        if power in ['plant', 'generator']:
            return 'power_plant', RISK_RADII['power_plant'], PRIORITY_SCORES['power_plant']
        elif power == 'substation':
            return 'power_substation', RISK_RADII['power_substation'], PRIORITY_SCORES['power_substation']

    # Check railway field
    if 'railway' in properties:
        railway = properties['railway']
        if railway == 'station':
            return 'station', RISK_RADII['station'], PRIORITY_SCORES['station']

    # Check building field
    if 'building' in properties:
        building = properties['building']
        if building == 'hospital':
            return 'hospital', RISK_RADII['hospital'], PRIORITY_SCORES['hospital']
        elif building == 'school':
            return 'school', RISK_RADII['school'], PRIORITY_SCORES['school']
        elif building in ['government', 'public']:
            return 'police', RISK_RADII['police'], PRIORITY_SCORES['police']

    return None, 0, 0


def get_feature_name(properties: Dict, infra_type: str) -> str:
    """Extract a meaningful name from feature properties."""
    # Try various name fields
    for name_field in ['name', 'name:en', 'official_name', 'alt_name']:
        if name_field in properties and properties[name_field]:
            return properties[name_field]

    # Try type-specific names
    if 'operator' in properties and properties['operator']:
        return properties['operator']

    # Fallback to infrastructure type
    return infra_type.replace('_', ' ').title()


def process_infrastructure_geojson(geojson_path: Path) -> List[Dict]:
    """Process a GeoJSON file and extract infrastructure features."""
    print(f"\nProcessing {geojson_path.name}...")

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    infrastructure = []
    features = data.get('features', [])

    print(f"  Total features in file: {len(features)}")

    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        if not geometry:
            continue

        # Classify the infrastructure
        infra_type, risk_radius, priority = classify_infrastructure(properties)

        if infra_type is None:
            continue

        # Extract coordinates
        lat, lon = extract_coordinates(geometry)

        if lat is None or lon is None:
            continue

        # Get name
        name = get_feature_name(properties, infra_type)

        infrastructure.append({
            'lat': float(lat),
            'lon': float(lon),
            'type': infra_type,
            'name': name,
            'risk_radius': int(risk_radius),
            'priority': int(priority)
        })

    print(f"  Extracted {len(infrastructure)} infrastructure items")

    return infrastructure


def process_water_geojson(geojson_path: Path) -> Dict:
    """Process water bodies GeoJSON to identify ocean safe zones."""
    print(f"\nProcessing {geojson_path.name}...")

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get('features', [])
    print(f"  Total water features: {len(features)}")

    # Identify ocean/sea bodies west of airport
    ocean_features = []

    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        if not geometry:
            continue

        # Check if it's a large water body (ocean, sea, bay)
        natural = properties.get('natural', '')
        water = properties.get('water', '')
        name = properties.get('name', '')

        # Extract coordinates to check if west of airport
        lat, lon = extract_coordinates(geometry)

        if lat is None or lon is None:
            continue

        # Check if west of the ocean threshold (indicating ocean)
        if lon < OCEAN_SAFE_ZONE_LON_THRESHOLD:
            ocean_features.append({
                'lat': float(lat),
                'lon': float(lon),
                'name': name or 'Ocean',
                'type': natural or water,
                'geometry': geometry
            })

    print(f"  Identified {len(ocean_features)} ocean/water features west of airport")

    return {
        'ocean_threshold_lon': OCEAN_SAFE_ZONE_LON_THRESHOLD,
        'ocean_features': ocean_features,
        'safe_zone_multiplier': 0.1  # Ocean areas have 10% of normal risk
    }


def print_summary_statistics(airport_name: str, infrastructure: List[Dict], water_data: Dict = None):
    """Print summary statistics for processed data."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {airport_name.upper()}")
    print(f"{'='*60}")

    # Count by type
    type_counts = {}
    for item in infrastructure:
        infra_type = item['type']
        type_counts[infra_type] = type_counts.get(infra_type, 0) + 1

    print(f"\nInfrastructure by Type:")
    for infra_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {infra_type:20s}: {count:3d}")

    print(f"\nTotal Infrastructure Items: {len(infrastructure)}")

    # High priority items
    high_priority = [i for i in infrastructure if i['priority'] >= 8]
    print(f"High Priority Items (≥8): {len(high_priority)}")

    # Critical infrastructure
    critical = ['hospital', 'school', 'terminal']
    critical_items = [i for i in infrastructure if i['type'] in critical]
    print(f"Critical Infrastructure: {len(critical_items)}")
    for crit_type in critical:
        count = len([i for i in infrastructure if i['type'] == crit_type])
        if count > 0:
            print(f"  {crit_type.title()}: {count}")

    # Water data
    if water_data:
        print(f"\nOcean Safe Zone:")
        print(f"  Threshold Longitude: {water_data['ocean_threshold_lon']}°E")
        print(f"  Ocean Features: {len(water_data['ocean_features'])}")
        print(f"  Risk Multiplier: {water_data['safe_zone_multiplier']} (90% reduction)")

    print(f"\n{'='*60}\n")


def main():
    """Main processing function."""
    print("="*60)
    print("INFRASTRUCTURE PROCESSOR")
    print("="*60)

    # Process Songshan infrastructure
    songshan_path = Path('songshan_infrastructure.geojson')
    if songshan_path.exists():
        songshan_infrastructure = process_infrastructure_geojson(songshan_path)

        # Save to pickle
        output_path = Path('songshan_infrastructure.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(songshan_infrastructure, f)
        print(f"  Saved: {output_path}")

        print_summary_statistics('Songshan', songshan_infrastructure)
    else:
        print(f"ERROR: {songshan_path} not found!")

    # Process Taoyuan infrastructure
    taoyuan_path = Path('taoyuan_infrastructure.geojson')
    taoyuan_water_path = Path('taoyuan_water.geojson')

    if taoyuan_path.exists():
        taoyuan_infrastructure = process_infrastructure_geojson(taoyuan_path)

        # Process water bodies for ocean safe zone
        water_data = None
        if taoyuan_water_path.exists():
            water_data = process_water_geojson(taoyuan_water_path)

            # Save water data separately
            water_output_path = Path('taoyuan_water_data.pkl')
            with open(water_output_path, 'wb') as f:
                pickle.dump(water_data, f)
            print(f"  Saved: {water_output_path}")

        # Save infrastructure to pickle
        output_path = Path('taoyuan_infrastructure.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(taoyuan_infrastructure, f)
        print(f"  Saved: {output_path}")

        print_summary_statistics('Taoyuan', taoyuan_infrastructure, water_data)
    else:
        print(f"ERROR: {taoyuan_path} not found!")

    print("="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - songshan_infrastructure.pkl")
    print("  - taoyuan_infrastructure.pkl")
    print("  - taoyuan_water_data.pkl")
    print("\nReady for dashboard integration!")


if __name__ == '__main__':
    main()
