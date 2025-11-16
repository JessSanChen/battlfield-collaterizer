#!/usr/bin/env python3
"""
Extract 10km x 10km population density windows around Taiwan airports.
Saves processed data as pickle files for fast dashboard loading.
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import pickle
from pathlib import Path

# Airport coordinates
AIRPORTS = {
    'Songshan': {
        'name': 'Songshan Airport (TSA)',
        'lat': 25.0694,
        'lon': 121.5517,
        'type': 'Urban - High Density'
    },
    'Taoyuan': {
        'name': 'Taoyuan Airport (TPE)',
        'lat': 25.0777,
        'lon': 121.2325,
        'type': 'Rural/Suburban - Low Density'
    }
}

# 10km radius in degrees (approximate: 1 degree ≈ 111km at this latitude)
RADIUS_KM = 10
RADIUS_DEG = RADIUS_KM / 111.0


def extract_airport_window(tif_path, airport_info, airport_key):
    """
    Extract a 10km x 10km window of population data around an airport.

    Args:
        tif_path: Path to the WorldPop TIF file
        airport_info: Dictionary with 'lat' and 'lon' keys
        airport_key: Name key for the airport

    Returns:
        Dictionary with extracted data and metadata
    """
    lat = airport_info['lat']
    lon = airport_info['lon']

    # Calculate bounding box (10km radius = 20km x 20km box)
    min_lon = lon - RADIUS_DEG
    max_lon = lon + RADIUS_DEG
    min_lat = lat - RADIUS_DEG
    max_lat = lat + RADIUS_DEG

    print(f"\n{airport_info['name']}:")
    print(f"  Location: {lat}°N, {lon}°E")
    print(f"  Bounding box: {min_lat:.4f}°N to {max_lat:.4f}°N, {min_lon:.4f}°E to {max_lon:.4f}°E")

    with rasterio.open(tif_path) as src:
        # Get the window for this bounding box
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)

        # Read the data for this window
        data = src.read(1, window=window)

        # Get the transform for this window
        window_transform = src.window_transform(window)

        # Calculate actual bounds
        bounds = rasterio.windows.bounds(window, src.transform)

        # Get statistics
        valid_data = data[data > 0]  # Only cells with population
        total_population = np.sum(valid_data)
        max_density = np.max(valid_data) if len(valid_data) > 0 else 0
        mean_density = np.mean(valid_data) if len(valid_data) > 0 else 0

        print(f"  Data shape: {data.shape}")
        print(f"  Total population: {total_population:,.0f}")
        print(f"  Max density: {max_density:.1f} people/cell")
        print(f"  Mean density: {mean_density:.1f} people/cell")
        print(f"  Populated cells: {len(valid_data):,} / {data.size:,}")

        # Prepare data package
        data_package = {
            'airport_key': airport_key,
            'airport_name': airport_info['name'],
            'airport_type': airport_info['type'],
            'center_lat': lat,
            'center_lon': lon,
            'bounds': bounds,  # (min_lon, min_lat, max_lon, max_lat)
            'data': data,
            'transform': window_transform,
            'crs': src.crs,
            'total_population': total_population,
            'max_density': max_density,
            'mean_density': mean_density,
            'radius_km': RADIUS_KM
        }

        return data_package


def main():
    """Extract and save population data for both airports."""
    tif_path = Path('taiwan_population.tif')

    if not tif_path.exists():
        print(f"ERROR: {tif_path} not found!")
        print("Please ensure the WorldPop data file is in the current directory.")
        return

    print("="*60)
    print("EXTRACTING AIRPORT POPULATION DATA")
    print("="*60)
    print(f"Source: {tif_path}")
    print(f"Radius: {RADIUS_KM}km around each airport")

    # Extract data for each airport
    extracted_data = {}
    for airport_key, airport_info in AIRPORTS.items():
        data_package = extract_airport_window(tif_path, airport_info, airport_key)
        extracted_data[airport_key] = data_package

        # Save individual pickle file
        output_file = Path(f'{airport_key.lower()}_data.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(data_package, f)
        print(f"  Saved: {output_file}")

    # Also save combined data
    combined_file = Path('airports_data.pkl')
    with open(combined_file, 'wb') as f:
        pickle.dump(extracted_data, f)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Files created:")
    print(f"  - songshan_data.pkl")
    print(f"  - taoyuan_data.pkl")
    print(f"  - airports_data.pkl (combined)")
    print(f"\nReady for dashboard.py!")


if __name__ == '__main__':
    main()
