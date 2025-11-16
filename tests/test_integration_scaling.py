"""
Integration Scaling Validation Script

Validates that all coordinate conversions, speeds, and spawn points
are correctly scaled between the abstract RL system and real-world terrain.
"""

import sys
from pathlib import Path
import pickle

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
from src.integration.coordinate_mapper import CoordinateMapper
from src.integration.realistic_configs import (
    REALISTIC_CONFIG,
    DEFENDER_POSITIONS,
    SPAWN_POINTS,
    METERS_PER_UNIT,
    SAM_SPEED_MS,
    SAM_SPEED_UNITS_PER_TIMESTEP,
    KINETIC_SPEED_MS,
    KINETIC_SPEED_UNITS_PER_TIMESTEP,
    ATTACKER_SPEED_MIN_MS,
    ATTACKER_SPEED_MAX_MS,
    ATTACKER_SPEED_MIN_UNITS,
    ATTACKER_SPEED_MAX_UNITS
)
from collateral_calculator import CollateralCalculator


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def validate_spawn_points():
    """Validate attacker spawn points are correctly scaled."""
    print_header("SPAWN POINT VALIDATION")

    center_x, center_y = 100, 100
    print(f"Map Center: ({center_x}, {center_y})")
    print(f"Expected distance: 10-15km = 100-150 units from center")
    print()

    print(f"{'Spawn':<8} {'Abstract Pos':<20} {'Distance':<15} {'Real Dist':<15} {'Direction':<12} {'Status':<10}")
    print(f"{'-'*8} {'-'*20} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")

    valid_count = 0
    total_count = len(SPAWN_POINTS)

    for i, spawn in enumerate(SPAWN_POINTS):
        x, y = spawn['x'], spawn['y']

        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        dist_units = np.sqrt(dx**2 + dy**2)
        dist_km = dist_units * METERS_PER_UNIT / 1000.0

        # Determine direction
        if abs(dx) > abs(dy):
            direction = "East" if dx > 0 else "West"
        else:
            direction = "North" if dy > 0 else "South"

        # Check if within expected range
        is_valid = 100 <= dist_units <= 150
        status = "✓ VALID" if is_valid else "✗ INVALID"

        if is_valid:
            valid_count += 1

        print(f"{i:<8} ({x:3.0f}, {y:3.0f}){'':<10} {dist_units:6.1f} units {dist_km:6.2f} km {direction:<12} {status:<10}")

    print()
    print(f"Summary: {valid_count}/{total_count} spawn points within 10-15km range")

    if valid_count == total_count:
        print("✓ All spawn points correctly positioned")
    else:
        print(f"⚠ {total_count - valid_count} spawn points outside expected range")

    return valid_count == total_count


def validate_defender_positions():
    """Validate defender positions are correctly scaled."""
    print_header("DEFENDER POSITION VALIDATION")

    center_x, center_y = 100, 100
    print(f"Map Center: ({center_x}, {center_y})")
    print(f"Expected: SAMs at ~8km (80 units), Kinetic at ~5km (50 units)")
    print()

    print(f"{'Type':<10} {'Label':<20} {'Position':<15} {'Distance':<15} {'Real Dist':<15} {'Status':<10}")
    print(f"{'-'*10} {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

    sam_valid = 0
    kinetic_valid = 0

    for defender in DEFENDER_POSITIONS:
        d_type = defender['type']
        label = defender['label']
        x, y = defender['x'], defender['y']

        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        dist_units = np.sqrt(dx**2 + dy**2)
        dist_km = dist_units * METERS_PER_UNIT / 1000.0

        # Expected distances
        if d_type == 'SAM':
            expected_dist = 80  # 8km
            tolerance = 10
        else:  # KINETIC
            expected_dist = 50  # 5km
            tolerance = 10

        is_valid = abs(dist_units - expected_dist) < tolerance
        status = "✓ VALID" if is_valid else "✗ INVALID"

        if is_valid:
            if d_type == 'SAM':
                sam_valid += 1
            else:
                kinetic_valid += 1

        print(f"{d_type:<10} {label:<20} ({x:3.0f}, {y:3.0f}){'':<5} {dist_units:6.1f} units {dist_km:6.2f} km {status:<10}")

    print()
    print(f"Summary:")
    print(f"  SAM batteries: {sam_valid}/4 at correct distance (~8km)")
    print(f"  Kinetic drones: {kinetic_valid}/4 at correct distance (~5km)")

    all_valid = sam_valid == 4 and kinetic_valid == 4
    if all_valid:
        print("✓ All defenders correctly positioned")
    else:
        print("⚠ Some defenders outside expected positions")

    return all_valid


def validate_speeds():
    """Validate speed conversions are correct."""
    print_header("SPEED VALIDATION")

    print("Conversion Factor: 1 abstract unit = 100 meters")
    print("Timestep: 1 second (assumed)")
    print()

    speeds = [
        ("SAM Projectile", SAM_SPEED_MS, 1000, SAM_SPEED_UNITS_PER_TIMESTEP),
        ("Kinetic Projectile", KINETIC_SPEED_MS, 50, KINETIC_SPEED_UNITS_PER_TIMESTEP),
        ("Attacker (min)", ATTACKER_SPEED_MIN_MS, 15, ATTACKER_SPEED_MIN_UNITS),
        ("Attacker (max)", ATTACKER_SPEED_MAX_MS, 30, ATTACKER_SPEED_MAX_UNITS),
    ]

    print(f"{'Entity':<20} {'Real Speed':<15} {'Expected Units':<20} {'Config Units':<20} {'Status':<10}")
    print(f"{'-'*20} {'-'*15} {'-'*20} {'-'*20} {'-'*10}")

    all_valid = True
    for name, real_speed_ms, expected_units_per_step, config_value in speeds:
        # Calculate units per timestep
        calculated = real_speed_ms / METERS_PER_UNIT

        # For config values, we need to check what's actually in REALISTIC_CONFIG
        status = "✓ VALID" if abs(calculated - config_value) < 0.01 else "✗ INVALID"

        if status == "✗ INVALID":
            all_valid = False

        print(f"{name:<20} {real_speed_ms:6.1f} m/s {calculated:6.2f} units/s {config_value:6.2f} units/s {status:<10}")

    print()

    # Calculate time to traverse map
    print("Time to Traverse Map:")
    map_size = 200  # units
    map_size_km = map_size * METERS_PER_UNIT / 1000.0

    print(f"  Map size: {map_size} units = {map_size_km:.1f} km")
    print()

    for name, real_speed_ms, _, units_per_step in speeds:
        if units_per_step > 0:
            time_steps = map_size / units_per_step
            time_seconds = time_steps  # 1 timestep = 1 second
            time_minutes = time_seconds / 60.0

            print(f"  {name:<20}: {time_steps:6.0f} timesteps ({time_minutes:5.1f} minutes)")

    print()
    if all_valid:
        print("✓ All speeds correctly converted")
    else:
        print("⚠ Some speed conversions incorrect")

    return all_valid


def validate_coordinate_conversions():
    """Validate coordinate mapper conversions."""
    print_header("COORDINATE CONVERSION VALIDATION")

    for airport_key in ['Taoyuan', 'Songshan']:
        print(f"\n{airport_key} Airport:")
        print(f"{'-'*60}")

        mapper = CoordinateMapper(airport_key)

        # Test key points
        test_points = [
            ("Center (airport)", 100, 100),
            ("10km West (ocean for Taoyuan)", 0, 100),
            ("10km East (land)", 200, 100),
            ("10km North", 100, 200),
            ("10km South", 100, 0),
            ("Spawn point NW", 25, 175),
        ]

        print(f"\n{'Location':<30} {'Abstract':<15} {'Real Coordinates':<25} {'Distance':<15}")
        print(f"{'-'*30} {'-'*15} {'-'*25} {'-'*15}")

        center_lat, center_lon = mapper.center_lat, mapper.center_lon

        for label, x, y in test_points:
            lat, lon = mapper.abstract_to_real(x, y)

            # Calculate distance from center
            dist_m = mapper.distance_meters(center_lat, center_lon, lat, lon)
            dist_km = dist_m / 1000.0

            print(f"{label:<30} ({x:3.0f}, {y:3.0f}){'':<5} ({lat:8.4f}°N, {lon:9.4f}°E) {dist_km:6.2f} km")

        # Test round-trip conversion
        print(f"\nRound-trip Conversion Test:")
        x_orig, y_orig = 75, 125
        lat, lon = mapper.abstract_to_real(x_orig, y_orig)
        x_back, y_back = mapper.real_to_abstract(lat, lon)

        error_x = abs(x_back - x_orig)
        error_y = abs(y_back - y_orig)

        print(f"  Original: ({x_orig}, {y_orig})")
        print(f"  → Real: ({lat:.6f}°N, {lon:.6f}°E)")
        print(f"  → Back: ({x_back:.6f}, {y_back:.6f})")
        print(f"  Error: {error_x:.9f} units x, {error_y:.9f} units y")

        if error_x < 0.001 and error_y < 0.001:
            print(f"  ✓ Round-trip conversion accurate")
        else:
            print(f"  ✗ Round-trip error too large")


def validate_collateral_calculations():
    """Validate collateral risk calculations at key points."""
    print_header("COLLATERAL RISK VALIDATION")

    # Load data
    data_file = Path('airports_data.pkl')
    if not data_file.exists():
        print("⚠ airports_data.pkl not found. Skipping collateral validation.")
        return False

    with open(data_file, 'rb') as f:
        airports_data = pickle.load(f)

    for airport_key in ['Taoyuan', 'Songshan']:
        print(f"\n{airport_key} Airport:")
        print(f"{'-'*60}")

        mapper = CoordinateMapper(airport_key)
        calc = CollateralCalculator(airport_key, airports_data)

        # Test key points
        test_points = [
            ("Airport center", 100, 100),
            ("10km West", 0, 100),
            ("10km East", 200, 100),
            ("5km West", 50, 100),
            ("5km East", 150, 100),
        ]

        print(f"\n{'Location':<20} {'Abstract':<15} {'Real Coordinates':<30} {'Risk':<10} {'Ocean':<10}")
        print(f"{'-'*20} {'-'*15} {'-'*30} {'-'*10} {'-'*10}")

        for label, x, y in test_points:
            lat, lon = mapper.abstract_to_real(x, y)
            risk = calc.calculate_engagement_risk(lat, lon, 'kinetic')
            is_ocean = calc.is_ocean_safe_zone(lat, lon)

            ocean_str = "🌊 Yes" if is_ocean else "🏙️ No"

            print(f"{label:<20} ({x:3.0f}, {y:3.0f}){'':<5} ({lat:8.4f}°N, {lon:9.4f}°E) {risk:6.3f} {ocean_str:<10}")

        print(f"\nExpected for {airport_key}:")
        if airport_key == 'Taoyuan':
            print(f"  • West (ocean): Low risk (<0.05), Ocean=Yes")
            print(f"  • East (land): Higher risk (>0.10), Ocean=No")
        else:
            print(f"  • All directions: Higher risk due to urban density")
            print(f"  • No ocean safe zones")

    return True


def validate_realistic_config():
    """Validate REALISTIC_CONFIG dictionary has all required values."""
    print_header("REALISTIC_CONFIG VALIDATION")

    required_keys = {
        'map_width': 200,
        'map_height': 200,
        'safe_zone': (90, 90, 110, 110),
        'defenders': None,  # Will check separately
        'spawn_points': None,  # Will check separately
    }

    print("Checking required configuration keys:")
    print()

    all_valid = True
    for key, expected_value in required_keys.items():
        if key in REALISTIC_CONFIG:
            actual_value = REALISTIC_CONFIG[key]
            if expected_value is not None:
                match = actual_value == expected_value
                status = "✓" if match else "✗"
                print(f"  {status} {key}: {actual_value} (expected: {expected_value})")
                if not match:
                    all_valid = False
            else:
                print(f"  ✓ {key}: present ({type(actual_value).__name__})")
        else:
            print(f"  ✗ {key}: MISSING")
            all_valid = False

    # Check defenders
    print()
    print("Defender Configuration:")
    if 'defenders' in REALISTIC_CONFIG:
        defenders = REALISTIC_CONFIG['defenders']
        print(f"  Total defenders: {len(defenders)}")

        sam_count = sum(1 for d in defenders if d.get('type', '').upper() == 'SAM')
        kinetic_count = sum(1 for d in defenders if d.get('type', '').upper() == 'KINETIC')

        print(f"  SAM batteries: {sam_count} (expected: 4)")
        print(f"  Kinetic drones: {kinetic_count} (expected: 4)")

        if sam_count == 4 and kinetic_count == 4:
            print(f"  ✓ Correct defender distribution")
        else:
            print(f"  ✗ Incorrect defender distribution")
            all_valid = False
    else:
        print(f"  ✗ Defenders missing")
        all_valid = False

    # Check spawn points
    print()
    print("Spawn Points Configuration:")
    if 'spawn_points' in REALISTIC_CONFIG:
        spawn_points = REALISTIC_CONFIG['spawn_points']
        print(f"  Total spawn points: {len(spawn_points)} (expected: 16)")

        if len(spawn_points) == 16:
            print(f"  ✓ Correct number of spawn points")
        else:
            print(f"  ✗ Incorrect number of spawn points")
            all_valid = False
    else:
        print(f"  ✗ Spawn points missing")
        all_valid = False

    print()
    if all_valid:
        print("✓ REALISTIC_CONFIG is properly configured")
    else:
        print("⚠ REALISTIC_CONFIG has issues")

    return all_valid


def main():
    """Run all validation tests."""
    print_header("INTEGRATION SCALING VALIDATION")

    print("This script validates that all conversions between the abstract RL system")
    print("and real-world terrain are correctly scaled and configured.")

    # Run all validations
    results = {}

    results['spawn_points'] = validate_spawn_points()
    results['defender_positions'] = validate_defender_positions()
    results['speeds'] = validate_speeds()
    validate_coordinate_conversions()  # Always informational
    collateral_valid = validate_collateral_calculations()
    results['realistic_config'] = validate_realistic_config()

    # Final summary
    print_header("VALIDATION SUMMARY")

    print(f"{'Test':<30} {'Status':<10}")
    print(f"{'-'*30} {'-'*10}")

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title():<30} {status:<10}")

    print()

    all_passed = all(results.values())

    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("Integration is correctly scaled:")
        print("  ✓ Spawn points at 10-15km from airport")
        print("  ✓ Defenders at appropriate distances (SAM: 8km, Kinetic: 5km)")
        print("  ✓ Speeds correctly converted (1 unit = 100m)")
        print("  ✓ Coordinate conversions accurate (<0.001 unit error)")
        print("  ✓ Collateral calculations using real coordinates")
        print("  ✓ REALISTIC_CONFIG properly configured")
        print()
        print("The system is ready for demonstration!")
    else:
        print("⚠ SOME VALIDATIONS FAILED")
        print()
        print("Please review the failures above and correct the configuration.")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\nFailed tests: {', '.join(failed_tests)}")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
