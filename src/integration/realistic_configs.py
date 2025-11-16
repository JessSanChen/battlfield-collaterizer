"""
Realistic Configurations for RL Drone Defense System

Scales the partner's abstract RL system to real-world parameters based on:
- 1 abstract unit = 100 meters
- Map size: 200×200 units = 20km×20km
- Center: (100, 100) = Airport location

Real-world systems modeled:
- SAM batteries: NASAMS, Patriot-class systems (~50km range)
- Kinetic drones: Counter-UAS interceptors (~20km range)
- Attacker drones: Commercial/tactical UAVs (15-30 m/s)
"""

import math
from typing import Dict, List, Tuple


# ============================================================================
# SCALE CONSTANTS
# ============================================================================
METERS_PER_UNIT = 100.0  # 1 abstract unit = 100 meters
MAP_SIZE = 200  # 200×200 units
MAP_SIZE_KM = MAP_SIZE * METERS_PER_UNIT / 1000.0  # 20km
MAP_CENTER = (100, 100)  # Airport location

# ============================================================================
# REAL-WORLD DEFENDER PARAMETERS
# ============================================================================

# SAM Battery (NASAMS, Patriot-class)
SAM_RANGE_KM = 50.0  # Real SAM range
SAM_RANGE_METERS = SAM_RANGE_KM * 1000.0
SAM_RANGE_UNITS = SAM_RANGE_METERS / METERS_PER_UNIT  # 500 units
SAM_RANGE_CAPPED = min(SAM_RANGE_UNITS, MAP_SIZE)  # Cap at map size (200 units)

SAM_SPEED_MS = 1000.0  # Mach 3 ≈ 1000 m/s
SAM_SPEED_UNITS_PER_TIMESTEP = SAM_SPEED_MS / METERS_PER_UNIT  # 10 units/timestep

SAM_RELOAD_TIME_SEC = 10.0  # 10 seconds between shots
SAM_TIMESTEPS_PER_SEC = 1.0  # Assuming 1 timestep = 1 second
SAM_RELOAD_TIMESTEPS = int(SAM_RELOAD_TIME_SEC * SAM_TIMESTEPS_PER_SEC)

SAM_AMMO = 12  # Typical SAM battery load
SAM_PK = 0.85  # Probability of kill

# Kinetic Interceptor Drone (Counter-UAS)
KINETIC_RANGE_KM = 20.0  # Interceptor drone range
KINETIC_RANGE_METERS = KINETIC_RANGE_KM * 1000.0
KINETIC_RANGE_UNITS = KINETIC_RANGE_METERS / METERS_PER_UNIT  # 200 units (entire map)

KINETIC_SPEED_MS = 50.0  # High-speed interceptor drone
KINETIC_SPEED_UNITS_PER_TIMESTEP = KINETIC_SPEED_MS / METERS_PER_UNIT  # 0.5 units/timestep

KINETIC_RELOAD_TIME_SEC = 2.0  # Quick reload/launch
KINETIC_RELOAD_TIMESTEPS = int(KINETIC_RELOAD_TIME_SEC * SAM_TIMESTEPS_PER_SEC)

KINETIC_AMMO = 30  # More ammo, cheaper interceptors
KINETIC_PK = 0.75  # Slightly lower P(kill) than SAM

# ============================================================================
# REAL-WORLD ATTACKER PARAMETERS
# ============================================================================

# Commercial/Tactical UAV speeds
ATTACKER_SPEED_MIN_MS = 15.0  # Slow commercial drone
ATTACKER_SPEED_MAX_MS = 30.0  # Fast tactical drone
ATTACKER_SPEED_MIN_UNITS = ATTACKER_SPEED_MIN_MS / METERS_PER_UNIT  # 0.15 units/timestep
ATTACKER_SPEED_MAX_UNITS = ATTACKER_SPEED_MAX_MS / METERS_PER_UNIT  # 0.3 units/timestep

# Warhead mass (kg)
ATTACKER_WARHEAD_MIN_KG = 1.0   # Small commercial drone payload
ATTACKER_WARHEAD_MAX_KG = 10.0  # Large tactical drone warhead

# ============================================================================
# SAFE ZONE (AIRPORT TERMINAL AREA)
# ============================================================================

# Terminal area: 2km × 2km centered at airport
SAFE_ZONE_SIZE_KM = 2.0
SAFE_ZONE_SIZE_METERS = SAFE_ZONE_SIZE_KM * 1000.0
SAFE_ZONE_SIZE_UNITS = SAFE_ZONE_SIZE_METERS / METERS_PER_UNIT  # 20 units

SAFE_ZONE_HALF_SIZE = SAFE_ZONE_SIZE_UNITS / 2.0  # 10 units

SAFE_ZONE = {
    'x_min': int(MAP_CENTER[0] - SAFE_ZONE_HALF_SIZE),  # 90
    'y_min': int(MAP_CENTER[1] - SAFE_ZONE_HALF_SIZE),  # 90
    'x_max': int(MAP_CENTER[0] + SAFE_ZONE_HALF_SIZE),  # 110
    'y_max': int(MAP_CENTER[1] + SAFE_ZONE_HALF_SIZE),  # 110
}

# ============================================================================
# DEFENDER CONFIGURATIONS
# ============================================================================

DEFENDER_STATS = {
    'SAM': {
        'max_range': float(SAM_RANGE_CAPPED),
        'ammo': SAM_AMMO,
        'reload_time': SAM_RELOAD_TIMESTEPS,
        'shot_speed': float(SAM_SPEED_UNITS_PER_TIMESTEP),
        'base_probability_of_kill': SAM_PK,
    },
    'KINETIC': {
        'max_range': float(KINETIC_RANGE_UNITS),
        'ammo': KINETIC_AMMO,
        'reload_time': KINETIC_RELOAD_TIMESTEPS,
        'shot_speed': float(KINETIC_SPEED_UNITS_PER_TIMESTEP),
        'base_probability_of_kill': KINETIC_PK,
    }
}

# ============================================================================
# DEFENDER POSITIONS (Airport Defense Layout)
# ============================================================================

# Standard airport defense: SAMs at perimeter, kinetics closer in
DEFENDER_POSITIONS = [
    # SAM batteries at 4 cardinal points, 4km from center (40 units) - halved radius
    {'type': 'SAM', 'x': 100, 'y': 140, 'label': 'SAM North'},      # North
    {'type': 'SAM', 'x': 100, 'y': 60,  'label': 'SAM South'},      # South
    {'type': 'SAM', 'x': 60,  'y': 100, 'label': 'SAM West'},       # West
    {'type': 'SAM', 'x': 140, 'y': 100, 'label': 'SAM East'},       # East

    # Kinetic drones closer to terminal, 2.5km from center (25 units) - halved radius
    {'type': 'KINETIC', 'x': 100, 'y': 125, 'label': 'Kinetic North'},  # North
    {'type': 'KINETIC', 'x': 100, 'y': 75,  'label': 'Kinetic South'},  # South
    {'type': 'KINETIC', 'x': 75,  'y': 100, 'label': 'Kinetic West'},   # West
    {'type': 'KINETIC', 'x': 125, 'y': 100, 'label': 'Kinetic East'},   # East
]

# ============================================================================
# ATTACKER SPAWN CONFIGURATION
# ============================================================================

# Spawn outside 10km radius (100 units from center)
SPAWN_RADIUS_MIN_KM = 10.0
SPAWN_RADIUS_MIN_UNITS = SPAWN_RADIUS_MIN_KM * 1000.0 / METERS_PER_UNIT  # 100 units

def generate_spawn_points(
    num_points: int = 16,
    min_radius_units: float = SPAWN_RADIUS_MIN_UNITS,
    max_radius_units: float = MAP_SIZE / math.sqrt(2)  # Corner distance
) -> List[Dict[str, float]]:
    """
    Generate spawn points around the perimeter, outside the 10km radius.

    Args:
        num_points: Number of spawn points to generate
        min_radius_units: Minimum distance from center (default: 100 units = 10km)
        max_radius_units: Maximum distance from center (default: map corner)

    Returns:
        List of spawn point dictionaries with x, y, angle, distance
    """
    spawn_points = []

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points

        # Spawn at edge of map, outside min radius
        radius = max_radius_units

        x = MAP_CENTER[0] + radius * math.cos(angle)
        y = MAP_CENTER[1] + radius * math.sin(angle)

        # Clamp to map bounds
        x = max(0, min(MAP_SIZE, x))
        y = max(0, min(MAP_SIZE, y))

        # Calculate actual distance from center
        distance = math.sqrt((x - MAP_CENTER[0])**2 + (y - MAP_CENTER[1])**2)

        spawn_points.append({
            'x': x,
            'y': y,
            'angle_deg': math.degrees(angle),
            'distance_units': distance,
            'distance_km': distance * METERS_PER_UNIT / 1000.0,
        })

    return spawn_points

# Generate default spawn points (16 positions around perimeter)
SPAWN_POINTS = generate_spawn_points(16)

# ============================================================================
# ATTACKER SPAWN CONFIGURATION
# ============================================================================

ATTACKER_SPAWN_CONFIG = {
    'initial_count': 4,
    'spawn_probability_initial': 0.3,
    'spawn_probability_decay': 0.97,
    'spawn_edges': ['top', 'bottom', 'left', 'right'],
    'speed_min': ATTACKER_SPEED_MIN_UNITS,
    'speed_max': ATTACKER_SPEED_MAX_UNITS,
    'warhead_mass_min': ATTACKER_WARHEAD_MIN_KG,
    'warhead_mass_max': ATTACKER_WARHEAD_MAX_KG,
}

# ============================================================================
# COMPLETE CONFIGURATION DICTIONARY
# ============================================================================

REALISTIC_CONFIG = {
    'map': {
        'width': MAP_SIZE,
        'height': MAP_SIZE,
        'size_km': MAP_SIZE_KM,
        'center': MAP_CENTER,
    },
    'safe_zone': SAFE_ZONE,
    'safe_zone_size_km': SAFE_ZONE_SIZE_KM,

    'episode': {
        'max_timesteps': 600,  # 10 minutes at 1 sec/timestep
        'max_total_attackers': 50,
        'max_concurrent_attackers': 15,
    },

    'defender_stats': DEFENDER_STATS,
    'defenders': DEFENDER_POSITIONS,

    'attacker_spawn': ATTACKER_SPAWN_CONFIG,
    'spawn_points': SPAWN_POINTS,

    'visualization': {
        'fps_delay': 0.05,
        'figure_size': 12,
        'engagement_display_time': 10,
        'colors': {
            'sam': 'blue',
            'kinetic': 'darkorange',
            'attacker': 'red',
            'projectile': 'yellow',
            'engagement_highlight': 'gold',
            'safe_zone': 'green',
        },
    },
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_config_summary():
    """Print a human-readable summary of the realistic configuration."""
    print("=" * 80)
    print("REALISTIC RL SYSTEM CONFIGURATION")
    print("=" * 80)
    print()

    print("SCALE:")
    print(f"  1 abstract unit = {METERS_PER_UNIT}m")
    print(f"  Map size: {MAP_SIZE}×{MAP_SIZE} units = {MAP_SIZE_KM}km×{MAP_SIZE_KM}km")
    print(f"  Map center: {MAP_CENTER} = Airport location")
    print()

    print("SAFE ZONE (Airport Terminal):")
    print(f"  Size: {SAFE_ZONE_SIZE_KM}km × {SAFE_ZONE_SIZE_KM}km")
    print(f"  Size: {SAFE_ZONE_SIZE_UNITS}×{SAFE_ZONE_SIZE_UNITS} units")
    print(f"  Bounds: ({SAFE_ZONE['x_min']}, {SAFE_ZONE['y_min']}) to "
          f"({SAFE_ZONE['x_max']}, {SAFE_ZONE['y_max']})")
    print()

    print("SAM BATTERY:")
    print(f"  Range: {SAM_RANGE_KM}km → {SAM_RANGE_UNITS:.0f} units (capped at {SAM_RANGE_CAPPED:.0f})")
    print(f"  Speed: {SAM_SPEED_MS}m/s → {SAM_SPEED_UNITS_PER_TIMESTEP:.1f} units/timestep")
    print(f"  Reload: {SAM_RELOAD_TIME_SEC}s → {SAM_RELOAD_TIMESTEPS} timesteps")
    print(f"  Ammo: {SAM_AMMO} rounds")
    print(f"  P(kill): {SAM_PK}")
    print()

    print("KINETIC INTERCEPTOR:")
    print(f"  Range: {KINETIC_RANGE_KM}km → {KINETIC_RANGE_UNITS:.0f} units (covers entire map)")
    print(f"  Speed: {KINETIC_SPEED_MS}m/s → {KINETIC_SPEED_UNITS_PER_TIMESTEP:.2f} units/timestep")
    print(f"  Reload: {KINETIC_RELOAD_TIME_SEC}s → {KINETIC_RELOAD_TIMESTEPS} timesteps")
    print(f"  Ammo: {KINETIC_AMMO} rounds")
    print(f"  P(kill): {KINETIC_PK}")
    print()

    print("ATTACKER DRONES:")
    print(f"  Speed: {ATTACKER_SPEED_MIN_MS}-{ATTACKER_SPEED_MAX_MS}m/s → "
          f"{ATTACKER_SPEED_MIN_UNITS:.2f}-{ATTACKER_SPEED_MAX_UNITS:.2f} units/timestep")
    print(f"  Warhead: {ATTACKER_WARHEAD_MIN_KG}-{ATTACKER_WARHEAD_MAX_KG}kg")
    print(f"  Spawn: Outside {SPAWN_RADIUS_MIN_KM}km radius ({SPAWN_RADIUS_MIN_UNITS:.0f} units)")
    print()

    print("DEFENDER POSITIONS:")
    for defender in DEFENDER_POSITIONS:
        x, y = defender['x'], defender['y']
        distance = math.sqrt((x - MAP_CENTER[0])**2 + (y - MAP_CENTER[1])**2)
        distance_km = distance * METERS_PER_UNIT / 1000.0
        print(f"  {defender['label']:20s}: ({x:3.0f}, {y:3.0f}) = "
              f"{distance_km:.1f}km from center")
    print()


def verify_spawn_points():
    """Verify that all spawn points are outside the 10km radius."""
    print("=" * 80)
    print("SPAWN POINT VERIFICATION")
    print("=" * 80)
    print(f"Minimum spawn radius: {SPAWN_RADIUS_MIN_KM}km ({SPAWN_RADIUS_MIN_UNITS:.0f} units)")
    print()

    all_valid = True
    for i, point in enumerate(SPAWN_POINTS):
        x, y = point['x'], point['y']
        distance = point['distance_units']
        distance_km = point['distance_km']
        angle = point['angle_deg']

        valid = distance >= SPAWN_RADIUS_MIN_UNITS
        status = "✓" if valid else "✗ INVALID"

        print(f"  Point {i:2d}: ({x:6.1f}, {y:6.1f}) @ {angle:6.1f}° = "
              f"{distance:6.1f} units ({distance_km:5.2f}km) {status}")

        if not valid:
            all_valid = False

    print()
    if all_valid:
        print("✓ All spawn points are outside 10km radius")
    else:
        print("✗ Some spawn points are inside 10km radius!")
    print()

    return all_valid


def print_comparison_to_abstract():
    """Print comparison between abstract config and realistic config."""
    print("=" * 80)
    print("ABSTRACT vs REALISTIC CONFIGURATION")
    print("=" * 80)
    print()

    print(f"{'Parameter':<30s} {'Abstract':<20s} {'Realistic':<20s}")
    print("-" * 70)
    print(f"{'SAM Range':<30s} {'40 units':<20s} {f'{SAM_RANGE_CAPPED:.0f} units':<20s}")
    print(f"{'SAM Speed':<30s} {'15 units/timestep':<20s} {f'{SAM_SPEED_UNITS_PER_TIMESTEP:.1f} units/timestep':<20s}")
    print(f"{'SAM Reload':<30s} {'3 timesteps':<20s} {f'{SAM_RELOAD_TIMESTEPS} timesteps':<20s}")
    print(f"{'SAM Ammo':<30s} {'10 rounds':<20s} {f'{SAM_AMMO} rounds':<20s}")
    print()
    print(f"{'Kinetic Range':<30s} {'30 units':<20s} {f'{KINETIC_RANGE_UNITS:.0f} units':<20s}")
    print(f"{'Kinetic Speed':<30s} {'8 units/timestep':<20s} {f'{KINETIC_SPEED_UNITS_PER_TIMESTEP:.2f} units/timestep':<20s}")
    print(f"{'Kinetic Reload':<30s} {'1 timestep':<20s} {f'{KINETIC_RELOAD_TIMESTEPS} timesteps':<20s}")
    print(f"{'Kinetic Ammo':<30s} {'20 rounds':<20s} {f'{KINETIC_AMMO} rounds':<20s}")
    print()
    print(f"{'Attacker Speed':<30s} {'0.5-2.0 units/timestep':<20s} "
          f"{f'{ATTACKER_SPEED_MIN_UNITS:.2f}-{ATTACKER_SPEED_MAX_UNITS:.2f} units/timestep':<20s}")
    print(f"{'Safe Zone Size':<30s} {'20×20 units':<20s} {f'{SAFE_ZONE_SIZE_UNITS:.0f}×{SAFE_ZONE_SIZE_UNITS:.0f} units':<20s}")
    print()


def export_to_yaml(filename: str = 'realistic_config.yaml'):
    """Export configuration to YAML format for the partner's system."""
    import yaml

    # Convert to YAML-friendly format
    yaml_config = {
        'map': REALISTIC_CONFIG['map'],
        'safe_zone': REALISTIC_CONFIG['safe_zone'],
        'episode': REALISTIC_CONFIG['episode'],
        'defenders': [
            {
                'type': d['type'],
                'x': int(d['x']),
                'y': int(d['y']),
            }
            for d in DEFENDER_POSITIONS
        ],
        'defender_stats': REALISTIC_CONFIG['defender_stats'],
        'attacker_spawn': REALISTIC_CONFIG['attacker_spawn'],
        'visualization': REALISTIC_CONFIG['visualization'],
    }

    with open(filename, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Configuration exported to {filename}")


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == '__main__':
    print_config_summary()
    print()
    verify_spawn_points()
    print()
    print_comparison_to_abstract()
    print()

    # Export to YAML
    try:
        export_to_yaml('realistic_config.yaml')
        print()
    except ImportError:
        print("Note: Install PyYAML to export configuration: pip install pyyaml")
        print()

    print("=" * 80)
    print("CONFIGURATION READY FOR INTEGRATION")
    print("=" * 80)
    print("\nUsage:")
    print("  from src.integration.realistic_configs import REALISTIC_CONFIG")
    print("  env = DroneDefenseEnv(config_dict=REALISTIC_CONFIG)")
