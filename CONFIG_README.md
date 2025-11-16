# YAML Configuration Guide

All simulation parameters are now centralized in `config.yaml` for easy tuning!

## 📝 Quick Start

All parameters are in **config.yaml**. Just edit the values and run:

```bash
python3 demo_heuristic.py
```

The environment automatically loads the config file.

## 📋 Configuration Sections

### 1. Map Configuration
```yaml
map:
  width: 200      # Map width in units
  height: 200     # Map height in units

safe_zone:
  x_min: 80       # Bottom-left corner
  y_min: 80
  x_max: 120      # Top-right corner
  y_max: 120
```

### 2. Episode Configuration
```yaml
episode:
  max_timesteps: 300                # Maximum episode length
  max_total_attackers: 50           # Total attackers per episode (NEW!)
  max_concurrent_attackers: 20      # Max alive at once
```

**New Feature**: `max_total_attackers` limits total spawns per episode, preventing infinite games.

### 3. Defender Positions
```yaml
defenders:
  - type: SAM           # Type: SAM or KINETIC
    x: 70
    y: 50

  - type: KINETIC
    x: 50
    y: 20

  # Add more defenders here...
```

**To add defenders**: Just add more entries to the list!

### 4. Defender Stats
```yaml
defender_stats:
  SAM:
    max_range: 40.0
    ammo: 10
    reload_time: 3
    shot_speed: 15.0
    base_probability_of_kill: 0.85

  KINETIC:
    max_range: 30.0
    ammo: 20
    reload_time: 1
    shot_speed: 8.0
    base_probability_of_kill: 0.70
```

**Note**: Changes affect ALL defenders of that type.

### 5. Attacker Spawn Configuration
```yaml
attacker_spawn:
  initial_count: 8                      # Starting attackers
  spawn_probability_initial: 0.4        # Spawn chance per timestep
  spawn_probability_decay: 0.95         # Decay rate (0-1)

  spawn_edges:                          # Where attackers spawn
    - top
    - bottom
    - left
    - right

  speed_min: 0.5                        # Attacker speed range
  speed_max: 2.0

  warhead_mass_min: 1.0                 # Warhead mass range
  warhead_mass_max: 10.0
```

**To disable an edge**: Remove it from `spawn_edges` list

### 6. Visualization Settings
```yaml
visualization:
  fps_delay: 0.1                        # Animation speed (lower=faster)
  figure_size: 10                       # Figure size in inches
  engagement_display_time: 10           # Highlight duration

  colors:
    sam: blue
    kinetic: darkorange
    attacker: red
    projectile: orange
    engagement_highlight: gold
    safe_zone: green

  # Marker sizes
  highlight_box_size: 8
  marker_size_defender: 14
  marker_size_attacker: 12
  marker_size_projectile: 5
```

## 🎯 Common Tuning Scenarios

### Make Game Easier
```yaml
defender_stats:
  SAM:
    ammo: 20                 # More ammo
    base_probability_of_kill: 0.95  # Higher hit rate

attacker_spawn:
  spawn_probability_initial: 0.2      # Fewer spawns
  speed_max: 1.5                      # Slower attackers

episode:
  max_total_attackers: 30             # Fewer total enemies
```

### Make Game Harder
```yaml
episode:
  max_total_attackers: 100            # More enemies

attacker_spawn:
  initial_count: 15                   # Larger initial wave
  spawn_probability_initial: 0.6      # More frequent spawns
  speed_min: 1.0                      # Faster attackers
  speed_max: 3.0

defender_stats:
  SAM:
    ammo: 5                           # Less ammo
    reload_time: 5                    # Slower reload
```

### Speed Up Visualization
```yaml
visualization:
  fps_delay: 0.05        # Fast (20 FPS)
```

### Slow Down for Demo
```yaml
visualization:
  fps_delay: 0.3         # Slow (3 FPS, easy to follow)
```

### Add More Defenders
```yaml
defenders:
  # Existing defenders...

  - type: SAM
    x: 100
    y: 100

  - type: KINETIC
    x: 75
    y: 75

  # Add as many as you want!
```

### Spawn Only from Top/Bottom
```yaml
attacker_spawn:
  spawn_edges:
    - top
    - bottom
    # Removed left and right
```

## 🔧 Using Config in Code

The environment automatically loads config:

```python
# Default: loads config.yaml
env = DroneDefenseEnv()

# Custom config file:
env = DroneDefenseEnv(config_path="my_config.yaml")
```

All demo scripts automatically use the config file!

## 📊 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map.width` | int | 200 | Map width |
| `map.height` | int | 200 | Map height |
| `safe_zone.x_min` | float | 80 | Safe zone left |
| `safe_zone.y_min` | float | 80 | Safe zone bottom |
| `safe_zone.x_max` | float | 120 | Safe zone right |
| `safe_zone.y_max` | float | 120 | Safe zone top |
| `episode.max_timesteps` | int | 300 | Episode length |
| `episode.max_total_attackers` | int | 50 | Total spawns limit |
| `episode.max_concurrent_attackers` | int | 20 | Concurrent limit |
| `attacker_spawn.initial_count` | int | 8 | Starting attackers |
| `attacker_spawn.spawn_probability_initial` | float | 0.4 | Initial spawn rate |
| `attacker_spawn.spawn_probability_decay` | float | 0.95 | Decay rate (0-1) |
| `attacker_spawn.speed_min` | float | 0.5 | Min attacker speed |
| `attacker_spawn.speed_max` | float | 2.0 | Max attacker speed |
| `attacker_spawn.warhead_mass_min` | float | 1.0 | Min warhead mass |
| `attacker_spawn.warhead_mass_max` | float | 10.0 | Max warhead mass |
| `defender_stats.SAM.max_range` | float | 40.0 | SAM range |
| `defender_stats.SAM.ammo` | int | 10 | SAM ammo |
| `defender_stats.SAM.reload_time` | int | 3 | SAM reload |
| `defender_stats.SAM.shot_speed` | float | 15.0 | SAM missile speed |
| `defender_stats.SAM.base_probability_of_kill` | float | 0.85 | SAM base P(kill) |
| `defender_stats.KINETIC.max_range` | float | 30.0 | Kinetic range |
| `defender_stats.KINETIC.ammo` | int | 20 | Kinetic ammo |
| `defender_stats.KINETIC.reload_time` | int | 1 | Kinetic reload |
| `defender_stats.KINETIC.shot_speed` | float | 8.0 | Kinetic drone speed |
| `defender_stats.KINETIC.base_probability_of_kill` | float | 0.70 | Kinetic base P(kill) |
| `visualization.fps_delay` | float | 0.1 | Animation delay |
| `visualization.figure_size` | int | 10 | Figure size (inches) |
| `visualization.engagement_display_time` | int | 10 | Highlight duration |
| `visualization.highlight_box_size` | int | 8 | Engagement box size |
| `visualization.marker_size_defender` | int | 14 | Defender marker size |
| `visualization.marker_size_attacker` | int | 12 | Attacker marker size |
| `visualization.marker_size_projectile` | int | 5 | Projectile marker size |

## 🎨 Color Options

Supported colors:
- `red`, `blue`, `green`, `yellow`, `orange`, `purple`, `pink`
- `darkred`, `darkblue`, `darkgreen`, `darkorange`
- `cyan`, `magenta`, `brown`, `gray`, `black`, `white`
- `gold`, `silver`, `navy`, `teal`, `lime`

Or use hex codes: `'#FF5733'`

## 💡 Tips

1. **Make a backup** of `config.yaml` before experimenting
2. **Start small**: Change one parameter at a time
3. **Test quickly**: Use `--episodes 1` to test changes
4. **Copy configs**: Create `easy_config.yaml`, `hard_config.yaml` etc.
5. **Version control**: Commit working configs to git

## 🚨 Common Mistakes

❌ **Wrong indentation** (YAML is sensitive!)
```yaml
defenders:
- type: SAM    # Wrong! Should be indented
  x: 50
```

✅ **Correct**:
```yaml
defenders:
  - type: SAM  # Correct indentation
    x: 50
```

❌ **Missing colon**:
```yaml
sam blue     # Wrong!
```

✅ **Correct**:
```yaml
sam: blue    # Correct
```

❌ **String without quotes** (for special characters):
```yaml
color: #FF5733  # Wrong! # is a comment
```

✅ **Correct**:
```yaml
color: '#FF5733'  # Correct
```

## 🔄 Reloading Config

Config is loaded once when environment is created. To use new config:

```bash
# Just restart the demo
python3 demo_heuristic.py
```

No code changes needed - just edit config.yaml!

---

**All parameters are now tunable without touching code!** 🎉
