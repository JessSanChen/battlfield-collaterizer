# Configuration Guide

Quick reference for customizing the simulation for your hackathon demo.

## 🎨 Visualization Settings

### FPS / Animation Speed

**File**: `drone_defense_env.py`
**Line**: 666

```python
plt.pause(0.1)  # Change this value to adjust visualization speed
```

**Examples**:
- `0.05` = ~20 FPS (fast, harder to follow)
- `0.1` = ~10 FPS (default, good balance) ⭐
- `0.2` = ~5 FPS (slower, easier to see details)
- `0.5` = ~2 FPS (very slow, good for screenshots)
- `1.0` = ~1 FPS (slideshow mode)

**Recommendation**: Use 0.2 for demos so judges can see what's happening.

### Engagement Highlight Square

**File**: `drone_defense_env.py`
**Lines**: 622-625

```python
# Draw smaller square around target with darker yellow (gold)
highlight_rect = patches.Rectangle(
    (target_attacker.x - 4, target_attacker.y - 4),  # Position
    8, 8,                                             # Size (width, height)
    linewidth=2,                                      # Border thickness
    edgecolor='gold',                                 # Color (was 'yellow')
    facecolor='none',
    linestyle='--'
)
```

**To adjust**:
- **Size**: Change the `8, 8` (current) to make bigger/smaller
  - `6, 6` = smaller square
  - `10, 10` = bigger square
  - Offset should be half the size: `(x - 3, y - 3)` for 6×6

- **Color**: Change `edgecolor='gold'`
  - `'gold'` = darker yellow (current) ⭐
  - `'yellow'` = bright yellow
  - `'orange'` = orange
  - `'darkorange'` = darker orange
  - Or use hex: `'#FFD700'` (gold), `'#FFA500'` (orange)

---

## 🗺️ Map Configuration

### Safe Zone Position

**File**: `drone_defense_env.py`
**Line**: 27

```python
def __init__(self, map_size: Tuple[int, int] = (200, 200),
             safe_zone: Tuple[float, float, float, float] = (80, 80, 120, 120),
             #                                                 ^   ^    ^    ^
             #                                               x_min y_min x_max y_max
```

**Current safe zone**:
- Bottom-left corner: (80, 80)
- Top-right corner: (120, 120)
- Size: 40×40
- Center of 200×200 map

**Examples**:

```python
# Larger safe zone (easier to defend)
safe_zone = (60, 60, 140, 140)  # 80×80 square

# Smaller safe zone (harder to defend)
safe_zone = (90, 90, 110, 110)  # 20×20 square

# Off-center safe zone (top-left)
safe_zone = (40, 120, 80, 160)

# Off-center safe zone (bottom-right)
safe_zone = (120, 40, 160, 80)

# Rectangular safe zone
safe_zone = (70, 90, 130, 110)  # Wide rectangle
```

**Visual aid**:
```
200 ┌────────────────────────┐
    │                        │
    │    (80,120)──────┐     │
    │       │          │     │
100 │       │ SAFE ZONE│     │
    │       │          │     │
    │       └──────(120,80)  │
    │                        │
  0 └────────────────────────┘
    0        100            200
```

### Map Size

**File**: `drone_defense_env.py`
**Line**: 26

```python
def __init__(self, map_size: Tuple[int, int] = (200, 200),
             #                                    width height
```

**Examples**:
```python
map_size = (150, 150)  # Smaller, faster games
map_size = (200, 200)  # Current, good balance ⭐
map_size = (300, 300)  # Larger, longer games
map_size = (200, 150)  # Rectangular (wide)
```

**Important**: If you change map size, update:
1. Safe zone coordinates (to keep it centered)
2. Defender positions (see below)

---

## 🛡️ Defender Configuration

### Defender Positions

**File**: `drone_defense_env.py`
**Lines**: 97-105

```python
def _create_defenders(self):
    """Create initial defender configuration."""
    # Scale defender positions for larger map
    self.defenders = [
        SAMBattery(x=40, y=100),          # Left SAM
        SAMBattery(x=160, y=100),         # Right SAM
        KineticDroneDepot(x=100, y=60),   # Bottom Kinetic
        KineticDroneDepot(x=100, y=140),  # Top Kinetic
    ]
```

**Visual layout** (200×200 map):
```
200 ┌────────────────────────┐
    │                        │
    │        Kinetic         │
140 │          D3            │
    │                        │
    │   SAM  SAFE   SAM      │
100 │   D0   ZONE   D1       │
    │                        │
 60 │          D2            │
    │        Kinetic         │
  0 └────────────────────────┘
    0   40   100   160      200
```

### Adding More Defenders

**Same file**, same function:

```python
def _create_defenders(self):
    """Create initial defender configuration."""
    self.defenders = [
        # Existing 4 defenders
        SAMBattery(x=40, y=100),
        SAMBattery(x=160, y=100),
        KineticDroneDepot(x=100, y=60),
        KineticDroneDepot(x=100, y=140),

        # Add more here:
        SAMBattery(x=100, y=100),         # Center SAM
        KineticDroneDepot(x=50, y=100),   # Mid-left Kinetic
        KineticDroneDepot(x=150, y=100),  # Mid-right Kinetic
    ]
```

### Defender Types

**SAM Battery** (long-range, high damage):
```python
SAMBattery(x=50, y=150)
```
- Range: 40 units
- Ammo: 10 rounds
- Reload: 3 timesteps
- Shot speed: 15 units/timestep
- P(kill): 85% base

**Kinetic Drone Depot** (medium-range, fast reload):
```python
KineticDroneDepot(x=150, y=50)
```
- Range: 30 units
- Ammo: 20 rounds
- Reload: 1 timestep
- Shot speed: 8 units/timestep
- P(kill): 70% base

### Custom Defender Placement Patterns

**Perimeter Defense**:
```python
self.defenders = [
    SAMBattery(x=20, y=20),     # Bottom-left corner
    SAMBattery(x=180, y=20),    # Bottom-right corner
    SAMBattery(x=20, y=180),    # Top-left corner
    SAMBattery(x=180, y=180),   # Top-right corner
]
```

**Ring Around Safe Zone**:
```python
self.defenders = [
    SAMBattery(x=100, y=50),    # South
    SAMBattery(x=100, y=150),   # North
    SAMBattery(x=50, y=100),    # West
    SAMBattery(x=150, y=100),   # East
]
```

**Heavy Defense (8 defenders)**:
```python
self.defenders = [
    # Outer ring
    SAMBattery(x=40, y=40),
    SAMBattery(x=160, y=40),
    SAMBattery(x=40, y=160),
    SAMBattery(x=160, y=160),
    # Inner ring
    KineticDroneDepot(x=70, y=100),
    KineticDroneDepot(x=130, y=100),
    KineticDroneDepot(x=100, y=70),
    KineticDroneDepot(x=100, y=130),
]
```

---

## 🎯 Attacker Configuration

### Spawn Parameters

**File**: `drone_defense_env.py`
**Lines**: 28-29, 62-64

```python
# In __init__:
max_attackers: int = 20,          # Maximum concurrent attackers
max_timesteps: int = 300,         # Episode length

# Later in __init__:
self.min_attackers = 8            # Initial spawn count
self.spawn_prob_initial = 0.4     # Initial spawn probability
self.spawn_prob_decay = 0.95      # Decay rate per timestep
```

**To make easier**:
```python
self.min_attackers = 3
self.spawn_prob_initial = 0.2
self.max_attackers = 10
```

**To make harder**:
```python
self.min_attackers = 12
self.spawn_prob_initial = 0.6
self.max_attackers = 30
```

### Spawn Probability Over Time

The formula is: `spawn_prob = initial × (decay ^ timestep)`

```
Timestep  | Probability (0.4 initial, 0.95 decay)
----------|---------------------------------------
0-10      | ~40% (very high)
20        | ~14%
50        | ~3%
100       | ~0.6%
200+      | ~0% (basically stops)
```

**Adjust decay**:
- `0.90` = faster decay (attackers stop spawning sooner)
- `0.95` = balanced (current) ⭐
- `0.98` = slower decay (attackers keep coming longer)

---

## ⚡ Defender Stats

If you want to customize defender characteristics, modify the classes:

**File**: `entities.py`

### SAM Battery Stats
**Lines**: 104-117

```python
class SAMBattery(Defender):
    """Surface-to-Air Missile battery."""

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            max_range=40.0,      # ← Change range
            ammo=10,             # ← Change ammo capacity
            reload_time=3,       # ← Change reload speed (timesteps)
            shot_speed=15.0,     # ← Change missile speed
            defender_type='SAM'
        )
        self.base_pk = 0.85      # ← Change base kill probability (0-1)
```

### Kinetic Depot Stats
**Lines**: 120-133

```python
class KineticDroneDepot(Defender):
    """Kinetic drone depot that launches interceptor drones."""

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            max_range=30.0,      # ← Change range
            ammo=20,             # ← Change ammo capacity
            reload_time=1,       # ← Change reload speed
            shot_speed=8.0,      # ← Change drone speed
            defender_type='KINETIC'
        )
        self.base_pk = 0.70      # ← Change base kill probability
```

---

## 📊 Quick Reference Table

| Setting | File | Line | Current Value | Purpose |
|---------|------|------|---------------|---------|
| FPS | `drone_defense_env.py` | 666 | `0.1` | Visualization speed |
| Map Size | `drone_defense_env.py` | 26 | `(200, 200)` | Battlefield dimensions |
| Safe Zone | `drone_defense_env.py` | 27 | `(80,80,120,120)` | Protected area |
| Defender Positions | `drone_defense_env.py` | 100-104 | 4 positions | Where defenders spawn |
| Initial Attackers | `drone_defense_env.py` | 62 | `8` | Starting wave size |
| Max Attackers | `drone_defense_env.py` | 28 | `20` | Concurrent limit |
| Spawn Rate | `drone_defense_env.py` | 63 | `0.4` | New attacker probability |
| Highlight Color | `drone_defense_env.py` | 625 | `'gold'` | Engagement box color |
| Highlight Size | `drone_defense_env.py` | 624 | `8, 8` | Engagement box size |

---

## 🎬 Demo Presets

### Easy (for showing mechanics)
```python
# In drone_defense_env.py __init__:
map_size = (150, 150)
safe_zone = (60, 60, 90, 90)
max_attackers = 8
self.min_attackers = 3
self.spawn_prob_initial = 0.2

# In _create_defenders:
self.defenders = [
    SAMBattery(x=30, y=75),
    SAMBattery(x=120, y=75),
    KineticDroneDepot(x=75, y=40),
    KineticDroneDepot(x=75, y=110),
    KineticDroneDepot(x=75, y=75),  # Extra defender
]

# For visualization:
plt.pause(0.2)  # Slower for clarity
```

### Hard (for showing AI capabilities)
```python
# In drone_defense_env.py __init__:
map_size = (250, 250)
safe_zone = (100, 100, 150, 150)
max_attackers = 30
self.min_attackers = 15
self.spawn_prob_initial = 0.5

# In _create_defenders:
self.defenders = [
    SAMBattery(x=50, y=125),
    SAMBattery(x=200, y=125),
    KineticDroneDepot(x=125, y=75),
    KineticDroneDepot(x=125, y=175),
]

# For visualization:
plt.pause(0.05)  # Faster, more action
```

### Cinematic (for recording)
```python
map_size = (200, 200)
safe_zone = (80, 80, 120, 120)
max_attackers = 15
plt.pause(0.3)  # Slow enough to see everything
```

---

## 💡 Tips

1. **Keep safe zone centered**: For best visuals, center = (map_width/2, map_height/2)
2. **Defender coverage**: Place defenders so range circles overlap slightly
3. **FPS for demos**: 0.15-0.2 is ideal for live presentations
4. **Testing**: Use `--episodes 1` to quickly test configuration changes
5. **Screenshots**: Use `plt.pause(1.0)` and press pause on interesting frames

---

## 🔄 After Changing Configurations

Remember to test:
```bash
python3 demo_heuristic.py --episodes 1
```

Watch for:
- Are defenders positioned correctly?
- Is safe zone in the right place?
- Is the speed comfortable to watch?
- Do engagement highlights look good?

---

**All configuration points documented!** Edit the values above to customize your demo.
