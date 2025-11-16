# Visualization Guide

## Display Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Drone Defense - Timestep 42                         [LEGEND]        │
│                                                      ■ SAM Battery   │
│ ┌─Stats──────────┐                                  ■ Kinetic Depot │
│ │Attackers: 12   │                                  ▲ Enemy Drone   │
│ │alive, 3 dest.  │                                  ● Projectile    │
│ │Defenders: 4    │                                                  │
│ │Projectiles: 5  │                                                  │
│ └────────────────┘                                                  │
│                                                                      │
│                           D0                                         │
│                           ■  ← Blue SAM                             │
│                      Ammo:7 (R:2)                                   │
│                       ┆ ┆ ┆  ← Reload timer                         │
│                       └─┴─┘  ← Range circle (dashed)                │
│                                                                      │
│                  A5                                                  │
│     ┌─────┐      ▲  ← Red triangle                                 │
│     │     │   WH:8.3 ← Warhead mass                                │
│     └─────┘      ↑   ← Velocity arrow                              │
│     Yellow box   │                                                  │
│     ← D2         │                                                  │
│     Engagement   │                                                  │
│     highlight    │                                                  │
│                  │                                                  │
│                  ●────  ← Projectile with trail                     │
│                                                                      │
│              D2                D3                                    │
│              ■                 ■   ← Orange Kinetic                 │
│           Ammo:15           Ammo:18                                 │
│                                                                      │
│                    ┌─────────────┐                                  │
│                    │             │                                  │
│                    │  SAFE ZONE  │ ← Green box                      │
│                    │   (Airport) │                                  │
│                    └─────────────┘                                  │
│                                                                      │
│                           D1                                         │
│                           ■  ← Blue SAM                             │
│                        Ammo:9                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Entity Reference

### Defenders

#### SAM Battery (Blue Square ■)
```
       D0          ← ID label (white background)
       ■           ← Blue square with black outline
  Ammo:7 (R:2)     ← Status: 7 rounds, reloading (2 steps left)

   or when ready:

       D0
       ■
     Ammo:10       ← Status: 10 rounds, ready to fire
```

**Characteristics**:
- Color: Blue
- Shape: Square
- Range: 40 units (blue dashed circle)
- Ammo: 10 rounds
- Reload: 3 timesteps
- Shot speed: 15 units/timestep
- P(kill) base: 85%

#### Kinetic Drone Depot (Orange Square ■)
```
       D2          ← ID label
       ■           ← Orange square with black outline
     Ammo:15       ← Status: 15 rounds ready
```

**Characteristics**:
- Color: Orange (darkorange)
- Shape: Square
- Range: 30 units (orange dashed circle)
- Ammo: 20 rounds
- Reload: 1 timestep
- Shot speed: 8 units/timestep
- P(kill) base: 70%

### Attackers

#### Enemy Drone (Red Triangle ▲)
```
       A5          ← ID label (white background)
       ▲           ← Red triangle with dark red outline
     ↗             ← Velocity arrow (shows direction & speed)
    WH:8.3         ← Warhead mass (1.0 to 10.0)
```

**Characteristics**:
- Color: Red
- Shape: Triangle (pointing up)
- Speed: 0.5 to 2.0 units/timestep
- Warhead: 1-10 (random)
- Spawns from edges
- Moves toward safe zone

### Projectiles

#### In-Flight Projectile (Orange Circle ●)
```
    ●────          ← Orange circle with trailing line
    projectile     ← Trail shows recent path
```

**Characteristics**:
- Color: Orange
- Shape: Small circle
- Trail: Shows last 3 positions
- Speed: Defender-dependent (8 or 15)
- Outcome: Hit or miss on impact

## Engagement System

### Active Engagement Visualization
```
       D1 is engaging A5:

       D1              A5
       ■           ┌─────┐
     Ammo:8        │  ▲  │  ← Yellow dashed box
                   │WH:7 │
                   └─────┘
                     ↖
                    ← D1  ← Yellow label (black background)

    Plus projectile traveling from D1 to A5:

       ■  ●────────→ ▲
       D1 projectile  A5
```

**Duration**: Highlight persists for 10 timesteps after shot

### Multiple Engagements
```
    Two defenders targeting same attacker:

    D0                D1
    ■                 ■

           ┌─────┐
           │  ▲  │  A5
           │WH:9 │
           └─────┘
          ← D0
          ← D1

    (Both labels shown)
```

## Status Indicators

### Defender Status States

1. **Ready to Fire**:
   ```
   D0
   ■
   Ammo:10
   ```

2. **Reloading**:
   ```
   D0
   ■
   Ammo:7 (R:3)  ← "R:3" means 3 timesteps until ready
   ```

3. **Out of Ammo**:
   ```
   D0
   ■
   Ammo:0 (R:3)  ← Still reloading, will have 1 round after
   ```

### Attacker States

1. **Approaching**:
   ```
   A5
   ▲
   ↗
   WH:8.3
   ```

2. **Being Targeted**:
   ```
   ┌─────┐
   │ A5  │
   │  ▲  │  ← Yellow box
   │WH:8 │
   └─────┘
   ← D2
   ```

3. **Hit by Projectile**:
   ```
   ●  ← Projectile impacts
   ▲  ← Attacker disappears (if hit)
      ← OR survives (if miss)
   ```

## Information Displays

### Top-Left Stats Box
```
┌──────────────────┐
│Attackers: 12     │ ← Alive count
│alive, 3 dest.    │ ← Destroyed count
│Defenders: 4 alive│ ← Active defenders
│Projectiles: 5    │ ← In-flight count
└──────────────────┘
```

### Legend (Top-Right)
```
■ SAM Battery       ← Blue square
■ Kinetic Depot     ← Orange square
▲ Enemy Drone       ← Red triangle
● Projectile        ← Orange circle
```

## Color Codes

| Element | Color | Meaning |
|---------|-------|---------|
| Blue | SAM Battery | High P(kill), long range, slow reload |
| Orange | Kinetic Depot | Medium P(kill), medium range, fast reload |
| Red | Enemy Attacker | Threat level varies by warhead mass |
| Yellow | Engagement Box | Active targeting |
| Orange | Projectile | In-flight ordnance |
| Green | Safe Zone | Protected area |
| Black | Outline | Entity border |
| White | ID Background | Label clarity |

## Range Circles

Dashed circles around defenders show maximum engagement range:
```
       D0
       ■
     /   \
    |  ·  |  ← Blue dashed circle (SAM: 40 units)
     \   /

       D2
       ■
     / \    ← Orange dashed circle (Kinetic: 30 units)
    |   |
     \ /
```

**Range represents**: Maximum distance at which defender can engage

## Velocity Arrows

Arrows on attackers show movement:
```
   ↑   ← Fast (long arrow)
   ▲

   →   ← Medium
   ▲

   ↘   ← Slow (short arrow)
   ▲
```

**Arrow length** ∝ Speed
**Arrow direction** = Movement direction

## Projectile Trails

Trails show recent flight path:
```
   ●────          Short trail (just fired)

   ●────────      Medium trail

   ●──────────→   Long trail (about to hit)
```

## Example Scenarios

### Scenario 1: Successful Interception
```
Timestep 10:           Timestep 15:           Timestep 18:
    D0                     D0                     D0
    ■                      ■                      ■
  Ammo:10                Ammo:9                 Ammo:8

              ┌─────┐          ●────        (attacker destroyed)
              │  ▲  │A5          ▲ A5
              └─────┘         ┌─────┐
              ← D0            └─────┘
                              ← D0

Defender engages → Projectile launched → Hit! (85% chance)
```

### Scenario 2: Miss
```
Timestep 10:           Timestep 15:           Timestep 18:
    D2                     D2                     D2
    ■                      ■                      ■
  Ammo:20                Ammo:19 (R:1)          Ammo:19

              ┌─────┐          ●────             ▲ A8
              │  ▲  │A8          ▲ A8          WH:7.2
              └─────┘         ┌─────┐         (still alive)
              ← D2            └─────┘
                              ← D2

Defender engages → Projectile launched → Miss! (30% chance)
```

### Scenario 3: Multi-Defender Coordination
```
    D0              D1
    ■               ■
  Ammo:9          Ammo:8

    ●────      ────●
         \    /
          ▼  ▼
       ┌─────┐
       │  ▲  │ A3
       │WH:9 │  ← High-priority target
       └─────┘
        ← D0
        ← D1

Both defenders engage high-value target
```

## Tips for Observers

1. **Watch the IDs**: Track specific entities across timesteps
2. **Monitor ammo**: Defenders with low ammo are less effective
3. **Yellow boxes**: Show AI decision-making in real-time
4. **Projectile trails**: Visualize interception attempts
5. **Miss vs. Hit**: Not all projectiles succeed (realistic!)
6. **Range circles**: Show defender coverage areas

## Performance Indicators

### Good Defense:
- Most attackers destroyed before safe zone
- Defenders maintain ammo reserves
- Minimal breaches
- Coordinated engagements (yellow boxes)

### Struggling Defense:
- Many attackers reaching safe zone
- Defenders out of ammo
- Projectiles missing frequently
- Uncoordinated (no yellow boxes)

---

**Use this guide to understand the real-time visualization during demos!**
