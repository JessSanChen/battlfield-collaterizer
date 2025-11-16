# System Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     GAME ENVIRONMENT                         │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Defenders   │         │  Attackers   │                 │
│  │              │         │              │                 │
│  │ • SAM x2     │         │ • Spawn from │                 │
│  │ • Kinetic x2 │         │   edges      │                 │
│  │              │         │ • Move toward│                 │
│  │ [x,y,ammo,   │         │   safe zone  │                 │
│  │  range,...]  │         │              │                 │
│  └──────────────┘         │ [x,y,vx,vy,  │                 │
│         │                 │  warhead]    │                 │
│         │                 └──────────────┘                 │
│         │                        │                          │
│         └────────┬───────────────┘                          │
│                  │                                          │
│         ┌────────▼────────┐                                │
│         │ Graph Builder   │                                │
│         │                 │                                │
│         │ Adjacency[i,j]= │                                │
│         │  1 if in range  │                                │
│         └────────┬────────┘                                │
│                  │                                          │
└──────────────────┼──────────────────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    │   TWO ALLOCATION MODES:     │
    │                             │
┌───▼──────────────┐   ┌──────────▼──────────┐
│    HEURISTIC     │   │  ATTENTION NETWORK  │
│                  │   │                     │
│  For each pair:  │   │  ┌───────────────┐ │
│                  │   │  │ QKV Network   │ │
│  score =         │   │  │               │ │
│    time_to_zone  │   │  │ Q,K,V = f(D) │ │
│    × warhead     │   │  └───────┬───────┘ │
│                  │   │          │         │
│  Result:         │   │  ┌───────▼───────┐ │
│  Score Matrix    │   │  │  Attention    │ │
│  [D × A]         │   │  │  Mechanism    │ │
│                  │   │  │               │ │
└──────┬───────────┘   │  │ Context[16D]  │ │
       │               │  └───────┬───────┘ │
       │               │          │         │
       │               │  ┌───────▼───────┐ │
       │               │  │  Evaluation   │ │
       │               │  │   Network     │ │
       │               │  │               │ │
       │               │  │ score(D+C, A) │ │
       │               │  └───────┬───────┘ │
       │               │          │         │
       │               │    Score Matrix    │
       │               │      [D × A]       │
       │               └──────────┬─────────┘
       │                          │
       └──────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │    Hungarian    │
         │    Algorithm    │
         │                 │
         │ Optimal pairs:  │
         │ [(D1,A3),       │
         │  (D2,A1),       │
         │  (D3,A2), ...]  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Execute Attacks │
         │                 │
         │ For each pair:  │
         │  - Check P(kill)│
         │  - Roll dice    │
         │  - Update state │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Compute Reward  │
         │                 │
         │ +kill -breach   │
         │ +ammo -defender │
         └────────┬────────┘
                  │
                  ▼
            (Next Timestep)
```

## Entity Relationships

```
┌─────────────────────────────────────┐
│          Defender (Base)            │
├─────────────────────────────────────┤
│ • x, y: float                       │
│ • max_range: float                  │
│ • ammo: int                         │
│ • reload_time: int                  │
│ • shot_speed: float                 │
│ • type: str                         │
├─────────────────────────────────────┤
│ + probability_of_kill(d, v) → float│
│ + can_shoot() → bool                │
│ + shoot() → void                    │
│ + to_feature_vector() → [6D]        │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───────┐ ┌──▼─────────────┐
│ SAMBattery│ │ KineticDepot   │
├───────────┤ ├────────────────┤
│ range: 40 │ │ range: 30      │
│ ammo: 10  │ │ ammo: 20       │
│ reload: 3 │ │ reload: 1      │
│ speed: 15 │ │ speed: 8       │
│ pk: 0.85  │ │ pk: 0.70       │
└───────────┘ └────────────────┘

┌─────────────────────────────────────┐
│            Attacker                 │
├─────────────────────────────────────┤
│ • x, y: float                       │
│ • vx, vy: float                     │
│ • warhead_mass: float (1-10)        │
│ • alive: bool                       │
├─────────────────────────────────────┤
│ + update() → void                   │
│ + get_speed() → float               │
│ + to_feature_vector() → [6D]        │
└─────────────────────────────────────┘
```

## Attention Mechanism Detail

```
Step 1: Feature Extraction
─────────────────────────
Defender i: [x, y, ammo_ratio, reload, type_sam, type_kinetic] → 6D

Step 2: QKV Transform
─────────────────────
       ┌────────────┐
  D ──►│ FC(6→32)   │
       │ ReLU       │
       │ FC(32→32)  │
       │ ReLU       │──┬──► Q = FC(32→16)
       └────────────┘  ├──► K = FC(32→16)
                       └──► V = FC(32→16)

Step 3: Self-Attention
──────────────────────
       Q @ K^T
  A = ─────────    [num_defenders × num_defenders]
       sqrt(16)

  Context = softmax(A) @ V    [num_defenders × 16]

Step 4: Evaluation
──────────────────
  For each (defender i, attacker j):

    Input: [defender_i [6D], context_i [16D], attacker_j [6D]]
           = [28D total]

    ┌────────────┐
    │ FC(28→64)  │
    │ ReLU       │
    │ FC(64→64)  │
    │ ReLU       │
    │ FC(64→32)  │
    │ ReLU       │
    │ FC(32→1)   │  ──► score_{i,j}
    └────────────┘

Step 5: Hungarian
─────────────────
  Cost = -score    (negate because we maximize)

  Solve: min Σ cost[i, assignment[i]]
         subject to: each defender assigned to ≤1 attacker
                     each attacker assigned to ≤1 defender
```

## Reward Function Detail

```python
def calculate_reward(timestep_state):
    reward = 0.0

    # 1. Attacker kills (primary objective)
    reward += attackers_killed_this_step * 10.0

    # 2. Ammo conservation (efficiency)
    ammo_ratio = total_ammo / max_total_ammo
    reward += ammo_ratio * 0.1

    # 3. Defender losses (expensive)
    reward -= defenders_lost * 50.0

    # 4. Safe zone breach (catastrophic)
    if safe_zone_breached:
        reward -= 200.0

    return reward

# Example:
# - Kill 3 attackers: +30
# - 80% ammo remaining: +0.08
# - No losses: 0
# - No breach: 0
# Total: +30.08
```

## Episode Lifecycle

```
┌────────────────────────────────────────────┐
│ 1. RESET                                   │
│    • Create 4 defenders (2 SAM, 2 Kinetic) │
│    • Spawn 5 initial attackers             │
│    • timestep = 0                          │
└──────────────┬─────────────────────────────┘
               │
┌──────────────▼─────────────────────────────┐
│ 2. STEP LOOP (max 200 timesteps)           │
│    ┌─────────────────────────────────────┐ │
│    │ a. Spawn new attackers (prob decay) │ │
│    │ b. Build adjacency graph            │ │
│    │ c. Compute score matrix             │ │
│    │ d. Hungarian allocation             │ │
│    │ e. Execute attacks (P(kill) rolls)  │ │
│    │ f. Update entities (move, reload)   │ │
│    │ g. Check termination                │ │
│    │ h. Compute reward                   │ │
│    └─────────────────────────────────────┘ │
│         │                                   │
│         └──► Continue or Exit?              │
└───────────────────┬────────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    │                               │
┌───▼─────────┐          ┌──────────▼────────┐
│ TERMINATE   │          │ TRUNCATE          │
│ • Breach    │          │ • Max timesteps   │
│ • All dead  │          │                   │
└─────────────┘          └───────────────────┘
```

## Training Loop (PPO)

```
┌─────────────────────────────────────────┐
│ 1. Initialize                           │
│    • Create env                         │
│    • Create PPO agent                   │
│    • Random attention network weights   │
└──────────┬──────────────────────────────┘
           │
┌──────────▼──────────────────────────────┐
│ 2. Rollout Phase (n_steps=2048)         │
│    • Run episodes                       │
│    • Collect (s, a, r, s')              │
│    • Store in buffer                    │
└──────────┬──────────────────────────────┘
           │
┌──────────▼──────────────────────────────┐
│ 3. Update Phase (n_epochs=10)           │
│    • Sample minibatches (64)            │
│    • Compute advantages (GAE)           │
│    • Update policy (clip)               │
│    • Update value function              │
│    • Update attention network (backprop)│
└──────────┬──────────────────────────────┘
           │
           └──► Repeat until total_timesteps

Final: Save PPO model + Attention weights
```

## Deployment for Demo

```
┌───────────────────────────────────────┐
│ Heuristic Mode                        │
│ • use_attention = False               │
│ • score = time × mass                 │
│ • Good baseline                       │
│ • Fast, deterministic                 │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│ Random Network Mode                   │
│ • use_attention = True                │
│ • No training                         │
│ • Shows architecture works            │
│ • Worse than heuristic                │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│ Trained RL Mode                       │
│ • use_attention = True                │
│ • Load trained weights                │
│ • Best performance                    │
│ • Learned coordination patterns       │
└───────────────────────────────────────┘
```

## Key Design Decisions

### Why Graph-Based?
- Natural representation of engagement zones
- Sparse connections (only in-range pairs)
- Scales to many defenders/attackers

### Why Attention?
- Defenders need to coordinate (not act independently)
- Context-aware decision making
- State-of-the-art in sequence modeling
- Impressive for judges

### Why Hungarian?
- Optimal allocation (provably)
- Fast (O(n³) but n is small)
- Deterministic given scores
- Well-studied algorithm

### Why Two Modes?
- Baseline for comparison
- Demonstrates improvement
- Fallback if training fails
- Explains problem before solution

## Performance Characteristics

| Component           | Complexity | Typical Time  |
|---------------------|-----------|---------------|
| Graph Construction  | O(D × A)  | <1ms          |
| Attention Forward   | O(D²)     | ~5ms (CPU)    |
| Evaluation Network  | O(D × A)  | ~10ms (CPU)   |
| Hungarian          | O(D³)     | <1ms (D≤10)   |
| **Total per step** |           | ~20ms         |
| Episode (200 steps)|           | ~4s           |
| Training (100k)    |           | ~10-15 min    |

## Extension Points

1. **New Defender Types**: Inherit from `Defender`, set parameters
2. **New Reward Terms**: Modify `_calculate_reward()`
3. **Terrain**: Add obstacles to `_construct_graph()`
4. **Multi-Agent**: Switch to PettingZoo, one policy per defender
5. **3D**: Add z-dimension, update distance calculations
6. **Real Data**: Load from JSON, use actual coordinates
7. **Uncertainty**: Add sensor noise, imperfect information
8. **Communication**: Explicit message passing between defenders

---

This architecture is designed for:
- ✅ Rapid development (hackathon)
- ✅ Easy demonstration
- ✅ Technical depth
- ✅ Extensibility
- ✅ Impressive visuals
