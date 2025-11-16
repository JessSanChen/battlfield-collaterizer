# Drone Defense Wargame Simulator

A 2D reinforcement learning environment for simulating airport defense against drone attacks. Built for rapid prototyping and demonstration at hackathons.

## Features

- **Custom Gymnasium Environment**: Standard RL interface compatible with modern algorithms
- **Attention-Based Allocation**: Neural network using Query-Key-Value architecture for intelligent defender-attacker matching
- **Hungarian Algorithm**: Optimal resource allocation using linear assignment
- **Two Defender Types**:
  - SAM Batteries (long-range, high kill probability, slower reload)
  - Kinetic Drone Depots (medium-range, fast reload, moderate kill probability)
- **Heuristic Baseline**: Simple time-to-target × warhead mass prioritization
- **Real-time Visualization**: Matplotlib-based 2D rendering
- **Pre-integrated RL**: Ready to train with Stable-Baselines3 (PPO, A2C, DQN, etc.)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test to verify everything works
python quick_test.py
```

### Run Heuristic Demo

```bash
# Run 3 episodes with visualization
python demo_heuristic.py

# Run without visualization
python demo_heuristic.py --no-render

# Run 10 episodes
python demo_heuristic.py --episodes 10
```

### Run Attention Network Demo

```bash
# Run with untrained (random) network
python demo_attention.py

# Run with trained model
python demo_attention.py --model models/ppo_drone_defense_attention_network.pth
```

### Train RL Agent

```bash
# Train for 100k timesteps (default)
python train_rl.py --mode train

# Train for longer
python train_rl.py --mode train --timesteps 500000

# Evaluate trained agent
python train_rl.py --mode eval --eval-episodes 20
```

## Architecture

### Environment Structure

```
DroneDefenseEnv (Gymnasium)
├── Defenders (SAM + Kinetic Drones)
│   ├── Position, Range, Ammo
│   ├── Reload Time, Shot Speed
│   └── P(kill) = f(range, attacker speed)
│
├── Attackers (Drones)
│   ├── Position, Velocity
│   └── Warhead Mass (1-10)
│
├── Safe Zone (Protected Area)
└── Graph-based Engagement Model
```

### Attention Network Architecture

```
Defender Features [6D]
    ↓
QKV Network
    ├── Query [16D]
    ├── Key [16D]
    └── Value [16D]
    ↓
Attention Mechanism → Context Vectors [16D]
    ↓
[Defender + Context + Attacker] → Evaluation Network
    ↓
Score Matrix [num_defenders × num_attackers]
    ↓
Hungarian Algorithm → Optimal Allocation
```

### Reward Function

```
Reward Components:
  +10.0  per attacker destroyed
  +0.1   × ammo conservation ratio
  -50.0  per defender lost
  -200.0 if safe zone breached
```

## File Structure

```
.
├── entities.py              # Defender and Attacker classes
├── attention_network.py     # QKV, Attention, Evaluation networks
├── drone_defense_env.py     # Gymnasium environment
├── demo_heuristic.py        # Run heuristic-based demo
├── demo_attention.py        # Run attention-based demo
├── train_rl.py             # Train/evaluate RL agents
├── quick_test.py           # Quick verification test
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Customization for Hackathon

### Quick Tweaks for Demo

1. **Adjust Difficulty** (`drone_defense_env.py`):
   ```python
   # Line 36-38: Attacker spawning
   self.min_attackers = 5        # Initial wave size
   self.spawn_prob_initial = 0.3 # Spawn probability
   self.spawn_prob_decay = 0.95  # How quickly spawning decreases
   ```

2. **Change Defender Configuration** (`drone_defense_env.py:169`):
   ```python
   def _create_defenders(self):
       self.defenders = [
           SAMBattery(x=20, y=50),    # Add more defenders
           SAMBattery(x=80, y=50),
           KineticDroneDepot(x=50, y=30),
           KineticDroneDepot(x=50, y=70),
           # Add more here...
       ]
   ```

3. **Tune Reward Function** (`drone_defense_env.py:365`):
   ```python
   reward += attackers_killed * 10.0  # Increase to prioritize kills
   reward -= defenders_lost * 50.0    # Adjust penalty
   ```

### Extending for Advanced Features

1. **Add New Defender Types** (`entities.py`):
   - Inherit from `Defender` class
   - Customize `probability_of_kill()` method
   - Set unique parameters (range, ammo, reload time)

2. **Implement Multi-Agent RL**:
   - Use PettingZoo instead of Gymnasium
   - Each defender becomes an independent agent
   - Requires coordination learning

3. **Add Terrain/Obstacles**:
   - Modify `_construct_graph()` to consider line-of-sight
   - Update rendering to show obstacles

## Performance Tips

- **CPU Training**: ~100k timesteps in ~10-15 minutes
- **GPU Training**: 5-10x faster with CUDA-enabled PyTorch
- **Visualization**: Disable during training (`render_mode=None`)
- **Vectorized Envs**: Use `SubprocVecEnv` for parallel rollouts

## Hackathon Demo Strategy

1. **First Hour**: Run `demo_heuristic.py` to show baseline
2. **Next 3 Hours**: Train RL agent with `train_rl.py --timesteps 200000`
3. **Demo Prep**: Create comparison video (heuristic vs. RL)
4. **Pitch**: Emphasize attention mechanism and graph-based allocation

## Troubleshooting

**Issue**: Matplotlib window doesn't show
- **Fix**: Ensure you're not in headless environment, try `plt.show()` or different backend

**Issue**: Training is slow
- **Fix**: Reduce `max_timesteps` in environment, use fewer attackers, or enable GPU

**Issue**: All episodes end in defeat
- **Fix**: Reduce `spawn_prob_initial`, increase defender count, or adjust `max_range`

## References

- **Gymnasium**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Hungarian Algorithm**: `scipy.optimize.linear_sum_assignment`
- **Attention Mechanism**: Vaswani et al., "Attention Is All You Need" (2017)

## License

MIT License - Built for hackathon rapid prototyping

---

**Good luck at your hackathon!** 🚀