# Quick Start Guide - 10-Hour Hackathon Edition

## 1. Setup (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# OR use pip3 if python is python3:
pip3 install -r requirements.txt

# Verify installation
python3 quick_test.py
```

## 2. Immediate Demo (First Hour)

### Option A: Heuristic Baseline
```bash
# Show the simple heuristic approach (good for explaining the problem)
python3 demo_heuristic.py --episodes 3
```

**What to show judges:**
- Red dots = attacking drones (number shows warhead mass)
- Blue triangles = SAM batteries (numbers show ammo)
- Cyan squares = Kinetic drone depots
- Green box = protected safe zone (airport)
- Strategy: Prioritize closest threats with heaviest warheads

### Option B: Attention Network (Untrained)
```bash
# Show the neural network approach (even untrained)
python3 demo_attention.py --episodes 3
```

**What to show judges:**
- Same visualization, but using attention mechanism
- Explain: "This uses Query-Key-Value attention to learn defender coordination"
- Even random weights show the architecture works

## 3. Training Phase (Hours 2-5)

```bash
# Start training in background (this will take 1-2 hours)
python3 train_rl.py --mode train --timesteps 200000 > training.log 2>&1 &

# Check progress
tail -f training.log

# Or train faster with fewer timesteps for demo
python3 train_rl.py --mode train --timesteps 50000
```

**While training:**
- Work on your pitch deck
- Prepare comparison metrics
- Create demo scenarios

## 4. Evaluation (Hour 6)

```bash
# Evaluate the trained model
python3 train_rl.py --mode eval --eval-episodes 20

# Run demo with trained model
python3 demo_attention.py --model models/ppo_drone_defense_attention_network.pth --episodes 5
```

## 5. Demo Preparation (Hours 7-9)

### Create Comparison Video
```bash
# Terminal 1: Heuristic
python3 demo_heuristic.py --episodes 5 > heuristic_results.txt

# Terminal 2: RL (after training)
python3 demo_attention.py --model models/ppo_drone_defense_attention_network.pth --episodes 5 > rl_results.txt
```

### Key Metrics to Compare:
- Victory rate (% of episodes where safe zone not breached)
- Average attackers destroyed
- Average ammo efficiency
- Average episode length

## 6. Pitch Strategy (Hour 10)

### 30-Second Hook
"We built an AI-powered drone defense system using cutting-edge attention mechanisms and reinforcement learning. Our system learns to optimally allocate defensive resources in real-time to protect critical infrastructure from drone swarms."

### 2-Minute Technical Deep Dive

1. **Problem**: Defending airports from coordinated drone attacks requires real-time decision making
2. **Innovation**:
   - Graph-based engagement model
   - Attention mechanism for defender coordination
   - Hungarian algorithm for optimal allocation
3. **Results**: Show comparison (heuristic vs RL)
4. **Scalability**: Easily extends to more defender types, 3D, etc.

### Demo Flow
1. Show heuristic baseline (30 sec)
2. Explain attention architecture (1 min)
3. Show trained RL agent (30 sec)
4. Compare metrics (30 sec)
5. Q&A

## Emergency Shortcuts

### If training is too slow:
```python
# In drone_defense_env.py, reduce complexity:
max_attackers=8,           # Instead of 15
max_timesteps=100,         # Instead of 200
```

### If visualization is laggy:
```bash
# Run without rendering
python3 demo_heuristic.py --no-render --episodes 10
```

### If you need quick results:
```bash
# Train for minimal timesteps (5 minutes)
python3 train_rl.py --mode train --timesteps 20000
```

## Troubleshooting

**Error: No module named 'gymnasium'**
```bash
pip3 install -r requirements.txt --user
```

**Error: Display issues with matplotlib**
```bash
# Add to top of demo files:
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Error: CUDA out of memory**
```bash
# Training uses CPU by default, this shouldn't happen
# If it does, your system has GPU but not enough memory
# Solution: Ignore, CPU training works fine
```

## Customization Ideas (If Time Permits)

1. **Add 3rd defender type** (30 min)
   - Laser system (instant hit, limited range)
   - Copy SAMBattery class, modify parameters

2. **Add terrain** (1 hour)
   - Mountains block line of sight
   - Modify `_construct_graph()` method

3. **Multi-agent RL** (2 hours)
   - Switch to PettingZoo
   - Each defender learns independently

4. **Real-world data** (1 hour)
   - Load actual airport coordinates
   - Use real drone specifications

## Judging Criteria Alignment

### Technical Complexity ⭐⭐⭐⭐⭐
- Custom RL environment
- Attention mechanism
- Hungarian algorithm
- Graph-based modeling

### Innovation ⭐⭐⭐⭐
- Novel application of attention to resource allocation
- Hybrid approach (learning + optimization)

### Completeness ⭐⭐⭐⭐⭐
- Working demo
- Both baseline and RL
- Visualization
- Evaluation metrics

### Presentation ⭐⭐⭐⭐
- Real-time visualization
- Clear comparison
- Technical depth

## Time Allocation

- Setup & verification: 30 min
- Initial demo preparation: 30 min
- Training (background): 2-3 hours
- Pitch deck: 1 hour
- Practice & polish: 1 hour
- Demo recording: 30 min
- Buffer for issues: 3-4 hours

**Total: 9-10 hours**

---

Good luck! 🚀 You've got this! 💪
