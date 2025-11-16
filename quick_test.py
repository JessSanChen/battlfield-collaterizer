"""
Quick test script to verify everything works.
Runs a single episode with heuristic mode and minimal output.
"""
from drone_defense_env import DroneDefenseEnv
import warnings
warnings.filterwarnings('ignore')


def quick_test():
    """Run a quick test to verify the simulation works."""
    print("Running quick test...")
    print("-" * 50)

    # Create environment
    env = DroneDefenseEnv(
        map_size=(100, 100),
        safe_zone=(40, 40, 60, 60),
        max_attackers=10,
        max_timesteps=100,
        use_attention=False,  # Heuristic mode (faster)
        render_mode=None  # No visualization for quick test
    )

    # Run one episode
    obs, info = env.reset()
    total_reward = 0
    done = False
    truncated = False
    steps = 0

    while not done and not truncated and steps < 100:
        obs, reward, done, truncated, info = env.step(0)
        total_reward += reward
        steps += 1

    print(f"\nTest Results:")
    print(f"  ✓ Environment initialized successfully")
    print(f"  ✓ Episode ran for {steps} timesteps")
    print(f"  ✓ Total reward: {total_reward:.2f}")
    print(f"  ✓ Attackers destroyed: {info['attackers_destroyed']}")
    print(f"  ✓ Safe zone breached: {info['safe_zone_breached']}")
    print(f"  ✓ Final result: {'DEFEAT' if info['safe_zone_breached'] else 'VICTORY'}")
    print("\n" + "-" * 50)
    print("✅ All systems operational!\n")
    print("Next steps:")
    print("  1. Run demo:     python demo_heuristic.py")
    print("  2. Train agent:  python train_rl.py --mode train --timesteps 50000")
    print("  3. View README:  cat README.md")

    env.close()


if __name__ == "__main__":
    quick_test()
