"""
Demo script for heuristic-based drone defense.
Runs the simulation using the simple time-to-target heuristic.
"""
import numpy as np
from drone_defense_env import DroneDefenseEnv


def run_heuristic_demo(num_episodes: int = 3, render: bool = True):
    """
    Run drone defense simulation with heuristic allocation.

    Args:
        num_episodes: Number of episodes to run
        render: Whether to visualize
    """
    # Create environment with heuristic mode
    env = DroneDefenseEnv(
        # map_size=(100, 100),
        # safe_zone=(40, 40, 60, 60),
        # max_attackers=15,
        # max_timesteps=200,
        use_attention=False,  # Use heuristic
        render_mode='human' if render else None
    )

    print("=" * 60)
    print("DRONE DEFENSE SIMULATION - HEURISTIC MODE")
    print("=" * 60)
    print("\nHeuristic Strategy:")
    print("  Priority = time_until_safe_zone * warhead_mass")
    print("  Targets closer to safe zone with heavier warheads are prioritized")
    print("\nDefenders:")
    print("  - 2x SAM Batteries (long range, high kill probability)")
    print("  - 2x Kinetic Drone Depots (medium range, fast reload)")
    print("\n" + "=" * 60 + "\n")

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}\n")

        obs, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            # Dummy action (allocation is done internally)
            obs, reward, done, truncated, info = env.step(0)
            total_reward += reward

            if render:
                env.render()

            # Print status every 20 timesteps
            if info['timestep'] % 20 == 0:
                print(f"  Timestep {info['timestep']:3d} | "
                      f"Attackers: {info['attackers_alive']:2d} alive, "
                      f"{info['attackers_destroyed']:2d} destroyed | "
                      f"Defenders: {info['defenders_alive']} alive | "
                      f"Reward: {total_reward:7.2f}")

        # Episode summary
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"  Duration: {info['timestep']} timesteps")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Attackers Destroyed: {info['attackers_destroyed']}")
        print(f"  Attackers Alive: {info['attackers_alive']}")
        print(f"  Defenders Alive: {info['defenders_alive']}")
        print(f"  Safe Zone Breached: {'YES' if info['safe_zone_breached'] else 'NO'}")
        print(f"  Result: {'DEFEAT' if info['safe_zone_breached'] else 'VICTORY'}")
        print(f"{'='*60}\n")

        if render:
            input("Press Enter to continue to next episode...")

    env.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run heuristic drone defense demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    run_heuristic_demo(num_episodes=args.episodes, render=not args.no_render)
