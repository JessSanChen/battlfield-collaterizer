"""
Detailed Collateral Engagement Demo

Shows individual engagement decisions with collateral risk calculations.
Runs a short scenario with verbose output to demonstrate terrain-aware allocation.
"""

import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.integration.collateral_env import CollateralDroneDefenseEnv


def run_detailed_demo(airport_key: str, max_timesteps: int = 30):
    """
    Run a detailed demo showing individual engagement decisions.

    Args:
        airport_key: 'Taoyuan' or 'Songshan'
        max_timesteps: Maximum timesteps to run
    """
    print(f"\n{'='*80}")
    print(f"DETAILED ENGAGEMENT DEMO: {airport_key} Airport")
    print(f"{'='*80}\n")

    # Create environment with verbose=True to see collateral impact
    env = CollateralDroneDefenseEnv(
        airport_key=airport_key,
        use_attention=False,  # Use heuristic for deterministic behavior
        verbose=True  # Enable collateral logging
    )

    # Reset
    obs, info = env.reset()

    print(f"Initial Setup:")
    print(f"  Attackers: {info['attackers_alive']}")
    print(f"  Defenders: {info['defenders_alive']}")
    print(f"  Safe Zone: {env.safe_zone}")
    print()

    # Run timesteps
    for t in range(max_timesteps):
        # Dummy action
        action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Print summary every 5 steps
        if (t + 1) % 5 == 0:
            stats = env.get_collateral_stats()
            print(f"\n[Timestep {t+1}]")
            print(f"  Attackers: alive={info['attackers_alive']}, destroyed={info['attackers_destroyed']}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Cumulative collateral stats:")
            print(f"    - Avg risk: {stats['average_risk']:.3f}")
            print(f"    - Engagements tracked: {stats['engagements_tracked']}")
            print(f"    - Penalties applied: {stats['penalties_applied']}")

        if terminated or truncated:
            print(f"\n[Episode Ended at timestep {t+1}]")
            print(f"  Reason: {'All attackers destroyed' if terminated else 'Max timesteps'}")
            break

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY: {airport_key}")
    print(f"{'='*80}")

    stats = env.get_collateral_stats()
    print(f"\nCollateral Impact:")
    print(f"  Total engagements tracked: {stats['engagements_tracked']}")
    print(f"  Average collateral risk: {stats['average_risk']:.3f}")
    print(f"  Total collateral risk: {stats['total_risk']:.3f}")
    print(f"  Score penalties applied: {stats['penalties_applied']}")

    print(f"\nDefense Outcome:")
    print(f"  Attackers destroyed: {info['attackers_destroyed']}")
    print(f"  Attackers remaining: {info['attackers_alive']}")
    print(f"  Safe zone breached: {info.get('safe_zone_breached', False)}")

    env.close()


def main():
    """Run detailed demo for both airports."""
    print("="*80)
    print("TERRAIN-AWARE COLLATERAL MINIMIZATION - DETAILED ENGAGEMENT DEMO")
    print("="*80)
    print()
    print("This demo shows individual engagement decisions and their collateral impact.")
    print("Watch for high-risk warnings when engagements occur over populated areas.")
    print()

    # Run Taoyuan
    run_detailed_demo('Taoyuan', max_timesteps=30)

    print("\n" + "="*80)
    print()

    # Run Songshan
    run_detailed_demo('Songshan', max_timesteps=30)

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print()
    print("Key Observations:")
    print("  • Taoyuan: Look for ocean intercepts (low risk < 0.10)")
    print("  • Songshan: Look for urban intercepts (higher risk > 0.15)")
    print("  • Same threats handled differently based on geography")
    print()


if __name__ == '__main__':
    main()
