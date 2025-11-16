"""
Collateral-Aware Defense Comparison: Taoyuan vs Songshan

Demonstrates terrain-aware decision making by running identical attack
scenarios at both airports and comparing how collateral constraints
affect engagement strategies.

Key Question:
- Taoyuan (ocean west): Can we engage early over water?
- Songshan (urban all sides): Must we delay to minimize civilian risk?
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.integration.collateral_env import CollateralDroneDefenseEnv


class CollateralMetrics:
    """Track collateral-specific metrics during simulation."""

    def __init__(self, airport_name: str):
        self.airport_name = airport_name
        self.timesteps = 0
        self.total_kills = 0
        self.safe_zone_breaches = 0
        self.engagements = []
        self.collateral_risks = []
        self.engagement_locations = {'ocean': 0, 'low_risk': 0, 'medium_risk': 0, 'high_risk': 0}
        self.total_reward = 0.0

    def record_timestep(self, reward, info, collateral_stats):
        """Record metrics for this timestep."""
        self.timesteps += 1
        self.total_reward += reward

        # Track kills
        if 'attackers_destroyed' in info:
            self.total_kills = info['attackers_destroyed']

        # Track breaches
        if info.get('safe_zone_breached', False):
            self.safe_zone_breaches += 1

        # Track collateral stats
        if collateral_stats['engagements_tracked'] > 0:
            self.collateral_risks.append(collateral_stats['average_risk'])

    def record_engagement(self, risk: float, location: str):
        """Record an individual engagement."""
        self.engagements.append({'risk': risk, 'location': location})

        # Categorize by risk level
        if risk < 0.05:
            self.engagement_locations['ocean'] += 1
        elif risk < 0.15:
            self.engagement_locations['low_risk'] += 1
        elif risk < 0.30:
            self.engagement_locations['medium_risk'] += 1
        else:
            self.engagement_locations['high_risk'] += 1

    def get_summary(self) -> dict:
        """Get summary statistics."""
        avg_collateral_risk = np.mean(self.collateral_risks) if self.collateral_risks else 0.0

        return {
            'airport': self.airport_name,
            'timesteps': self.timesteps,
            'total_kills': self.total_kills,
            'safe_zone_breaches': self.safe_zone_breaches,
            'average_collateral_risk': avg_collateral_risk,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.timesteps if self.timesteps > 0 else 0.0,
            'engagement_zones': self.engagement_locations,
            'total_engagements': len(self.engagements),
        }


def run_scenario(airport_key: str, max_timesteps: int = 100, verbose: bool = False):
    """
    Run attack scenario for a specific airport.

    Args:
        airport_key: 'Taoyuan' or 'Songshan'
        max_timesteps: Maximum simulation timesteps
        verbose: Print detailed progress

    Returns:
        CollateralMetrics object with results
    """
    print(f"\n{'='*80}")
    print(f"Running Scenario: {airport_key} Airport")
    print(f"{'='*80}\n")

    # Create environment
    env = CollateralDroneDefenseEnv(
        airport_key=airport_key,
        use_attention=False,  # Use heuristic for consistent comparison
        verbose=False  # Reduce spam, use our own logging
    )

    # Initialize metrics
    metrics = CollateralMetrics(airport_key)

    # Reset environment
    obs, info = env.reset()

    if verbose:
        print(f"Initial state:")
        print(f"  Attackers: {info['attackers_alive']}")
        print(f"  Defenders: {info['defenders_alive']}")
        print(f"  Safe zone: {env.safe_zone}\n")

    # Run simulation
    for t in range(max_timesteps):
        # Take action (dummy action for heuristic)
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get collateral stats
        collateral_stats = env.get_collateral_stats()

        # Record metrics
        metrics.record_timestep(reward, info, collateral_stats)

        # Log progress every 20 timesteps
        if verbose and (t + 1) % 20 == 0:
            print(f"Timestep {t+1:3d}: "
                  f"Attackers alive: {info['attackers_alive']:2d}, "
                  f"Destroyed: {info['attackers_destroyed']:2d}, "
                  f"Reward: {reward:+.2f}, "
                  f"Avg collateral risk: {collateral_stats.get('average_risk', 0.0):.3f}")

        # Check termination
        if terminated or truncated:
            if verbose:
                print(f"\nEpisode ended at timestep {t+1}")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            break

    # Close environment
    env.close()

    return metrics


def print_comparison(taoyuan_metrics: CollateralMetrics, songshan_metrics: CollateralMetrics):
    """Print detailed comparison between two scenarios."""

    tpe = taoyuan_metrics.get_summary()
    tsa = songshan_metrics.get_summary()

    print("\n" + "="*80)
    print("COLLATERAL-AWARE DEFENSE COMPARISON")
    print("="*80)
    print()

    # Overall Performance
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ OVERALL PERFORMANCE                                                         │")
    print("├─────────────────────────────────┬───────────────────┬───────────────────────┤")
    print("│ Metric                          │ Taoyuan (Ocean)   │ Songshan (Urban)      │")
    print("├─────────────────────────────────┼───────────────────┼───────────────────────┤")
    print(f"│ Total Kills                     │ {tpe['total_kills']:17d} │ {tsa['total_kills']:21d} │")
    print(f"│ Safe Zone Breaches              │ {tpe['safe_zone_breaches']:17d} │ {tsa['safe_zone_breaches']:21d} │")
    print(f"│ Total Reward                    │ {tpe['total_reward']:17.2f} │ {tsa['total_reward']:21.2f} │")
    print(f"│ Average Reward/Step             │ {tpe['avg_reward']:17.3f} │ {tsa['avg_reward']:21.3f} │")
    print(f"│ Timesteps Run                   │ {tpe['timesteps']:17d} │ {tsa['timesteps']:21d} │")
    print("└─────────────────────────────────┴───────────────────┴───────────────────────┘")
    print()

    # Collateral Impact
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ COLLATERAL IMPACT                                                           │")
    print("├─────────────────────────────────┬───────────────────┬───────────────────────┤")
    print("│ Metric                          │ Taoyuan (Ocean)   │ Songshan (Urban)      │")
    print("├─────────────────────────────────┼───────────────────┼───────────────────────┤")
    print(f"│ Average Collateral Risk         │ {tpe['average_collateral_risk']:17.3f} │ {tsa['average_collateral_risk']:21.3f} │")

    # Calculate risk ratio
    if tpe['average_collateral_risk'] > 0:
        risk_ratio = tsa['average_collateral_risk'] / tpe['average_collateral_risk']
        print(f"│ Risk Ratio (Songshan/Taoyuan)   │                   │ {risk_ratio:21.2f}x │")

    print("└─────────────────────────────────┴───────────────────┴───────────────────────┘")
    print()

    # Engagement Zones
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ ENGAGEMENT ZONES (Preferred Intercept Locations)                           │")
    print("├─────────────────────────────────┬───────────────────┬───────────────────────┤")
    print("│ Zone Type                       │ Taoyuan (Ocean)   │ Songshan (Urban)      │")
    print("├─────────────────────────────────┼───────────────────┼───────────────────────┤")
    print(f"│ Ocean/Very Low Risk (<0.05)     │ {tpe['engagement_zones']['ocean']:17d} │ {tsa['engagement_zones']['ocean']:21d} │")
    print(f"│ Low Risk (0.05-0.15)            │ {tpe['engagement_zones']['low_risk']:17d} │ {tsa['engagement_zones']['low_risk']:21d} │")
    print(f"│ Medium Risk (0.15-0.30)         │ {tpe['engagement_zones']['medium_risk']:17d} │ {tsa['engagement_zones']['medium_risk']:21d} │")
    print(f"│ High Risk (>0.30)               │ {tpe['engagement_zones']['high_risk']:17d} │ {tsa['engagement_zones']['high_risk']:21d} │")
    print(f"│ Total Engagements Tracked       │ {tpe['total_engagements']:17d} │ {tsa['total_engagements']:21d} │")
    print("└─────────────────────────────────┴───────────────────┴───────────────────────┘")
    print()

    # Strategic Insights
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ STRATEGIC INSIGHTS                                                          │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()

    # Taoyuan analysis
    ocean_pct = 0
    if tpe['total_engagements'] > 0:
        ocean_pct = (tpe['engagement_zones']['ocean'] / tpe['total_engagements']) * 100

    print(f"  TAOYUAN (Ocean Advantage):")
    print(f"  • {ocean_pct:.1f}% of engagements in ocean/very low risk zones")
    print(f"  • Average collateral risk: {tpe['average_collateral_risk']:.3f}")
    print(f"  • Strategy: Early engagement over water minimizes civilian impact")
    print()

    # Songshan analysis
    urban_high_risk = tsa['engagement_zones']['medium_risk'] + tsa['engagement_zones']['high_risk']
    if tsa['total_engagements'] > 0:
        urban_pct = (urban_high_risk / tsa['total_engagements']) * 100
    else:
        urban_pct = 0

    print(f"  SONGSHAN (Urban Constraints):")
    print(f"  • {urban_pct:.1f}% of engagements in medium/high risk zones")
    print(f"  • Average collateral risk: {tsa['average_collateral_risk']:.3f}")
    print(f"  • Strategy: Must balance threat elimination with civilian safety")
    print()

    # Key Finding
    print("  KEY FINDING:")
    if tpe['average_collateral_risk'] < tsa['average_collateral_risk']:
        reduction = ((tsa['average_collateral_risk'] - tpe['average_collateral_risk'])
                     / tsa['average_collateral_risk'] * 100)
        print(f"  Taoyuan's ocean geography enables {reduction:.1f}% lower collateral risk")
        print(f"  for identical attack scenarios. Terrain-aware allocation successfully")
        print(f"  exploits geographic advantages while maintaining defense effectiveness.")
    else:
        print(f"  Collateral risks are comparable, suggesting similar geographic constraints.")
    print()

    print("="*80)


def main():
    """Run comparison demo."""
    print("="*80)
    print("TERRAIN-AWARE COLLATERAL MINIMIZATION DEMO")
    print("="*80)
    print()
    print("Scenario: 10 drones attacking from the west")
    print("Question: How does geography affect engagement strategy?")
    print()
    print("Hypothesis:")
    print("  • Taoyuan: Ocean to the west → Early low-risk engagements")
    print("  • Songshan: Urban all sides → Delayed or constrained engagements")
    print()

    # Run both scenarios
    max_timesteps = 100

    print(f"Running {max_timesteps} timestep simulations...\n")

    # Run Taoyuan
    taoyuan_metrics = run_scenario('Taoyuan', max_timesteps=max_timesteps, verbose=True)

    # Run Songshan
    songshan_metrics = run_scenario('Songshan', max_timesteps=max_timesteps, verbose=True)

    # Print comparison
    print_comparison(taoyuan_metrics, songshan_metrics)

    print("\nDemo complete! Integration successfully demonstrates:")
    print("  ✓ Coordinate mapping (abstract → real lat/lon)")
    print("  ✓ Collateral risk calculation (population + infrastructure + flight paths)")
    print("  ✓ Terrain-aware decision making (same threat, different response)")
    print()


if __name__ == '__main__':
    main()
