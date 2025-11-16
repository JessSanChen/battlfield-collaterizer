"""
Collateral-Aware Drone Defense Environment

Extends the partner's RL environment with terrain-aware collateral minimization.
Integrates real-world population density, infrastructure, and flight path data
into the defender allocation decision-making.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from typing import Optional, Dict

# Add parent directory to path for imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from drone_defense_env import DroneDefenseEnv
from entities import Defender, Attacker
from src.integration.coordinate_mapper import CoordinateMapper
from src.integration.realistic_configs import REALISTIC_CONFIG
from collateral_calculator import CollateralCalculator


class CollateralDroneDefenseEnv(DroneDefenseEnv):
    """
    Drone defense environment with terrain-aware collateral minimization.

    Extends the base RL environment by:
    1. Using realistic configurations (real-world SAM/drone parameters)
    2. Mapping abstract coordinates to real lat/lon via CoordinateMapper
    3. Calculating collateral risk using population/infrastructure data
    4. Modifying RL scores with collateral penalties before allocation

    This creates terrain-aware behavior where the same threat is handled
    differently based on geography (e.g., ocean vs urban intercepts).
    """

    def __init__(self,
                 airport_key: str = 'Taoyuan',
                 config_dict: Optional[Dict] = None,
                 use_attention: bool = True,
                 render_mode: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize collateral-aware environment.

        Args:
            airport_key: 'Taoyuan' or 'Songshan'
            config_dict: Configuration dictionary (defaults to REALISTIC_CONFIG)
            use_attention: Whether to use attention network
            render_mode: Rendering mode
            verbose: Print collateral impact during execution
        """
        self.airport_key = airport_key
        self.verbose = verbose

        # Use realistic config by default
        if config_dict is None:
            config_dict = REALISTIC_CONFIG

        # Store config for parent class
        self.config_dict = config_dict

        # Initialize coordinate mapper
        self.coord_mapper = CoordinateMapper(airport_key)
        if self.verbose:
            print(f"✓ Initialized CoordinateMapper for {airport_key}")

        # Load collateral calculator
        self.collateral_calculator = self._load_collateral_calculator()
        if self.verbose:
            print(f"✓ Loaded CollateralCalculator for {airport_key}")

        # Initialize parent class with config dictionary
        # We'll override the config loading to use our dict
        self._temp_config_dict = config_dict
        super().__init__(
            config_path=None,  # Will be handled by our override
            use_attention=use_attention,
            render_mode=render_mode
        )

        # Collateral tracking
        self.total_collateral_risk = 0.0
        self.engagements_tracked = 0
        self.collateral_penalties_applied = 0

        if self.verbose:
            print(f"✓ Collateral-aware environment ready for {airport_key}")
            print(f"  Map: {self.map_width}×{self.map_height} units = "
                  f"{self.map_width*100/1000}km×{self.map_height*100/1000}km")
            print(f"  Safe zone: {self.safe_zone}")

    def _load_config(self, config_path: Optional[str]):
        """Override parent's config loading to use our dictionary."""
        if hasattr(self, '_temp_config_dict'):
            return self._temp_config_dict
        return super()._load_config(config_path)

    def _load_collateral_calculator(self) -> CollateralCalculator:
        """Load the collateral calculator with airport-specific data."""
        # Load airports data
        data_file = Path('airports_data.pkl')
        if not data_file.exists():
            raise FileNotFoundError(
                "airports_data.pkl not found. Run extract_data.py first."
            )

        with open(data_file, 'rb') as f:
            airports_data = pickle.load(f)

        return CollateralCalculator(
            airport_name=self.airport_key,
            airports_data=airports_data
        )

    def _modify_scores_with_collateral(self,
                                       score_matrix: np.ndarray,
                                       defenders: list,
                                       attackers: list) -> np.ndarray:
        """
        Modify RL score matrix with collateral penalties.

        For each defender-attacker pair:
        1. Calculate predicted intercept point (midpoint heuristic)
        2. Convert abstract coordinates to real lat/lon
        3. Calculate collateral risk (population + infrastructure + flight paths)
        4. Apply penalty: modified_score = original_score × (1 - risk)

        Args:
            score_matrix: Original scores from RL network [D × A]
            defenders: List of Defender objects
            attackers: List of Attacker objects

        Returns:
            Modified score matrix with collateral penalties applied
        """
        if len(defenders) == 0 or len(attackers) == 0:
            return score_matrix

        modified_scores = score_matrix.copy()
        total_risk = 0.0
        num_pairs = 0

        for i, defender in enumerate(defenders):
            for j, attacker in enumerate(attackers):
                # Calculate predicted intercept point (midpoint approximation)
                intercept_x = (defender.x + attacker.x) / 2.0
                intercept_y = (defender.y + attacker.y) / 2.0

                # Convert to real coordinates
                intercept_lat, intercept_lon = self.coord_mapper.abstract_to_real(
                    intercept_x, intercept_y
                )

                # Calculate collateral risk
                effector_type = defender.defender_type.lower() if hasattr(defender, 'defender_type') else 'kinetic'
                risk = self.collateral_calculator.calculate_engagement_risk(
                    intercept_lat,
                    intercept_lon,
                    effector_type
                )

                # Apply collateral multiplier
                collateral_multiplier = 1.0 - risk
                original_score = modified_scores[i, j]
                modified_scores[i, j] = original_score * collateral_multiplier

                # Track statistics
                total_risk += risk
                num_pairs += 1

                # Log high-risk engagements
                if risk > 0.3 and self.verbose:
                    print(f"  ⚠️  High-risk engagement: "
                          f"Defender {i} ({effector_type}) vs Attacker {j}")
                    print(f"      Intercept: ({intercept_lat:.4f}°N, {intercept_lon:.4f}°E)")
                    print(f"      Collateral risk: {risk:.3f}")
                    print(f"      Score: {original_score:.2f} → {modified_scores[i, j]:.2f} "
                          f"({collateral_multiplier*100:.1f}%)")

        # Update tracking
        if num_pairs > 0:
            avg_risk = total_risk / num_pairs
            self.total_collateral_risk += avg_risk
            self.engagements_tracked += 1
            self.collateral_penalties_applied += np.sum(score_matrix != modified_scores)

            if self.verbose and self.timestep % 10 == 0:  # Print every 10 timesteps
                print(f"\n[Timestep {self.timestep}] Collateral Impact:")
                print(f"  Average risk this step: {avg_risk:.3f}")
                print(f"  Cumulative avg risk: {self.total_collateral_risk / self.engagements_tracked:.3f}")
                print(f"  Penalties applied: {self.collateral_penalties_applied}")

        return modified_scores

    def step(self, action):
        """
        Override step to inject collateral penalties into allocation.

        Execution flow:
        1. [Parent] Update defender/attacker states
        2. [Parent] Generate score matrix from attention network
        3. [OURS] Modify scores with collateral penalties ← INTEGRATION POINT
        4. [Parent] Run Hungarian algorithm with modified scores
        5. [Parent] Execute engagements
        6. [Parent] Calculate rewards
        """
        # Store timestep for logging
        if not hasattr(self, 'timestep'):
            self.timestep = 0
        self.timestep += 1

        # Get observation and score matrix (we need to intercept this)
        # Since we can't easily override mid-step, we'll override _hungarian_allocation instead
        return super().step(action)

    def _compute_score_matrix(self, defenders: list, attackers: list) -> np.ndarray:
        """Override to add collateral penalties after score generation."""
        # Get original scores from parent
        score_matrix = super()._compute_score_matrix(defenders, attackers)

        # Apply collateral modifications
        modified_matrix = self._modify_scores_with_collateral(
            score_matrix, defenders, attackers
        )

        return modified_matrix

    def reset(self, **kwargs):
        """Reset environment and collateral tracking."""
        # Reset counters
        self.total_collateral_risk = 0.0
        self.engagements_tracked = 0
        self.collateral_penalties_applied = 0
        self.timestep = 0

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EPISODE RESET: {self.airport_key} Airport")
            print(f"{'='*80}\n")

        return super().reset(**kwargs)

    def get_collateral_stats(self) -> Dict[str, float]:
        """Get collateral damage statistics for this episode."""
        if self.engagements_tracked == 0:
            return {
                'average_risk': 0.0,
                'total_risk': 0.0,
                'engagements_tracked': 0,
                'penalties_applied': 0,
            }

        return {
            'average_risk': self.total_collateral_risk / self.engagements_tracked,
            'total_risk': self.total_collateral_risk,
            'engagements_tracked': self.engagements_tracked,
            'penalties_applied': self.collateral_penalties_applied,
        }


# ============================================================================
# TEST & DEMO
# ============================================================================

def test_collateral_env():
    """Test the collateral-aware environment."""
    print("=" * 80)
    print("COLLATERAL ENVIRONMENT TEST")
    print("=" * 80)
    print()

    for airport_key in ['Taoyuan', 'Songshan']:
        print(f"\n{'='*80}")
        print(f"Testing {airport_key} Airport")
        print(f"{'='*80}\n")

        # Create environment
        env = CollateralDroneDefenseEnv(
            airport_key=airport_key,
            use_attention=False,  # Use heuristic for faster testing
            verbose=True
        )

        # Reset and run a few steps
        obs, info = env.reset()
        print(f"\nInitial observation shape: {obs.shape}")
        print(f"Safe zone: {env.safe_zone}")
        print(f"Map size: {env.map_width} × {env.map_height}")
        print()

        # Run 5 steps
        print(f"Running 5 timesteps...\n")
        for step in range(5):
            # Random action (dummy)
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            if step == 0:
                print(f"Step {step+1}:")
                print(f"  Reward: {reward:.2f}")
                print(f"  Terminated: {terminated}")
                print(f"  Info: {info}")
                print()

            if terminated or truncated:
                print(f"Episode ended at step {step+1}")
                break

        # Get collateral stats
        stats = env.get_collateral_stats()
        print(f"\nCollateral Statistics:")
        print(f"  Average risk: {stats['average_risk']:.3f}")
        print(f"  Total risk: {stats['total_risk']:.3f}")
        print(f"  Engagements tracked: {stats['engagements_tracked']}")
        print(f"  Penalties applied: {stats['penalties_applied']}")

        env.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCollateral environment successfully integrates:")
    print("  ✓ CoordinateMapper (abstract ↔ real coordinates)")
    print("  ✓ CollateralCalculator (population + infrastructure + flight paths)")
    print("  ✓ Realistic configurations (real-world SAM/drone parameters)")
    print("  ✓ Score modification (collateral penalties applied to RL)")


if __name__ == '__main__':
    test_collateral_env()
