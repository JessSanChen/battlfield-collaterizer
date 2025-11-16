"""
Gymnasium environment for drone defense simulation.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import yaml
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from entities import Defender, Attacker, SAMBattery, KineticDroneDepot, AESAEmitter, Projectile, reset_id_counters
from attention_network import AttentionAllocationSystem


class DroneDefenseEnv(gym.Env):
    """
    Drone defense environment for RL training.

    The agent controls defenders to protect a safe zone from attackers.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config_path: str = "config.yaml",
                 use_attention: bool = True,
                 render_mode: Optional[str] = None):
        """
        Args:
            config_path: Path to YAML configuration file
            use_attention: Whether to use attention network or heuristic
            render_mode: Rendering mode
        """
        super().__init__()

        # Load configuration from YAML
        self.config = self._load_config(config_path)

        # Map configuration
        self.map_width = self.config['map']['width']
        self.map_height = self.config['map']['height']

        self.safe_zone = (
            self.config['safe_zone']['x_min'],
            self.config['safe_zone']['y_min'],
            self.config['safe_zone']['x_max'],
            self.config['safe_zone']['y_max']
        )

        # Episode configuration
        self.max_timesteps = self.config['episode']['max_timesteps']
        self.max_total_attackers = self.config['episode']['max_total_attackers']
        self.max_concurrent_attackers = self.config['episode']['max_concurrent_attackers']

        # Spawn configuration
        self.min_attackers = self.config['attacker_spawn']['initial_count']
        self.spawn_prob_initial = self.config['attacker_spawn']['spawn_probability_initial']
        self.spawn_prob_decay = self.config['attacker_spawn']['spawn_probability_decay']
        self.spawn_edges = self.config['attacker_spawn']['spawn_edges']
        self.attacker_speed_range = (
            self.config['attacker_spawn']['speed_min'],
            self.config['attacker_spawn']['speed_max']
        )
        self.warhead_mass_range = (
            self.config['attacker_spawn']['warhead_mass_min'],
            self.config['attacker_spawn']['warhead_mass_max']
        )

        # Defender stats
        self.defender_stats = self.config['defender_stats']

        # Visualization configuration
        self.viz_config = self.config['visualization']
        self.engagement_display_time = self.viz_config['engagement_display_time']

        self.use_attention = use_attention
        self.render_mode = render_mode

        # Initialize entities
        self.defenders: List[Defender] = []
        self.attackers: List[Attacker] = []
        self.projectiles: List[Projectile] = []

        # Episode state
        self.timestep = 0
        self.total_attackers_spawned = 0
        self.attackers_destroyed = 0
        self.safe_zone_breached = False

        # Engagement tracking for visualization
        self.active_engagements = []  # List of (defender_id, attacker_id, time_remaining)
        self.active_aesa_cones = []  # List of (defender_id, orientation, time_remaining) for AESA visualization

        # Neural network for attention-based allocation
        if self.use_attention:
            self.attention_system = AttentionAllocationSystem(
                defender_dim=7,  # Updated to 7 (added type_aesa)
                attacker_dim=6,
                context_dim=16,
                hidden_dim=64
            )
        else:
            self.attention_system = None

        # Observation and action spaces
        # For simplicity, we'll use a dummy action space
        # The actual actions are determined by the allocation algorithm
        self.action_space = spaces.Discrete(1)  # Dummy action space

        # Observation space: flattened state (for compatibility with SB3)
        # In practice, the network directly processes defender/attacker features
        max_state_size = (10 * 7 + 10 * 6)  # Max defenders (7 features) + attackers (6 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_state_size,), dtype=np.float32
        )

        # Visualization
        self.fig = None
        self.ax = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _create_defenders(self):
        """Create initial defender configuration from config file."""
        self.defenders = []

        for defender_config in self.config['defenders']:
            defender_type = defender_config['type']
            x = defender_config['x']
            y = defender_config['y']

            # Get stats for this defender type
            stats = self.defender_stats[defender_type]

            # Create defender instance
            if defender_type == 'SAM':
                defender = SAMBattery(x=x, y=y, stats=stats)
            elif defender_type == 'KINETIC':
                defender = KineticDroneDepot(x=x, y=y, stats=stats)
            elif defender_type == 'AESA':
                defender = AESAEmitter(x=x, y=y, stats=stats)
            else:
                raise ValueError(f"Unknown defender type: {defender_type}")

            self.defenders.append(defender)

    def _spawn_attacker(self):
        """Spawn a new attacker at a random edge of the map using config parameters."""
        # Spawn from configured edges
        edge = np.random.choice(self.spawn_edges)

        safe_zone_center_x = (self.safe_zone[0] + self.safe_zone[2]) / 2
        safe_zone_center_y = (self.safe_zone[1] + self.safe_zone[3]) / 2

        if edge == 'top':
            x = np.random.uniform(0, self.map_width)
            y = 0
        elif edge == 'bottom':
            x = np.random.uniform(0, self.map_width)
            y = self.map_height
        elif edge == 'left':
            x = 0
            y = np.random.uniform(0, self.map_height)
        else:  # right
            x = self.map_width
            y = np.random.uniform(0, self.map_height)

        # Velocity towards safe zone center
        dx = safe_zone_center_x - x
        dy = safe_zone_center_y - y
        distance = np.sqrt(dx**2 + dy**2)

        # Use configured speed range
        speed = np.random.uniform(self.attacker_speed_range[0], self.attacker_speed_range[1])
        vx = (dx / distance) * speed
        vy = (dy / distance) * speed

        # Use configured warhead mass range
        warhead_mass = np.random.uniform(self.warhead_mass_range[0], self.warhead_mass_range[1])

        attacker = Attacker(x=x, y=y, vx=vx, vy=vy, warhead_mass=warhead_mass)
        self.attackers.append(attacker)
        self.total_attackers_spawned += 1

    def _construct_graph(self) -> np.ndarray:
        """
        Construct adjacency matrix (graph) between defenders and attackers.

        Returns:
            adjacency_matrix: [num_defenders, num_attackers]
                             1 if attacker in range, 0 otherwise
        """
        num_defenders = len([d for d in self.defenders if d.alive])
        num_attackers = len([a for a in self.attackers if a.alive])

        if num_defenders == 0 or num_attackers == 0:
            return np.zeros((num_defenders, num_attackers))

        adjacency = np.zeros((num_defenders, num_attackers))

        alive_defenders = [d for d in self.defenders if d.alive]
        alive_attackers = [a for a in self.attackers if a.alive]

        for i, defender in enumerate(alive_defenders):
            for j, attacker in enumerate(alive_attackers):
                distance = np.linalg.norm(
                    defender.get_position() - attacker.get_position()
                )
                if distance <= defender.max_range:
                    adjacency[i, j] = 1

        return adjacency

    def _compute_score_matrix_attention(self) -> np.ndarray:
        """
        Compute score matrix using attention network.

        Returns:
            score_matrix: [num_defenders, num_attackers]
        """
        alive_defenders = [d for d in self.defenders if d.alive]
        alive_attackers = [a for a in self.attackers if a.alive]

        if len(alive_defenders) == 0 or len(alive_attackers) == 0:
            return np.zeros((len(alive_defenders), len(alive_attackers)))

        # Collect features
        defender_features = np.stack([d.to_feature_vector() for d in alive_defenders])
        attacker_features = np.stack([a.to_feature_vector() for a in alive_attackers])

        # Add batch dimension
        defender_features = torch.tensor(defender_features, dtype=torch.float32).unsqueeze(0)
        attacker_features = torch.tensor(attacker_features, dtype=torch.float32).unsqueeze(0)

        # Forward pass through attention system
        with torch.no_grad():
            scores = self.attention_system(defender_features, attacker_features)

        # Remove batch dimension and convert to numpy
        scores = scores.squeeze(0).numpy()

        return scores

    def _compute_score_matrix_heuristic(self) -> np.ndarray:
        """
        Compute score matrix using simple heuristic:
        score = time_until_safe_zone * warhead_mass

        Higher score = higher priority target

        Returns:
            score_matrix: [num_defenders, num_attackers]
        """
        alive_defenders = [d for d in self.defenders if d.alive]
        alive_attackers = [a for a in self.attackers if a.alive]

        if len(alive_defenders) == 0 or len(alive_attackers) == 0:
            return np.zeros((len(alive_defenders), len(alive_attackers)))

        score_matrix = np.zeros((len(alive_defenders), len(alive_attackers)))

        safe_zone_center = np.array([
            (self.safe_zone[0] + self.safe_zone[2]) / 2,
            (self.safe_zone[1] + self.safe_zone[3]) / 2
        ])

        for j, attacker in enumerate(alive_attackers):
            # Distance to safe zone
            distance_to_safe_zone = np.linalg.norm(
                attacker.get_position() - safe_zone_center
            )

            # Time until reaching safe zone
            speed = attacker.get_speed()
            if speed > 0:
                time_until_safe_zone = distance_to_safe_zone / speed
            else:
                time_until_safe_zone = float('inf')

            # Priority score
            priority = time_until_safe_zone * attacker.warhead_mass

            # All defenders have same priority for this attacker
            score_matrix[:, j] = priority

        return score_matrix

    def _hungarian_allocation(self, score_matrix: np.ndarray,
                             adjacency_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Use Hungarian algorithm to find optimal allocation.

        Args:
            score_matrix: [num_defenders, num_attackers]
            adjacency_matrix: [num_defenders, num_attackers]

        Returns:
            List of (defender_idx, attacker_idx) pairs
        """
        if score_matrix.shape[0] == 0 or score_matrix.shape[1] == 0:
            return []

        # Mask out out-of-range pairs
        valid_scores = score_matrix.copy()
        valid_scores[adjacency_matrix == 0] = -1e9  # Very low score for out of range

        # For heuristic: we want to MAXIMIZE (higher score = higher priority)
        # Hungarian algorithm minimizes, so negate scores
        cost_matrix = -valid_scores

        # Run Hungarian algorithm
        defender_indices, attacker_indices = linear_sum_assignment(cost_matrix)

        # Filter out invalid assignments (out of range or can't shoot)
        allocations = []
        alive_defenders = [d for d in self.defenders if d.alive]

        for d_idx, a_idx in zip(defender_indices, attacker_indices):
            if adjacency_matrix[d_idx, a_idx] == 1:
                # Check if defender can shoot
                if alive_defenders[d_idx].can_shoot():
                    allocations.append((d_idx, a_idx))

        return allocations

    def _check_safe_zone_breach(self) -> bool:
        """Check if any attacker has reached the safe zone."""
        for attacker in self.attackers:
            if not attacker.alive:
                continue

            x, y = attacker.get_position()
            if (self.safe_zone[0] <= x <= self.safe_zone[2] and
                self.safe_zone[1] <= y <= self.safe_zone[3]):
                return True

        return False

    def _calculate_reward(self, allocations: List[Tuple[int, int]],
                         attackers_killed: int, defenders_lost: int,
                         safe_zone_breached: bool) -> float:
        """
        Calculate reward for this timestep.

        Reward components:
        - +10 for each attacker destroyed
        - +0.1 * ammo_conservation_ratio
        - -50 for each defender lost
        - -100 for each AESA emitter aiming below horizontal (toward runway)
        - -200 for safe zone breach (most severe penalty)

        Args:
            allocations: Defender-attacker allocations this timestep
            attackers_killed: Number of attackers killed this timestep
            defenders_lost: Number of defenders lost this timestep
            safe_zone_breached: Whether safe zone was breached

        Returns:
            reward: Total reward for this timestep
        """
        reward = 0.0

        # Reward for destroying attackers
        reward += attackers_killed * 10.0

        # Reward for ammo conservation
        alive_defenders = [d for d in self.defenders if d.alive]
        if len(alive_defenders) > 0:
            total_ammo = sum(d.ammo for d in alive_defenders)
            max_total_ammo = sum(d.max_ammo for d in alive_defenders)
            ammo_ratio = total_ammo / max_total_ammo if max_total_ammo > 0 else 0
            reward += ammo_ratio * 0.1

        # Penalty for losing defenders
        reward -= defenders_lost * 50.0

        # Large penalty for safe zone breach
        if safe_zone_breached:
            reward -= 200.0

        # Penalty for AESA emitters aiming south (below horizontal)
        # This discourages aiming toward the runway
        for defender in alive_defenders:
            if isinstance(defender, AESAEmitter):
                if defender.is_aiming_below_horizontal():
                    reward -= 100.0  # Heavy penalty, but less than safe zone breach

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset ID counters for new episode
        reset_id_counters()

        # Reset state
        self.timestep = 0
        self.total_attackers_spawned = 0
        self.attackers_destroyed = 0
        self.safe_zone_breached = False

        # Clear projectiles and engagements
        self.projectiles = []
        self.active_engagements = []
        self.active_aesa_cones = []

        # Create defenders
        self._create_defenders()

        # Create initial attackers
        self.attackers = []
        for _ in range(self.min_attackers):
            self._spawn_attacker()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one timestep.

        Args:
            action: Dummy action (allocation is done internally)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.timestep += 1

        # 1. Spawn new attackers (probability decreases over time)
        # Check both concurrent limit and total episode limit
        spawn_prob = self.spawn_prob_initial * (self.spawn_prob_decay ** self.timestep)
        current_alive = len([a for a in self.attackers if a.alive])

        if (np.random.random() < spawn_prob and
            current_alive < self.max_concurrent_attackers and
            self.total_attackers_spawned < self.max_total_attackers):
            self._spawn_attacker()

        # 2. Construct graph (adjacency matrix)
        adjacency_matrix = self._construct_graph()

        # 3. Compute score matrix
        if self.use_attention and self.attention_system is not None:
            score_matrix = self._compute_score_matrix_attention()
        else:
            score_matrix = self._compute_score_matrix_heuristic()

        # 4. Hungarian allocation
        allocations = self._hungarian_allocation(score_matrix, adjacency_matrix)

        # 5. Launch projectiles for allocations
        alive_defenders = [d for d in self.defenders if d.alive]
        alive_attackers = [a for a in self.attackers if a.alive]

        for d_idx, a_idx in allocations:
            defender = alive_defenders[d_idx]
            attacker = alive_attackers[a_idx]

            # Calculate probability of kill
            distance = np.linalg.norm(
                defender.get_position() - attacker.get_position()
            )
            pk = defender.probability_of_kill(distance, attacker.get_speed())

            # Create projectile with predictive targeting (pass target velocity)
            projectile = Projectile(
                defender_id=defender.id,
                attacker_id=attacker.id,
                start_x=defender.x,
                start_y=defender.y,
                target_x=attacker.x,
                target_y=attacker.y,
                target_vx=attacker.vx,
                target_vy=attacker.vy,
                speed=defender.shot_speed,
                probability_of_kill=pk
            )
            self.projectiles.append(projectile)

            # Add to active engagements for visualization
            self.active_engagements.append((defender.id, attacker.id, self.engagement_display_time))

            # Consume ammo
            defender.shoot()

        # 5.5. AESA Emitter firing (destroys everything in cone - including friendly projectiles)
        aesa_destroyed_attackers = 0
        aesa_destroyed_projectiles = 0

        for defender in alive_defenders:
            if not isinstance(defender, AESAEmitter):
                continue

            if not defender.can_shoot():
                continue

            # Find nearest attacker WITHIN RANGE to aim at
            nearest_attacker = None
            min_distance = float('inf')

            for attacker in alive_attackers:
                distance = np.linalg.norm(defender.get_position() - attacker.get_position())
                if distance <= defender.max_range and distance < min_distance:
                    min_distance = distance
                    nearest_attacker = attacker

            if nearest_attacker is None:
                continue  # No targets in range - don't waste a shot!

            # Aim at nearest attacker
            dx = nearest_attacker.x - defender.x
            dy = nearest_attacker.y - defender.y
            aim_angle = np.degrees(np.arctan2(dy, dx))
            defender.set_orientation(aim_angle)

            # Fire! Destroy all entities in cone
            # Check all attackers
            for attacker in alive_attackers:
                if defender.is_in_cone(attacker.x, attacker.y):
                    attacker.alive = False
                    aesa_destroyed_attackers += 1
                    self.attackers_destroyed += 1

            # Check all projectiles (including friendly ones!)
            for projectile in self.projectiles:
                if defender.is_in_cone(projectile.x, projectile.y):
                    projectile.active = False
                    aesa_destroyed_projectiles += 1

            # Track cone activation for visualization (longer persistence for visibility)
            self.active_aesa_cones.append((defender.id, defender.orientation, 30))  # 30 timesteps = 30 seconds

            # Debug output
            print(f"  🔥 AESA D{defender.id} FIRED! Orientation: {defender.orientation:.1f}° | "
                  f"Destroyed: {aesa_destroyed_attackers} attackers, {aesa_destroyed_projectiles} projectiles")

            # Consume ammo and trigger reload
            defender.shoot()

        # 6. Update projectiles and check for hits
        attackers_killed_this_step = aesa_destroyed_attackers  # Start with AESA kills
        projectiles_to_remove = []

        for projectile in self.projectiles:
            # Find target attacker
            target_attacker = None
            for attacker in self.attackers:
                if attacker.id == projectile.attacker_id and attacker.alive:
                    target_attacker = attacker
                    break

            if target_attacker is None:
                # Target is dead or doesn't exist, remove projectile
                projectiles_to_remove.append(projectile)
                continue

            # Update projectile
            hit_result = projectile.update(target_attacker.x, target_attacker.y)

            if hit_result is not None:
                # Projectile has resolved
                if hit_result:  # Hit
                    target_attacker.alive = False
                    attackers_killed_this_step += 1
                    self.attackers_destroyed += 1
                # If miss, attacker survives
                projectiles_to_remove.append(projectile)

        # Remove resolved projectiles
        for projectile in projectiles_to_remove:
            self.projectiles.remove(projectile)

        # 7. Update engagement display timers
        self.active_engagements = [
            (d_id, a_id, time - 1) for d_id, a_id, time in self.active_engagements if time > 1
        ]
        self.active_aesa_cones = [
            (d_id, orientation, time - 1) for d_id, orientation, time in self.active_aesa_cones if time > 1
        ]

        # 8. Update all entities
        for defender in self.defenders:
            defender.update()

        for attacker in self.attackers:
            attacker.update()

        # 9. Check safe zone breach
        safe_zone_breached = self._check_safe_zone_breach()
        if safe_zone_breached:
            self.safe_zone_breached = True

        # 10. Calculate reward
        defenders_lost = 0  # Not implemented yet
        reward = self._calculate_reward(
            allocations, attackers_killed_this_step, defenders_lost, safe_zone_breached
        )

        # 11. Check termination conditions
        alive_attackers_count = len([a for a in self.attackers if a.alive])
        terminated = safe_zone_breached or (alive_attackers_count == 0 and self.timestep > 20)
        truncated = self.timestep >= self.max_timesteps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation (flattened state)."""
        # Simplified observation for compatibility
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Pack defender features (7 features: x, y, ammo_ratio, reload_status, type_sam, type_kinetic, type_aesa)
        idx = 0
        for i, defender in enumerate(self.defenders[:10]):  # Max 10 defenders
            if defender.alive:
                features = defender.to_feature_vector()
                obs[idx:idx+7] = features
            idx += 7

        # Pack attacker features (6 features: x, y, vx, vy, warhead_mass, speed)
        for i, attacker in enumerate([a for a in self.attackers if a.alive][:10]):  # Max 10 attackers
            features = attacker.to_feature_vector()
            obs[idx:idx+6] = features
            idx += 6

        return obs

    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            'timestep': self.timestep,
            'attackers_alive': len([a for a in self.attackers if a.alive]),
            'attackers_destroyed': self.attackers_destroyed,
            'defenders_alive': len([d for d in self.defenders if d.alive]),
            'safe_zone_breached': self.safe_zone_breached
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        if self.fig is None:
            fig_size = self.viz_config['figure_size']
            self.fig, self.ax = plt.subplots(figsize=(fig_size, fig_size))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Drone Defense - Timestep {self.timestep}s | 20km x 20km Battlefield',
                         fontsize=14, fontweight='bold')

        # Get colors from config
        colors = self.viz_config['colors']

        # Draw safe zone (runway)
        safe_zone_rect = patches.Rectangle(
            (self.safe_zone[0], self.safe_zone[1]),
            self.safe_zone[2] - self.safe_zone[0],
            self.safe_zone[3] - self.safe_zone[1],
            linewidth=2, edgecolor=colors['safe_zone'], facecolor='light' + colors['safe_zone'], alpha=0.3
        )
        self.ax.add_patch(safe_zone_rect)

        # Add runway label
        runway_center_x = (self.safe_zone[0] + self.safe_zone[2]) / 2
        runway_center_y = (self.safe_zone[1] + self.safe_zone[3]) / 2
        self.ax.text(runway_center_x, runway_center_y, 'RUNWAY',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=colors['safe_zone'], alpha=0.8)

        # Draw defenders - ALL TYPES ARE SQUARES
        for defender in self.defenders:
            if not defender.alive:
                continue

            # Get color from config based on type
            if defender.defender_type == 'SAM':
                color = colors['sam']
            elif defender.defender_type == 'KINETIC':
                color = colors['kinetic']
            elif defender.defender_type == 'AESA':
                color = colors['aesa']
            else:
                color = 'gray'  # Fallback
            marker = 's'  # All are squares

            # Draw defender
            marker_size = self.viz_config['marker_size_defender']
            self.ax.plot(defender.x, defender.y, marker=marker, color=color,
                        markersize=marker_size, markeredgecolor='black', markeredgewidth=1.5)

            # Draw range circle
            range_circle = plt.Circle((defender.x, defender.y), defender.max_range,
                                     color=color, fill=False, linestyle='--', alpha=0.3, linewidth=1)
            self.ax.add_patch(range_circle)

            # Show ID at top
            self.ax.text(defender.x, defender.y + 5, f'D{defender.id}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Show ammo and reload timer below
            if defender.reload_counter > 0:
                status_text = f'Ammo:{defender.ammo} (R:{defender.reload_counter})'
            else:
                status_text = f'Ammo:{defender.ammo}'

            self.ax.text(defender.x, defender.y - 5, status_text,
                        ha='center', va='top', fontsize=8, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Draw attackers - RED TRIANGLES
        for attacker in self.attackers:
            if not attacker.alive:
                continue

            # Draw attacker as triangle (color from config)
            attacker_color = colors['attacker']
            marker_size = self.viz_config['marker_size_attacker']
            self.ax.plot(attacker.x, attacker.y, marker='^', color=attacker_color,
                        markersize=marker_size, markeredgecolor='dark' + attacker_color, markeredgewidth=1.5)

            # Draw velocity vector
            if attacker.get_speed() > 0.1:
                self.ax.arrow(attacker.x, attacker.y, attacker.vx * 8, attacker.vy * 8,
                             head_width=2, head_length=3, fc='red', ec='darkred', alpha=0.5, linewidth=1.5)

            # Show ID at top
            self.ax.text(attacker.x, attacker.y + 5, f'A{attacker.id}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Show warhead mass below
            self.ax.text(attacker.x, attacker.y - 5, f'WH:{attacker.warhead_mass:.1f}',
                        ha='center', va='top', fontsize=8, color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Draw active engagements (highlight targets)
        for defender_id, attacker_id, time_remaining in self.active_engagements:
            # Find the attacker
            target_attacker = None
            for attacker in self.attackers:
                if attacker.id == attacker_id and attacker.alive:
                    target_attacker = attacker
                    break

            if target_attacker:
                # Draw highlight box around target (size and color from config)
                box_size = self.viz_config['highlight_box_size']
                highlight_color = colors['engagement_highlight']
                highlight_rect = patches.Rectangle(
                    (target_attacker.x - box_size/2, target_attacker.y - box_size/2),
                    box_size, box_size,
                    linewidth=2, edgecolor=highlight_color, facecolor='none', linestyle='--'
                )
                self.ax.add_patch(highlight_rect)

                # Label which defender is targeting
                self.ax.text(target_attacker.x + box_size/2 + 2, target_attacker.y + box_size/2 + 2,
                            f'← D{defender_id}',
                            fontsize=8, color=highlight_color, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Draw active AESA emission cones
        for defender_id, orientation, time_remaining in self.active_aesa_cones:
            # Find the AESA defender
            aesa_defender = None
            for defender in self.defenders:
                if defender.id == defender_id and defender.alive:
                    aesa_defender = defender
                    break

            if aesa_defender and isinstance(aesa_defender, AESAEmitter):
                # Draw cone using Wedge
                cone_color = colors['aesa_cone']
                half_angle = aesa_defender.cone_angle / 2.0

                # Wedge uses counterclockwise angles from east (0°)
                theta1 = orientation - half_angle
                theta2 = orientation + half_angle
                mid_angle_rad = np.radians(orientation)  # Calculate this first

                cone_wedge = patches.Wedge(
                    (aesa_defender.x, aesa_defender.y),
                    aesa_defender.max_range,
                    theta1, theta2,
                    facecolor=cone_color, edgecolor='red',  # Red edge for visibility
                    alpha=0.6, linewidth=3  # Increased opacity and thicker edge
                )
                self.ax.add_patch(cone_wedge)

                # Draw a bright line from emitter in the direction of fire for extra visibility
                direction_line_end_x = aesa_defender.x + aesa_defender.max_range * np.cos(mid_angle_rad)
                direction_line_end_y = aesa_defender.y + aesa_defender.max_range * np.sin(mid_angle_rad)
                self.ax.plot([aesa_defender.x, direction_line_end_x],
                           [aesa_defender.y, direction_line_end_y],
                           color='red', linewidth=3, linestyle='-', alpha=0.8)

                # Add label
                label_distance = aesa_defender.max_range * 0.6
                label_x = aesa_defender.x + label_distance * np.cos(mid_angle_rad)
                label_y = aesa_defender.y + label_distance * np.sin(mid_angle_rad)

                self.ax.text(label_x, label_y, f'AESA D{defender_id}',
                            fontsize=9, color=cone_color, fontweight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # Draw projectiles (no trajectory lines to reduce clutter)
        projectile_color = colors['projectile']
        marker_size = self.viz_config['marker_size_projectile']
        for projectile in self.projectiles:
            # Draw projectile as small circle
            self.ax.plot(projectile.x, projectile.y, 'o', color=projectile_color,
                        markersize=marker_size, markeredgecolor='black', markeredgewidth=1)

        # Info text
        info_text = f'Attackers: {len([a for a in self.attackers if a.alive])} alive, {self.attackers_destroyed} destroyed\n'
        info_text += f'Defenders: {len([d for d in self.defenders if d.alive])} alive\n'
        info_text += f'Projectiles: {len(self.projectiles)} in flight'
        self.ax.text(5, self.map_height - 5, info_text, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='SAM Battery'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=10, label='Kinetic Depot'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Enemy Drone'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Projectile')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.draw()
        # FPS from config: Lower value = faster, higher value = slower
        # 0.05 = ~20 FPS, 0.1 = ~10 FPS, 0.2 = ~5 FPS, 0.5 = ~2 FPS
        plt.pause(self.viz_config['fps_delay'])

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
