"""
Demo script for AGGRESSIVE heuristic-based drone defense.
Strategy:
- All defenders engage at maximum range
- AESA ALWAYS fires when any target is in range (low cost priority)
- Maximizes firepower, accepts collateral damage
"""
import numpy as np
from drone_defense_env import DroneDefenseEnv
from entities import AESAEmitter


class AggressiveDroneDefenseEnv(DroneDefenseEnv):
    """Aggressive variant: AESA fires whenever possible, all defenders at max range."""

    def _compute_score_matrix_heuristic(self):
        """
        Aggressive heuristic: prioritize AESA and all long-range engagements.
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

        for i, defender in enumerate(alive_defenders):
            for j, attacker in enumerate(alive_attackers):
                # Calculate time until attacker reaches safe zone
                distance_to_safe_zone = np.linalg.norm(
                    attacker.get_position() - safe_zone_center
                )
                speed = attacker.get_speed()
                if speed > 0.01:
                    time_until_safe_zone = distance_to_safe_zone / speed
                else:
                    time_until_safe_zone = float('inf')

                # Base priority
                priority = time_until_safe_zone * attacker.warhead_mass

                # AGGRESSIVE: No penalties for any engagement type
                # All defenders engage at max range

                # Higher priority = lower cost
                if priority > 0:
                    score_matrix[i, j] = 1.0 / priority
                else:
                    score_matrix[i, j] = 1e6

        return score_matrix

    def step(self, action):
        """
        Modified step with aggressive AESA firing.
        AESA fires whenever ANY target is in range, even if aiming south.
        """
        self.timestep += 1

        # Standard steps 1-5
        spawn_prob = self.spawn_prob_initial * (self.spawn_prob_decay ** self.timestep)
        current_alive = len([a for a in self.attackers if a.alive])

        if (np.random.random() < spawn_prob and
            current_alive < self.max_concurrent_attackers and
            self.total_attackers_spawned < self.max_total_attackers):
            self._spawn_attacker()

        adjacency_matrix = self._construct_graph()

        if self.use_attention and self.attention_system is not None:
            score_matrix = self._compute_score_matrix_attention()
        else:
            score_matrix = self._compute_score_matrix_heuristic()

        allocations = self._hungarian_allocation(score_matrix, adjacency_matrix)

        alive_defenders = [d for d in self.defenders if d.alive]
        alive_attackers = [a for a in self.attackers if a.alive]

        for d_idx, a_idx in allocations:
            defender = alive_defenders[d_idx]
            attacker = alive_attackers[a_idx]

            distance = np.linalg.norm(
                defender.get_position() - attacker.get_position()
            )
            pk = defender.probability_of_kill(distance, attacker.get_speed())

            from entities import Projectile
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
            self.active_engagements.append((defender.id, attacker.id, self.engagement_display_time))
            defender.shoot()

        # AGGRESSIVE AESA firing: ALWAYS fire if ANY target in range
        aesa_destroyed_attackers = 0
        aesa_destroyed_projectiles = 0

        for defender in alive_defenders:
            if not isinstance(defender, AESAEmitter):
                continue

            if not defender.can_shoot():
                continue

            # Find nearest attacker within range
            nearest_attacker = None
            min_distance = float('inf')

            for attacker in alive_attackers:
                distance = np.linalg.norm(defender.get_position() - attacker.get_position())
                if distance <= defender.max_range and distance < min_distance:
                    min_distance = distance
                    nearest_attacker = attacker

            if nearest_attacker is None:
                continue

            # Aim at nearest attacker
            dx = nearest_attacker.x - defender.x
            dy = nearest_attacker.y - defender.y
            aim_angle = np.degrees(np.arctan2(dy, dx))
            defender.set_orientation(aim_angle)

            # AGGRESSIVE: ALWAYS fire if target in range (ignore aiming direction penalties)
            # Fire!
            for attacker in alive_attackers:
                if defender.is_in_cone(attacker.x, attacker.y):
                    attacker.alive = False
                    aesa_destroyed_attackers += 1
                    self.attackers_destroyed += 1

            for projectile in self.projectiles:
                if defender.is_in_cone(projectile.x, projectile.y):
                    projectile.active = False
                    aesa_destroyed_projectiles += 1

            # Track with LONG persistence for visibility
            self.active_aesa_cones.append((defender.id, defender.orientation, 30))  # 30 timesteps

            # Print when AESA fires for visibility
            print(f"  💥 AGGRESSIVE AESA D{defender.id} FIRED! Angle: {defender.orientation:.1f}° | "
                  f"Killed: {aesa_destroyed_attackers} attackers, {aesa_destroyed_projectiles} projectiles | "
                  f"Timestep: {self.timestep}")

            defender.shoot()

        # Rest of step function
        attackers_killed_this_step = aesa_destroyed_attackers
        projectiles_to_remove = []

        for projectile in self.projectiles:
            target_attacker = None
            for attacker in self.attackers:
                if attacker.id == projectile.attacker_id and attacker.alive:
                    target_attacker = attacker
                    break

            if target_attacker is None:
                projectiles_to_remove.append(projectile)
                continue

            hit_result = projectile.update(target_attacker.x, target_attacker.y)

            if hit_result is not None:
                if hit_result:
                    target_attacker.alive = False
                    attackers_killed_this_step += 1
                    self.attackers_destroyed += 1
                projectiles_to_remove.append(projectile)

        for projectile in projectiles_to_remove:
            self.projectiles.remove(projectile)

        self.active_engagements = [
            (d_id, a_id, time - 1) for d_id, a_id, time in self.active_engagements if time > 1
        ]
        self.active_aesa_cones = [
            (d_id, orientation, time - 1) for d_id, orientation, time in self.active_aesa_cones if time > 1
        ]

        for defender in self.defenders:
            defender.update()

        for attacker in self.attackers:
            attacker.update()

        safe_zone_breached = self._check_safe_zone_breach()
        if safe_zone_breached:
            self.safe_zone_breached = True

        defenders_lost = 0
        reward = self._calculate_reward(
            allocations, attackers_killed_this_step, defenders_lost, safe_zone_breached
        )

        alive_attackers_count = len([a for a in self.attackers if a.alive])
        terminated = safe_zone_breached or (alive_attackers_count == 0 and self.timestep > 20)
        truncated = self.timestep >= self.max_timesteps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


def run_aggressive_demo(num_episodes: int = 3, render: bool = True):
    """Run aggressive strategy demo."""
    env = AggressiveDroneDefenseEnv(
        use_attention=False,
        render_mode='human' if render else None
    )

    print("=" * 60)
    print("DRONE DEFENSE - AGGRESSIVE STRATEGY")
    print("=" * 60)
    print("\nAggressive Strategy:")
    print("  - All defenders: Engage at MAXIMUM range")
    print("  - AESA: ALWAYS fires when targets in range")
    print("  - Accepts collateral damage (destroys friendly projectiles)")
    print("  - Maximum firepower, high risk/reward")
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
            obs, reward, done, truncated, info = env.step(0)
            total_reward += reward

            if render:
                env.render()

            if info['timestep'] % 20 == 0:
                print(f"  Timestep {info['timestep']:3d} | "
                      f"Attackers: {info['attackers_alive']:2d} alive, "
                      f"{info['attackers_destroyed']:2d} destroyed | "
                      f"Defenders: {info['defenders_alive']} alive | "
                      f"Reward: {total_reward:7.2f}")

        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"  Duration: {info['timestep']} timesteps")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Attackers Destroyed: {info['attackers_destroyed']}")
        print(f"  Result: {'DEFEAT' if info['safe_zone_breached'] else 'VICTORY'}")
        print(f"{'='*60}\n")

        if render:
            input("Press Enter to continue...")

    env.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run aggressive drone defense demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    run_aggressive_demo(num_episodes=args.episodes, render=not args.no_render)
