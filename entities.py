"""
Entity definitions for the drone defense simulation.
"""
import numpy as np
from typing import Optional

# Global ID counters
_defender_id_counter = 0
_attacker_id_counter = 0
_projectile_id_counter = 0


def reset_id_counters():
    """Reset all ID counters (useful for new episodes)."""
    global _defender_id_counter, _attacker_id_counter, _projectile_id_counter
    _defender_id_counter = 0
    _attacker_id_counter = 0
    _projectile_id_counter = 0


class Defender:
    """Base class for defenders."""

    def __init__(self, x: float, y: float, max_range: float, ammo: int,
                 reload_time: int, shot_speed: float, defender_type: str):
        global _defender_id_counter
        self.id = _defender_id_counter
        _defender_id_counter += 1

        self.x = x
        self.y = y
        self.max_range = max_range
        self.ammo = ammo
        self.max_ammo = ammo
        self.reload_time = reload_time
        self.reload_counter = 0
        self.shot_speed = shot_speed
        self.defender_type = defender_type
        self.alive = True

    def probability_of_kill(self, distance: float, attacker_speed: float) -> float:
        """
        Calculate probability of kill based on range and attacker state.

        Args:
            distance: Distance to target
            attacker_speed: Speed of the attacker

        Returns:
            Probability of kill (0 to 1)
        """
        if distance > self.max_range:
            return 0.0

        # Base probability decreases with distance
        range_factor = 1.0 - (distance / self.max_range)

        # Harder to hit fast-moving targets
        speed_factor = 1.0 / (1.0 + attacker_speed * 0.1)

        return range_factor * speed_factor * self.base_pk

    def can_shoot(self) -> bool:
        """Check if defender can shoot."""
        return self.ammo > 0 and self.reload_counter == 0 and self.alive

    def shoot(self):
        """Consume ammunition and trigger reload."""
        if self.can_shoot():
            self.ammo -= 1
            self.reload_counter = self.reload_time

    def update(self):
        """Update defender state (e.g., reload counter)."""
        if self.reload_counter > 0:
            self.reload_counter -= 1

    def get_position(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array([self.x, self.y])

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert defender to feature vector for neural network.

        Returns:
            Feature vector: [x, y, ammo_ratio, reload_status, type_sam, type_kinetic]
        """
        ammo_ratio = self.ammo / self.max_ammo if self.max_ammo > 0 else 0.0
        reload_status = self.reload_counter / self.reload_time if self.reload_time > 0 else 0.0
        type_sam = 1.0 if self.defender_type == 'SAM' else 0.0
        type_kinetic = 1.0 if self.defender_type == 'KINETIC' else 0.0

        return np.array([
            self.x / 200.0,  # Normalize to ~0-1 range assuming map size 200
            self.y / 200.0,
            ammo_ratio,
            reload_status,
            type_sam,
            type_kinetic
        ], dtype=np.float32)


class SAMBattery(Defender):
    """Surface-to-Air Missile battery."""

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            max_range=40.0,  # Long range
            ammo=10,
            reload_time=3,  # 3 timesteps to reload
            shot_speed=15.0,  # Fast missiles
            defender_type='SAM'
        )
        self.base_pk = 0.85  # High kill probability


class KineticDroneDepot(Defender):
    """Kinetic drone depot that launches interceptor drones."""

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            max_range=30.0,  # Medium range
            ammo=20,
            reload_time=1,  # Fast reload
            shot_speed=8.0,  # Slower than SAM
            defender_type='KINETIC'
        )
        self.base_pk = 0.70  # Lower kill probability than SAM


class Attacker:
    """Enemy drone attacker."""

    def __init__(self, x: float, y: float, vx: float, vy: float, warhead_mass: float):
        global _attacker_id_counter
        self.id = _attacker_id_counter
        _attacker_id_counter += 1

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.warhead_mass = warhead_mass
        self.alive = True

    def update(self):
        """Update attacker position based on velocity."""
        if self.alive:
            self.x += self.vx
            self.y += self.vy

    def get_position(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array([self.x, self.y])

    def get_speed(self) -> float:
        """Get current speed."""
        return np.sqrt(self.vx**2 + self.vy**2)

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert attacker to feature vector for neural network.

        Returns:
            Feature vector: [x, y, vx, vy, warhead_mass, speed]
        """
        return np.array([
            self.x / 200.0,  # Normalize to ~0-1 range assuming map size 200
            self.y / 200.0,
            self.vx / 5.0,  # Normalize assuming max velocity ~5
            self.vy / 5.0,
            self.warhead_mass / 10.0,  # Normalize (1-10 range)
            self.get_speed() / 7.0  # Normalize speed
        ], dtype=np.float32)


class Projectile:
    """
    Projectile fired by a defender at an attacker.
    Travels over time and can hit or miss based on probability.
    """

    def __init__(self, defender_id: int, attacker_id: int,
                 start_x: float, start_y: float,
                 target_x: float, target_y: float,
                 speed: float, probability_of_kill: float):
        global _projectile_id_counter
        self.id = _projectile_id_counter
        _projectile_id_counter += 1

        self.defender_id = defender_id
        self.attacker_id = attacker_id

        self.x = start_x
        self.y = start_y
        self.target_x = target_x
        self.target_y = target_y

        # Calculate velocity to reach target
        dx = target_x - start_x
        dy = target_y - start_y
        distance = np.sqrt(dx**2 + dy**2)

        if distance > 0:
            self.vx = (dx / distance) * speed
            self.vy = (dy / distance) * speed
        else:
            self.vx = 0
            self.vy = 0

        self.speed = speed
        self.pk = probability_of_kill
        self.active = True
        self.hit_result = None  # None = still flying, True = hit, False = miss

        # Calculate estimated time to target
        if speed > 0:
            self.time_to_target = distance / speed
        else:
            self.time_to_target = 0

        self.time_elapsed = 0

    def update(self, attacker_x: float, attacker_y: float) -> Optional[bool]:
        """
        Update projectile position.

        Args:
            attacker_x, attacker_y: Current position of target attacker

        Returns:
            None if still flying, True if hit, False if miss
        """
        if not self.active:
            return self.hit_result

        # Move projectile
        self.x += self.vx
        self.y += self.vy
        self.time_elapsed += 1

        # Check if we've reached the target area (within small threshold)
        distance_to_target = np.sqrt((self.x - attacker_x)**2 + (self.y - attacker_y)**2)

        # If we're close enough or exceeded time to target, resolve hit/miss
        if distance_to_target < 2.0 or self.time_elapsed >= self.time_to_target + 1:
            self.active = False
            # Roll for hit based on probability of kill
            self.hit_result = np.random.random() < self.pk
            return self.hit_result

        return None

    def get_position(self) -> np.ndarray:
        """Get current position."""
        return np.array([self.x, self.y])

    def get_trajectory_line(self) -> tuple:
        """Get trajectory line for visualization."""
        return (self.x - self.vx * 3, self.y - self.vy * 3, self.x, self.y)
