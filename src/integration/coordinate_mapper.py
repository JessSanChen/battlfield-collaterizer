"""
Coordinate Mapper: Converts between abstract simulation space and real-world coordinates.

Abstract Space (Partner's RL System):
- 200 × 200 units
- Origin at (0, 0)
- Center at (100, 100)

Real World Space (Our Collateral System):
- 20km × 20km area (10km radius around airport)
- Coordinates in latitude/longitude (degrees)
- 1 abstract unit = 100 meters

Airport Centers:
- Taoyuan (TPE): 25.0777°N, 121.2325°E
- Songshan (TSA): 25.0694°N, 121.5517°E
"""

import math
from typing import Tuple


# Constants
EARTH_RADIUS_KM = 6371.0  # Earth's mean radius in kilometers
MAP_SIZE_ABSTRACT = 200  # Abstract map size in units
MAP_SIZE_REAL_KM = 20.0  # Real map size in kilometers (10km radius = 20km diameter)
METERS_PER_UNIT = 100.0  # 1 abstract unit = 100 meters

# Airport coordinates
AIRPORTS = {
    'Taoyuan': {
        'lat': 25.0777,
        'lon': 121.2325,
        'name': 'Taoyuan International Airport (TPE)',
        'bounds': None  # Will be calculated
    },
    'Songshan': {
        'lat': 25.0694,
        'lon': 121.5517,
        'name': 'Songshan Airport (TSA)',
        'bounds': None  # Will be calculated
    }
}


class CoordinateMapper:
    """
    Maps between abstract simulation coordinates and real-world lat/lon.

    Uses local tangent plane approximation for small areas (~20km).
    At Taiwan's latitude (~25°N):
    - 1 degree latitude ≈ 111.32 km
    - 1 degree longitude ≈ 101.5 km (varies with cos(latitude))
    """

    def __init__(self, airport_key: str, map_width: int = 200, map_height: int = 200):
        """
        Initialize coordinate mapper for a specific airport.

        Args:
            airport_key: 'Taoyuan' or 'Songshan'
            map_width: Abstract map width in units (default: 200)
            map_height: Abstract map height in units (default: 200)
        """
        if airport_key not in AIRPORTS:
            raise ValueError(f"Unknown airport: {airport_key}. Must be 'Taoyuan' or 'Songshan'")

        self.airport_key = airport_key
        self.airport_info = AIRPORTS[airport_key]
        self.center_lat = self.airport_info['lat']
        self.center_lon = self.airport_info['lon']
        self.map_width = map_width
        self.map_height = map_height

        # Center of abstract map
        self.center_x = map_width / 2.0
        self.center_y = map_height / 2.0

        # Calculate meters per degree at this latitude
        self.meters_per_deg_lat = 111320.0  # Approximately constant
        self.meters_per_deg_lon = 111320.0 * math.cos(math.radians(self.center_lat))

        # Calculate bounds
        half_size_km = MAP_SIZE_REAL_KM / 2.0
        half_size_m = half_size_km * 1000.0

        self.lat_min = self.center_lat - (half_size_m / self.meters_per_deg_lat)
        self.lat_max = self.center_lat + (half_size_m / self.meters_per_deg_lat)
        self.lon_min = self.center_lon - (half_size_m / self.meters_per_deg_lon)
        self.lon_max = self.center_lon + (half_size_m / self.meters_per_deg_lon)

        # Store bounds
        self.bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        AIRPORTS[airport_key]['bounds'] = self.bounds

    def abstract_to_real(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert abstract (x, y) coordinates to real (lat, lon).

        Abstract coordinate system:
        - (0, 0) is bottom-left
        - (100, 100) is center (airport location)
        - (200, 200) is top-right

        Args:
            x: Abstract x-coordinate (0-200)
            y: Abstract y-coordinate (0-200)

        Returns:
            (latitude, longitude) in degrees
        """
        # Calculate offset from center in abstract units
        dx_abstract = x - self.center_x  # Positive = east
        dy_abstract = y - self.center_y  # Positive = north

        # Convert to meters
        dx_meters = dx_abstract * METERS_PER_UNIT
        dy_meters = dy_abstract * METERS_PER_UNIT

        # Convert to degrees
        dlat = dy_meters / self.meters_per_deg_lat
        dlon = dx_meters / self.meters_per_deg_lon

        # Add to center coordinates
        lat = self.center_lat + dlat
        lon = self.center_lon + dlon

        return lat, lon

    def real_to_abstract(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert real (lat, lon) coordinates to abstract (x, y).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (x, y) in abstract coordinates
        """
        # Calculate offset from center in degrees
        dlat = lat - self.center_lat
        dlon = lon - self.center_lon

        # Convert to meters
        dy_meters = dlat * self.meters_per_deg_lat
        dx_meters = dlon * self.meters_per_deg_lon

        # Convert to abstract units
        dx_abstract = dx_meters / METERS_PER_UNIT
        dy_abstract = dy_meters / METERS_PER_UNIT

        # Add to center coordinates
        x = self.center_x + dx_abstract
        y = self.center_y + dy_abstract

        return x, y

    def distance_meters(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two lat/lon points using Haversine formula.

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))

        distance_km = EARTH_RADIUS_KM * c
        return distance_km * 1000.0  # Convert to meters

    def is_in_bounds(self, lat: float, lon: float) -> bool:
        """Check if a lat/lon coordinate is within the mapped area."""
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get the real-world bounds of the mapped area."""
        return self.bounds

    def __repr__(self) -> str:
        return (f"CoordinateMapper(airport={self.airport_key}, "
                f"center=({self.center_lat:.4f}°N, {self.center_lon:.4f}°E), "
                f"size={MAP_SIZE_REAL_KM}km)")


def run_tests():
    """Run comprehensive tests of the coordinate mapper."""
    print("=" * 80)
    print("COORDINATE MAPPER TEST SUITE")
    print("=" * 80)

    for airport_key in ['Taoyuan', 'Songshan']:
        print(f"\n{'=' * 80}")
        print(f"Testing {airport_key} Airport")
        print(f"{'=' * 80}\n")

        mapper = CoordinateMapper(airport_key)
        center_lat = mapper.center_lat
        center_lon = mapper.center_lon

        print(f"Airport: {mapper.airport_info['name']}")
        print(f"Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
        print(f"Bounds: {mapper.bounds}")
        print(f"Map size: {MAP_SIZE_REAL_KM}km × {MAP_SIZE_REAL_KM}km")
        print(f"Scale: 1 unit = {METERS_PER_UNIT}m\n")

        # Test 1: Center point
        print("Test 1: Center Point (100, 100) → Airport Location")
        print("-" * 60)
        lat, lon = mapper.abstract_to_real(100, 100)
        x, y = mapper.real_to_abstract(center_lat, center_lon)

        print(f"  Abstract (100, 100) → Real ({lat:.6f}°N, {lon:.6f}°E)")
        print(f"  Expected: ({center_lat:.6f}°N, {center_lon:.6f}°E)")
        print(f"  Error: {abs(lat - center_lat):.9f}° lat, {abs(lon - center_lon):.9f}° lon")

        print(f"\n  Real ({center_lat:.6f}°N, {center_lon:.6f}°E) → Abstract ({x:.2f}, {y:.2f})")
        print(f"  Expected: (100.00, 100.00)")
        print(f"  Error: {abs(x - 100):.6f} units x, {abs(y - 100):.6f} units y")

        assert abs(lat - center_lat) < 1e-6, "Center lat conversion failed"
        assert abs(lon - center_lon) < 1e-6, "Center lon conversion failed"
        assert abs(x - 100) < 0.01, "Center x conversion failed"
        assert abs(y - 100) < 0.01, "Center y conversion failed"
        print("  ✓ PASS\n")

        # Test 2: 10km west (left edge)
        print("Test 2: Left Edge (0, 100) → 10km West of Airport")
        print("-" * 60)
        lat_west, lon_west = mapper.abstract_to_real(0, 100)
        distance_west = mapper.distance_meters(center_lat, center_lon, lat_west, lon_west)

        print(f"  Abstract (0, 100) → Real ({lat_west:.6f}°N, {lon_west:.6f}°E)")
        print(f"  Distance from center: {distance_west:.1f}m ({distance_west/1000:.2f}km)")
        print(f"  Expected: ~10,000m (10km west)")
        print(f"  Error: {abs(distance_west - 10000):.1f}m ({abs(distance_west - 10000)/1000:.3f}km)")

        # Allow 1% error due to spherical geometry approximation
        assert abs(distance_west - 10000) < 100, "10km west conversion error too large"
        assert lon_west < center_lon, "West should have smaller longitude"
        print("  ✓ PASS\n")

        # Test 3: 10km east (right edge)
        print("Test 3: Right Edge (200, 100) → 10km East of Airport")
        print("-" * 60)
        lat_east, lon_east = mapper.abstract_to_real(200, 100)
        distance_east = mapper.distance_meters(center_lat, center_lon, lat_east, lon_east)

        print(f"  Abstract (200, 100) → Real ({lat_east:.6f}°N, {lon_east:.6f}°E)")
        print(f"  Distance from center: {distance_east:.1f}m ({distance_east/1000:.2f}km)")
        print(f"  Expected: ~10,000m (10km east)")
        print(f"  Error: {abs(distance_east - 10000):.1f}m ({abs(distance_east - 10000)/1000:.3f}km)")

        assert abs(distance_east - 10000) < 100, "10km east conversion error too large"
        assert lon_east > center_lon, "East should have larger longitude"
        print("  ✓ PASS\n")

        # Test 4: 10km north (top edge)
        print("Test 4: Top Edge (100, 200) → 10km North of Airport")
        print("-" * 60)
        lat_north, lon_north = mapper.abstract_to_real(100, 200)
        distance_north = mapper.distance_meters(center_lat, center_lon, lat_north, lon_north)

        print(f"  Abstract (100, 200) → Real ({lat_north:.6f}°N, {lon_north:.6f}°E)")
        print(f"  Distance from center: {distance_north:.1f}m ({distance_north/1000:.2f}km)")
        print(f"  Expected: ~10,000m (10km north)")
        print(f"  Error: {abs(distance_north - 10000):.1f}m ({abs(distance_north - 10000)/1000:.3f}km)")

        assert abs(distance_north - 10000) < 100, "10km north conversion error too large"
        assert lat_north > center_lat, "North should have larger latitude"
        print("  ✓ PASS\n")

        # Test 5: 10km south (bottom edge)
        print("Test 5: Bottom Edge (100, 0) → 10km South of Airport")
        print("-" * 60)
        lat_south, lon_south = mapper.abstract_to_real(100, 0)
        distance_south = mapper.distance_meters(center_lat, center_lon, lat_south, lon_south)

        print(f"  Abstract (100, 0) → Real ({lat_south:.6f}°N, {lon_south:.6f}°E)")
        print(f"  Distance from center: {distance_south:.1f}m ({distance_south/1000:.2f}km)")
        print(f"  Expected: ~10,000m (10km south)")
        print(f"  Error: {abs(distance_south - 10000):.1f}m ({abs(distance_south - 10000)/1000:.3f}km)")

        assert abs(distance_south - 10000) < 100, "10km south conversion error too large"
        assert lat_south < center_lat, "South should have smaller latitude"
        print("  ✓ PASS\n")

        # Test 6: Round-trip conversion
        print("Test 6: Round-Trip Conversion Accuracy")
        print("-" * 60)
        test_points = [
            (50, 75, "Southwest quadrant"),
            (150, 75, "Southeast quadrant"),
            (150, 125, "Northeast quadrant"),
            (50, 125, "Northwest quadrant"),
            (100, 100, "Center"),
        ]

        max_error_x = 0
        max_error_y = 0

        for x_orig, y_orig, label in test_points:
            lat, lon = mapper.abstract_to_real(x_orig, y_orig)
            x_back, y_back = mapper.real_to_abstract(lat, lon)

            error_x = abs(x_back - x_orig)
            error_y = abs(y_back - y_orig)
            max_error_x = max(max_error_x, error_x)
            max_error_y = max(max_error_y, error_y)

            print(f"  {label:20s}: ({x_orig:6.1f}, {y_orig:6.1f}) → "
                  f"({lat:.6f}, {lon:.6f}) → ({x_back:6.1f}, {y_back:6.1f})")
            print(f"                        Error: {error_x:.6f} units x, {error_y:.6f} units y")

        print(f"\n  Maximum round-trip error: {max_error_x:.6f} units x, {max_error_y:.6f} units y")
        assert max_error_x < 0.01, "Round-trip error too large in x"
        assert max_error_y < 0.01, "Round-trip error too large in y"
        print("  ✓ PASS\n")

        # Test 7: Bounds checking
        print("Test 7: Bounds Checking")
        print("-" * 60)
        print(f"  Bounds: ({mapper.lon_min:.6f}°E, {mapper.lat_min:.6f}°N) to "
              f"({mapper.lon_max:.6f}°E, {mapper.lat_max:.6f}°N)")

        # Points that should be in bounds
        assert mapper.is_in_bounds(center_lat, center_lon), "Center should be in bounds"
        assert mapper.is_in_bounds(lat_north, lon_north), "North edge should be in bounds"

        # Points that should be out of bounds
        far_north_lat = center_lat + 1.0  # 1 degree north (~111km)
        assert not mapper.is_in_bounds(far_north_lat, center_lon), "Far north should be out of bounds"
        print("  ✓ PASS\n")

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nCoordinate mapper is ready for integration with RL system.")
    print(f"Scale: 1 abstract unit = {METERS_PER_UNIT}m")
    print(f"Map size: {MAP_SIZE_ABSTRACT} × {MAP_SIZE_ABSTRACT} units = {MAP_SIZE_REAL_KM}km × {MAP_SIZE_REAL_KM}km")


if __name__ == '__main__':
    run_tests()
