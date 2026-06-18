import math
import pytest
from submission_2 import get_fleet_speed, predict_planet_pos, find_intercept, _as
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def test_fleet_speed():
    # 1 ship -> 1.0
    assert get_fleet_speed(1) == 1.0
    # 1000 ships -> 6.0
    assert math.isclose(get_fleet_speed(1000), 6.0, abs_tol=1e-5)
    # Intermediate
    assert 1.0 < get_fleet_speed(100) < 6.0

def test_predict_planet_pos_static():
    p = Planet(0, -1, 20, 20, 1, 10, 1) # Non-orbiting (radius=1, dist to sun=math.hypot(30,30) approx 42.4)
    # Wait, 42.4 + 1 < 50, so it IS orbiting.
    # Let's make it definitely static:
    p = Planet(0, -1, 0, 0, 1, 10, 1) # Dist to sun = 70.7. 70.7 + 1 > 50 -> Static.
    tx, ty = predict_planet_pos(p, 10)
    assert tx == 0 and ty == 0

def test_predict_planet_pos_orbiting():
    _as.angular_v = 0.1
    # Planet at (60, 50) -> dist 10 from sun, angle 0
    p = Planet(0, -1, 60, 50, 1, 10, 1)
    
    # After 1 turn, angle should be 0.1
    tx, ty = predict_planet_pos(p, 1)
    expected_x = 50 + 10 * math.cos(0.1)
    expected_y = 50 + 10 * math.sin(0.1)
    assert math.isclose(tx, expected_x)
    assert math.isclose(ty, expected_y)

def test_find_intercept_static():
    _as.angular_v = 0.0
    source = Planet(1, 0, 50, 80, 1, 100, 1) # Player 0
    target = Planet(2, -1, 50, 20, 1, 10, 1) # Neutral, dist 60
    
    # Target ships = 10. ships_needed = 11.
    # Speed for 11 ships:
    speed = get_fleet_speed(11)
    expected_t = math.ceil(60 / speed)
    
    result = find_intercept(source, target)
    assert result is not None
    angle, ships_needed, t = result
    assert ships_needed == 11
    assert t == expected_t
    assert math.isclose(angle, -math.pi / 2) # Pointing up

if __name__ == "__main__":
    pytest.main([__file__])
