from submission import is_orbiting, AgentState, should_be_retained, _as
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import math

def test_is_orbiting():
    sun = (50, 50, 10)
    planet = Planet(0, 1, 61, 55, 1 + math.log(5), 23, 5)
    
    assert is_orbiting(planet)

def test_is_targeted():
    p = Planet(*[0, 0, 0.3242, 0.29943, 3.1224, 15, 5])

    _as.target_size_map[0] = [15, 1]
    is_target_ships = should_be_retained(p, True, 14)
    assert is_target_ships

def test_is_targeted_retain_false_return_true():
    p = Planet(*[0, 0, 0.3242, 0.29943, 3.1224, 15, 5])
    
    _as.target_size_map[0] = [15, 1]
    is_target_ships = should_be_retained(p, False, 14)
    assert is_target_ships