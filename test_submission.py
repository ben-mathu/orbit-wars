from submission import is_orbiting
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import math

def test_is_orbiting():
    sun = (50, 50, 10)
    planet = Planet(0, 1, 61, 55, 1 + math.log(5), 23, 5)
    
    assert is_orbiting(planet)