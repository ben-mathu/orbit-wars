from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS
import json
from submission import *

with open("../episode-77084355.json", "r") as f:
    episode = json.load(f)
    orbiting_map = {}
    for e in episode["steps"]:
        planets = [Planet(*p) for p in e[0]["observation"]["planets"]]
        # planets += [Planet(*p) for p in e[1]["observation"]["planets"]]
        planets = [p for p in planets if is_orbiting(p)]
        
        for p in planets:
            orbiter = (round(p.x, 3), round(p.y, 3), p.id)
            if orbiter in orbiting_map:
                orbiting_map[orbiter] += 1
            else:
                orbiting_map[orbiter] = 1
                
    for k, v in orbiting_map.items(): print(f"Orbits: {k} {v} times")
