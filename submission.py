import functools
import math
import random
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS

INTERCEPT_THRESHOLD = 0.1
DAMPING_FACTOR = 0.05
FLEET_LAUNCH_THRESHOLD = 0.1 # retaining planet ship when launching does not help, occupy as much as possible
total_targets = 5

sun_config = (50.0, 50.0, 10.0)
        
time = 1

fleet_target_map = {} # maps already targeted planets
launch_history = []
planets_orbit_paths = {}
fleet_speed = {}
participation = 0
generation = 1
total_error = 0.0
best_threshold = INTERCEPT_THRESHOLD
best_damping = DAMPING_FACTOR
best_score = float("inf")
cache_score = 0.0
        
def cal_intercept(planet, angular_v, launcher_pos, threshold = INTERCEPT_THRESHOLD, damping = 1.0):
    dx = planet.x - sun_config[0]
    dy = planet.y - sun_config[1]
    
    current_angle = math.atan2(dy, dx)
    planet_r = cal_hypotenus(sun_config[0], sun_config[1], planet.x, planet.y) # distance fromt the sun's center to the target planet
    
    ships_needed = planet.ships + 1
    initial_dist = cal_hypotenus(launcher_pos.x, launcher_pos.y, planet.x, planet.y)
    t = initial_dist / cal_fleet_speed(ships_needed)
    
    tx = planet.x
    ty = planet.y
    
    for _ in range(100):
        # angular_v * time = angular displacement
        target_angle = current_angle + (angular_v * t) # calculate the next angle given the time, t
        
        # cal position from the new angle
        tx = sun_config[0] + (planet_r * math.cos(target_angle))
        ty = sun_config[1] + (planet_r * math.sin(target_angle))
        
        # cal the distance and time the planet would travel
        dist = cal_hypotenus(launcher_pos.x, launcher_pos.y, tx, ty)
        
        prod = planet.production if planet.owner > -1 else 0
        ships_needed = (t * prod) + planet.ships + 1

        travel_time = dist / cal_fleet_speed(ships_needed)
        
        # cal dot_product and miss distance from fleet/launch pos to target
        dx = tx - launcher_pos.x
        dy = tx - launcher_pos.y
        launch_angle = math.atan2(dy, dx)
        miss_distance = abs(dx * math.sin(launch_angle) - dy * math.cos(launch_angle))

        if abs(t - travel_time) < threshold and miss_distance <= planet.radius: # Threshold for 'close enough'
            arrival_turn = math.ceil(t)
            fleet_travel_dist = ships_needed * arrival_turn
            return abs(dist - fleet_travel_dist), tx, ty, math.ceil(ships_needed)
        
        # t = (t * ((1.0 - damping) + (travel_time + damping)))
        t = travel_time
    return None

def evaluate_policy(planet, angular_v, launcher_pos, test_threshold, test_damping):
    global total_error
    result = cal_intercept(planet, angular_v, launcher_pos, test_threshold, test_damping)
    if result:
        error, tx, ty, speed = result
        total_error += error
        
        return total_error / time, tx, ty, speed
    return None

def run_policy_search(planet, angular_v, launcher_pos):
    global generation
    global best_damping
    global best_score
    global best_threshold

    result = evaluate_policy(planet, angular_v, launcher_pos, best_threshold, best_damping)
    if result:
        score, tx, ty, speed = result
        for generation in range(1, 51):
            noise_scale = max(0.01, 1.0 / generation)

            test_threshold = max(1e-5, best_threshold + random.gauss(0, 0.1 * noise_scale))
            test_damping = max(0.1, min(0.9, best_damping + random.gauss(0, 0.2 * noise_scale)))
            # test_threshold = max(1e-4, best_threshold + random.gauss(0, 0.01))
            # test_damping = max(0.1, min(0.9, best_damping + random.gauss(0, 0.05)))
            result = evaluate_policy(planet, angular_v, launcher_pos, test_threshold, test_damping)

            if result:
                score, _, _, _ = result
                if score < best_score:
                    best_score = score
                    best_threshold = test_threshold
                    best_damping = test_damping
                    # generation += 1

            return score, tx, ty, speed
    return None

def cal_intercept_with_sun(start_x, start_y, nearest_x, nearest_y, angle):
    """calculations to check if fleet will hit the sun

    Args:
        start_x (float): launch position x
        start_y (float): launch position y
        nearest_x (float): target x
        nearest_y (float): target y
        angle (float): angle of launch to sun

    Returns:
        tuple:
            - distance from the sun's center to closest to sun,
            - dot_product - if target is behind or in front of sun,
            - nearest_dot_product - check if target is behind or in front of sun
    """
    dx = sun_config[0] - start_x
    dy = sun_config[1] - start_y
    
    nx = sun_config[0] - nearest_x
    ny = sun_config[1] - nearest_y
    
    miss_distance = abs(dx * math.sin(angle) - dy * math.cos(angle))
    dot_product = dx * math.cos(angle) + dy * math.sin(angle)
    
    nearest_angle = math.atan2(ny, nx) # calculates the angle of target to sun
    nearest_dot_product = nx * math.cos(nearest_angle) + ny * math.sin(nearest_angle)
    return miss_distance, dot_product, nearest_dot_product
        
def get_angle(dx, dy, target, angular_velocity, my_planet):
    angle = math.atan2(dy, dx)
    tx, ty, ships_needed = target.x, target.y, target.ships + 1
    if is_orbiting(target):
        result = cal_intercept(target, angular_velocity, my_planet)
        if result != None:
            _, tx, ty, ships_needed = result
            angle = math.atan2(ty - my_planet.y, tx - my_planet.x)
            return angle, tx, ty, ships_needed
    elif target.owner == -1:
        return angle, tx, ty, ships_needed
    else:
        # Calculates time to arrive and compares with travel time
        # production, distance, current number of ships
        distance = cal_hypotenus(my_planet.x, my_planet.y, target.x, target.y)
        for t in range(1, 20):
            num_ships = (t * target.production) + target.ships
            travel_time = distance / cal_fleet_speed(num_ships)
            
            if travel_time <= t: # check if the time take is optimal, larger number of ship == less time
                ships_needed = num_ships
                break
        return angle, tx, ty, math.ceil(ships_needed)
    return None

def get_inputs(obs):
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    
    comet_planet_ids = obs.get("comet_planet_ids", [])
    
    fleets_owned = [f for f in fleets if f.owner == player]
    
    return player, planets, fleets_owned, comet_planet_ids

@functools.lru_cache(maxsize=1024)
def cal_fleet_speed(num_of_ships):
    if num_of_ships in fleet_speed:
        return fleet_speed[num_of_ships]
    
    speed = 1.0 + (6.0 - 1.0) * (math.log(num_of_ships) / math.log(1000)) ** 1.5
    fleet_speed[num_of_ships] = speed
    return speed

def sync_new_fleets(fleet_owned):
    for f in fleet_owned:
        if f.id not in fleet_target_map:
            for i, record in enumerate(launch_history):
                origin_id, target_id, intent_angle, turn = record # turn - time in orbit wars 1-500
                
                # check where the fleet came from
                # check if ownership has change (target id is in planet ownership map)
                if f.from_planet_id == origin_id and math.isclose(f.angle, intent_angle, abs_tol=1e-4):
                    fleet_target_map[f.id] = target_id
                    launch_history.pop(i)
                    break
    
    launch_history[:] = [r for r in launch_history if time - r[3] < 3]

def cal_hypotenus(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)

def get_moves(target, reserved_targets, mine, angular_velocity, comet_planet_ids):
    # reserved_targets - target with launched fleet
    if target.id in comet_planet_ids or target.id in reserved_targets:
        return None

    dx = target.x - mine.x
    dy = target.y - mine.y
        
    result = get_angle(dx, dy, target, angular_velocity, mine)
    if result == None: return None
    
    angle, tx, ty, ships_needed = result
    if (tx, ty) == (target.x, target.y) and is_orbiting(target): return None
        
    if mine.ships >= ships_needed:
        distance_r, dot_product, nearest_dot_product = cal_intercept_with_sun(mine.x, mine.y, tx, ty, angle)

        move = [mine.id, angle, ships_needed]
        if not (dot_product > 0 and distance_r <= sun_config[2]) or nearest_dot_product > 0:
            launch_history.append([mine.id, target.id, angle, time])
            reserved_targets.add(target.id)
            return move

def is_orbiting(planet):
    sun_planet_distance = cal_hypotenus(sun_config[0], sun_config[1], planet.x, planet.y)
    return sun_planet_distance + planet.radius < 50

def cal_all_planet_orbits(planets, angular_v):
    for planet in planets:
        if is_orbiting(planet):
            dx = planet.x - sun_config[0]
            dy = planet.y - sun_config[1]
            
            R = math.hypot(dx, dy)
            current_angle = math.atan2(dy, dx)
            planets_orbit_paths[planet.id] = {}
            
            for t in range(100):
                angle = current_angle + (angular_v * t)
                
                tx = sun_config[0] + (R * math.cos(angle))
                ty = sun_config[1] + (R * math.sin(angle))
                planets_orbit_paths[planet.id][time+t] = (tx, ty)

def agent(obs):
    global time
    global participation
    global fleet_target_map
    
    moves = []
    player, planets, fleets_owned, comet_planet_ids = get_inputs(obs)

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    
    total_targets = 3 if my_planets < targets else int(FLEET_LAUNCH_THRESHOLD * len(targets))
    
    if not targets:
        return []
    
    new_fleet_target_map = {}
    for fid, tid in fleet_target_map.items():
        t = next((p for p in planets if p.id == tid), None)
        if not t : continue
        
        f = next((f for f in fleets_owned if f.id == fid), None)
        if not f: continue
        
        if t.owner == player or t.owner == -1:
            new_fleet_target_map[fid] = tid
        else:
            dist = cal_hypotenus(f.x, f.y, t.x, t.y)
            eta = dist / cal_fleet_speed(f.ships)
            future_defense = t.ships + (t.production * eta)
            
            if f.ships > future_defense:
                new_fleet_target_map[fid] = tid
                
    fleet_target_map = new_fleet_target_map
    
    sync_new_fleets(fleets_owned)
    reserved_targets = set(fleet_target_map.values())

    if len(my_planets) > len(targets):
        for target in targets:
            # get sorted list of nearest planets owned to launch from
            next_nearest_list = sorted(my_planets, key=lambda mine: cal_hypotenus(mine.x, mine.y, target.x, target.y))
            for mine in next_nearest_list:
                move = get_moves(target, reserved_targets, mine, obs.angular_velocity, comet_planet_ids)
                if move != None:
                    moves.append(move)
    else:
        for mine in my_planets:
            # Find nearest 3 planets we don't own
            next_nearest_list = sorted(targets, key=lambda t: cal_hypotenus(mine.x, mine.y, t.x, t.y))[:total_targets]
            for target in next_nearest_list:
                move = get_moves(target, reserved_targets, mine, obs.angular_velocity, comet_planet_ids)
                if move != None:
                    moves.append(move)
    time += 1
    participation += 1
    # print(f"Best Threshold: {best_threshold}, Best Damping: {best_damping}")
    return moves