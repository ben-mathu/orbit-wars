import functools
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS

INTERCEPT_THRESHOLD = 0.1
DAMPING_FACTOR = 0.05
FLEET_LAUNCH_THRESHOLD = 0.1 # retaining planet ship when launching does not help, occupy as much as possible

sun_config = (50.0, 50.0, 10.0)
        
time = 0

fleet_target_map = {} # maps already targeted planets
launch_history = []

comets_expire_time_map = {}

planets_num = (20, 40)
target_range = (4, 7)
time_past = 0

planet_production_range = (1, 5)
percentage_range = (0.3, 0.8)
show_logs = False

last_know_size_map = {}

def log(message: str):
    if show_logs: print(message)

def interpolate(input, input_range, output_range):
    ratio = (input - input_range[0]) / (input_range[1] - input_range[0])
    scale = output_range[1] - output_range[0]
    return output_range[0] + ratio * scale

def min_retention(planet_size, planet_prod):
    perc = interpolate(planet_prod, planet_production_range, percentage_range)
    return int(perc / 100 * planet_size)

def map_total_targets(actual):
    return int(interpolate(actual, planets_num, target_range))

def cal_intercept_2(target, angular_v, origin_x, origin_y, launch_perc=None):    
    dx = target.x - sun_config[0]
    dy = target.y - sun_config[1]
    current_angle = math.atan2(dy, dx)
    planet_r = cal_hypotenus(sun_config[0], sun_config[1], target.x, target.y) # distance fromt the sun's center to the target planet

    for t in range(0, 30):
        if target.owner > -1:
            ships = (t * target.production) + target.ships + 1
            ships_needed =  launch_perc * ships if launch_perc else ships
        else:
            ships_needed = target.ships + 1
            
        ships_needed = math.ceil(ships_needed)

        next_angle = current_angle + (angular_v * t)
        
        tx = sun_config[0] + (planet_r * math.cos(next_angle))
        ty = sun_config[1] + (planet_r * math.sin(next_angle))
        
        dist = cal_hypotenus(origin_x, origin_y, tx, ty)
        
        # round down because fleet over shot - this fixed 99 % of launches
        if math.floor(dist/cal_fleet_speed(ships_needed)) <= t:
            return None, tx, ty, ships_needed
        
    return None
        

def cal_intercept_with_transformations(target, angular_v, origin_x, origin_y, threshold = INTERCEPT_THRESHOLD, damping = 1.0, launch_perc=None):
    initial_dist = cal_hypotenus(origin_x, origin_y, target.x, target.y)

    ships_needed = target.ships + 1 # minimum ships neeeded
    t = initial_dist / cal_fleet_speed(ships_needed) # minimum time
    
    dx = target.x - sun_config[0]
    dy = target.y - sun_config[1]
    
    for _ in range(5):
        angular_displacement = angular_v * t
        cos_t = math.cos(angular_displacement)
        sin_t = math.sin(angular_displacement)

        tx = (dx * cos_t) - (dy * sin_t) + sun_config[0]
        ty = (dx * sin_t) + (dy * cos_t) + sun_config[1]
        
        dist = math.hypot(origin_x - tx, origin_y - ty)
        
        # Dynamic defense update
        if target.owner > -1:
            ships = (t * target.production) + target.ships + 1
            ships_needed =  launch_perc * ships if launch_perc else ships
            
        travel_time = dist / cal_fleet_speed(ships_needed)
        
        if abs(t - travel_time) < threshold:
            arrival_turn = math.ceil(t)
            fleet_travel_dist = ships_needed * arrival_turn
            return abs(dist - fleet_travel_dist), tx, ty, ships_needed
        
        # Fixed-point iteration step to adjust our time guess
        t = (t * (1.0 - damping)) + (travel_time * damping)
    return None

def cal_intercept(target, angular_v, player, origin_x, origin_y, threshold = INTERCEPT_THRESHOLD, damping = 1.0, launch_perc=None):
    """calculates the intercept point from an origin(fleet/planet) to an orbiting planet

    Args:
        target (Planet): target planet
        angular_v (float): rate of displacement of an orbiting planet
        origin_x (float): origin point x
        origin_y (float): origin point x
        threshold (float, optional): least time difference between the orbiting planet and arrival time. Defaults to INTERCEPT_THRESHOLD.
        damping (float, optional): I don't know it is used in RL to reduce noise. Defaults to 1.0.
        launch_perc (float): minmum percentage of fleet to deploy
        
    Returns:
        tuple: score in RL, coordinates and fleet speed
    """
    dx = target.x - sun_config[0]
    dy = target.y - sun_config[1]
    
    current_angle = math.atan2(dy, dx)
    planet_r = cal_hypotenus(sun_config[0], sun_config[1], target.x, target.y) # distance fromt the sun's center to the target planet
    
    initial_dist = cal_hypotenus(origin_x, origin_y, target.x, target.y)

    ships_needed = target.ships + 1 # minimum ships neeeded
    t = initial_dist / cal_fleet_speed(ships_needed) # minimum time
    
    tx = target.x
    ty = target.y
    
    for _ in range(10):
        # angular_v * time = angular displacement
        target_angle = current_angle + (angular_v * t) # calculate the next angle given the time, t
        
        # cal position from the new angle
        tx = sun_config[0] + (planet_r * math.cos(target_angle))
        ty = sun_config[1] + (planet_r * math.sin(target_angle))
        
        # cal the distance and time the planet would travel
        dist = cal_hypotenus(origin_x, origin_y, tx, ty)
        
        prod = target.production if target.owner > -1 and target.owner != player else 0
        ships_needed = launch_perc * ((t * prod) + target.ships + 1) if prod == 0 and launch_perc else (t * prod) + target.ships + 1

        travel_time = dist / cal_fleet_speed(ships_needed)
        
        # cal dot_product and miss distance from fleet/launch pos to target
        # calculating miss distance is unnecessary because projected coordinates is where the target center will be

        if abs(t - travel_time) < threshold: # Threshold for 'close enough'
            arrival_turn = math.ceil(t)
            fleet_travel_dist = ships_needed * arrival_turn
            return abs(dist - fleet_travel_dist), tx, ty, ships_needed
    
        t = travel_time
    return None

def collides_with_sun(start_x, start_y, target_x, target_y, angle):
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
    
    distance_to_sun = cal_hypotenus(start_x, start_y, sun_config[0], sun_config[1])
    distance_to_target = cal_hypotenus(start_x, start_y, target_x, target_y)
    will_hit = collides_with_obstacle(start_x, start_y, sun_config[0], sun_config[1], angle, sun_config[2])
    return will_hit and distance_to_target > distance_to_sun

def collides_with_obstacle(start_x, start_y, px, py, angle, radius):
    """checks collision with (px, py) given the angle of projection

    Args:
        start_x (_type_): _description_
        start_y (_type_): _description_
        px (_type_): _description_
        py (_type_): _description_
        angle (_type_): _description_
        radius (_type_): _description_

    Returns:
        _type_: _description_
    """
    dx = px - start_x
    dy = py - start_y
    
    miss_distance = abs(dx * math.sin(angle) - dy * math.cos(angle))
    dot_product = dx * math.cos(angle) + dy * math.sin(angle)
    
    if dot_product > 0 and miss_distance <= radius:
        return True
    return False

def search_collisions(start_x, start_y, launch_id, target_id, target_x, target_y, angle, planets):
    for p in planets:
        if p.id == target_id or p.id == launch_id: continue
        
        distance_to_target = cal_hypotenus(start_x, start_y, target_x, target_y)
        distance_to_obstacal = cal_hypotenus(start_x, start_y, p.x, p.y)
        
        if collides_with_obstacle(start_x, start_y, p.x, p.y, angle, p.radius) and distance_to_target > distance_to_obstacal:
            return True
    return False
        
def get_angle(dx, dy, target, angular_velocity, my_planet, launch_perc):
    angle = math.atan2(dy, dx)
    if is_orbiting(target):
        result = cal_intercept_2(target, angular_velocity, my_planet.x, my_planet.y, launch_perc=launch_perc)
        if result != None:
            _, tx, ty, ships_needed = result
            angle = math.atan2(ty - my_planet.y, tx - my_planet.x)
            return angle, ships_needed, tx, ty
    elif target.owner == -1:
        ships_needed = launch_perc * (target.ships + 1) if launch_perc else target.ships + 1
        return angle, ships_needed, target.x, target.y
    else:
        # Calculates time to arrive and compares with travel time
        # production, distance, current number of ships
        distance = cal_hypotenus(my_planet.x, my_planet.y, target.x, target.y)
        for t in range(1, 100):
            num_ships = (t * target.production) + target.ships
            travel_time = distance / cal_fleet_speed(num_ships)
            
            if travel_time <= t: # check if the time take is optimal, larger number of ship == less time
                ships_needed = num_ships
                return angle, math.ceil(ships_needed), target.x, target.y
    return None

def get_inputs(obs):
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    
    fleets_owned = [f for f in fleets if f.owner == player]
    other_fleets = [f for f in fleets if f.owner != player]
    
    return player, planets, fleets_owned, other_fleets

# @functools.lru_cache(maxsize=1024)
def cal_fleet_speed(num_of_ships):
    # if num_of_ships in fleet_speed:
    #     return fleet_speed[num_of_ships]
    
    ratio = math.log(num_of_ships) / math.log(1000)
    speed = 1.0 + (6.0 - 1.0) * ratio ** 1.5
    # fleet_speed[num_of_ships] = speed
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


def get_moves(target, reserved_targets, mine, angular_velocity, comet_planet_ids, planets, launch_perc=None):
    # reserved_targets - target with launched fleet
    if target.id in comet_planet_ids:
        return None
    
    if target.id in reserved_targets:
        return None

    dx = target.x - mine.x
    dy = target.y - mine.y
        
    result = get_angle(dx, dy, target, angular_velocity, mine, launch_perc)
    if result == None: return None
    
    angle, ships_needed, tx, ty = result
        
    if mine.ships >= (ships_needed + mine.ships * FLEET_LAUNCH_THRESHOLD):
        move = [mine.id, angle, math.ceil(ships_needed)]
        
        will_hit_sun = collides_with_sun(mine.x, mine.y, tx, ty, angle)
        will_hit_obstacle = search_collisions(mine.x, mine.y, mine.id, target.id, target.x, target.y, angle, planets)
        if not (will_hit_sun or will_hit_obstacle):
            launch_history.append([mine.id, target.id, angle, time])
            reserved_targets.add(target.id)
            return move
    return None

def is_orbiting(planet):
    sun_planet_distance = cal_hypotenus(sun_config[0], sun_config[1], planet.x, planet.y)
    return sun_planet_distance + planet.radius < 50
                
def deploy_comet_ships(times, mine, target):
    if time == times[0] + times[1] - 2:
        dx = target.x - mine.x
        dy = target.y - mine.y
        angle = math.atan2(dy, dx)
        return [mine.id, angle, mine.ships]
    return None
    
def agent(obs):
    global time
    global fleet_target_map
    global comets_expire_time_map
    global last_know_size_map
    
    time += 1
    moves = []
    player, planets, fleets_owned, other_fleets = get_inputs(obs)

    my_planets = {p.id: p for p in planets if p.owner == player}
    temp_last_know_size_map = {my_planets[p].id: my_planets[p].ships for p in my_planets}

    targets = [p for p in planets if p.owner != player]
    comet_planet_ids = obs.get("comet_planet_ids", [])
    
    planet_ids = []
    if time_past and time > time_past[0] and not comets_expire_time_map:
        comets = obs.get("comets", [])
        
        if comets:
            paths = comets[0]["paths"]
            planet_ids = comets[0]["planet_ids"]
            
            comets_expire_time_map = {planet_ids[i]: (time, len(c)) for i, c in enumerate(paths)}
        else: comets_expire_time_map = {}

    if not targets:
        return []
    
    for f in other_fleets:
        for id in my_planets:
            t = my_planets[id]
            
            # check if a planet goes below a retention threshold
            # IDEA: create a map of my planets with current ships
            # retention = min_retention(last_know_size_map[t.id], t.production) if t.id in last_know_size_map else None
            # if retention and t.ships <= retention:
            #     targets.append(t)
            #     continue
            
            if is_orbiting(t):
                result = cal_intercept_2(t, obs.angular_velocity, f.x, f.y)
                if result:
                    _, _, _, future_defense = result
                else:
                    continue
            else:
                dist = cal_hypotenus(f.x, f.y, t.x, t.y)
                eta = dist / cal_fleet_speed(f.ships)
                future_defense = t.ships + (t.production * eta)
            
            if f.ships > future_defense:
                targets.append(t)
    
    new_fleet_target_map = {}
    for fid, tid in fleet_target_map.items():
        t = next((p for p in planets if p.id == tid), None)
        if not t : continue
        
        f = next((f for f in fleets_owned if f.id == fid), None)
        if not f: continue
        
        attack_f = next((f for f in other_fleets if f.id == fid), None)
        if not f: continue
        
        if t.owner == player or t.owner == -1:
            if attack_f and f.ships > attack_f.ships:
                new_fleet_target_map[fid] = tid
            elif not attack_f:
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
    
    if comets_expire_time_map:
        for id in comets_expire_time_map:
            if id not in my_planets: continue
            
            mine = my_planets[id]
            target = sorted(planets, key=lambda t: cal_hypotenus(mine.x, mine.y, t.x, t.y))[0]
            move = deploy_comet_ships(comets_expire_time_map[id], mine, target)
            if move and not next((m for m in moves if move[0] == m[0]), None):
                if target.id in temp_last_know_size_map:
                    log(f"{time} [comet_targets]: Reinforced {target.id} with {move[2]} ships {target.ships}")
                    temp_last_know_size_map[target.id] += move[2]
                    
                log(f"{time} [comet_targets]: Planet {id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
                moves.append(move)

    if len(my_planets) > len(targets):
        for target in targets:
            # get sorted list of nearest planets owned to launch from
            next_nearest_list = sorted(my_planets, key=lambda id: cal_hypotenus(my_planets[id].x, my_planets[id].y, target.x, target.y))
            for id in next_nearest_list:
                move = get_moves(target, reserved_targets, my_planets[id], obs.angular_velocity, comet_planet_ids, planets)

                if not move and target.production >= 4 and target.owner == -1:
                    break

                if not (move == None or next((m for m in moves if move[0] == m[0]), None)):
                    log(f"{time} [my_group]: Planet {id} sending {move[2]} to capture target -> ID: {target.id}  production {target.production} Planet Ships {target.ships} Owner {target.owner}")
                    if target.id in temp_last_know_size_map:
                        log(f"{time} [my_group]: Reinforced {target.id} with {move[2]} ships {target.ships}")
                        temp_last_know_size_map[target.id] += move[2]
                        
                    moves.append(move)
    else:
        for id in my_planets:
            ships = my_planets[id].ships
            
            # Find nearest 3 planets we don't own
            targets_count = map_total_targets(len(planets))
            next_nearest_list = sorted(targets, key=lambda t: (cal_hypotenus(my_planets[id].x, my_planets[id].y, t.x, t.y), -t.production))[:targets_count]
            for i, target in enumerate(next_nearest_list):
                perc = None
                large_planet_found = False
                
                if i < 5 and target.production >= 4:
                    large_planet_found = True
                    move = get_moves(target, reserved_targets, my_planets[id], obs.angular_velocity, comet_planet_ids, planets)

                    if not (move == None or next((m for m in moves if move[0] == m[0]), None)):
                        log(f"{time} [nearest_largest]: Planet {id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
                        ships -= move[2]
                        moves.append(move)
                        continue
                    
                # if len(my_planets) > 1 and len(planets) == 20 and large_planet_found and not move:
                #     break
                            
                if ships <= 0: break
                
                move = get_moves(target, reserved_targets, my_planets[id], obs.angular_velocity, comet_planet_ids, planets, perc)
                if not (move == None or next((m for m in moves if move[0] == m[0]), None)):
                    log(f"{time} [target_groups]: Planet {id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
                    if target.id in temp_last_know_size_map:
                        log(f"{time} [target_groups]: Reinforced {target.id} with {move[2]} ships {target.ships}")
                        temp_last_know_size_map[target.id] += move[2]

                    ships -= move[2]
                    moves.append(move)

    log(f"{time} [overview]: Moves {len(moves)}")
    last_know_size_map = temp_last_know_size_map
    return moves