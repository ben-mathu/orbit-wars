import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS
from itertools import count

INTERCEPT_THRESHOLD = 0.01
FLEET_LAUNCH_THRESHOLD = 0.2

sun_config = (50.0, 50.0, 10.0)
time = 0

fleet_target_map = {} # maps already targeted planets 
planet_ownership_map = {}
launch_history = []

def find_planet(planets, id):
    for p in planets:
        if p.id == id:
            return p
    return None
        
def cal_intercept(planet, angular_v, launcher_pos):
    dx = planet.x - sun_config[0]
    dy = planet.y - sun_config[1]
    
    current_angle = math.atan2(dy, dx)
    planet_r = math.sqrt(dx**2 + dy**2) # distance fromt the sun's center to the target planet
    t = 1
    
    ships_needed = planet.ships + 1
    
    for _ in range(20):
        target_angle = current_angle + (angular_v * t) # calculate the next angle given the time, t
        # cal position from the new angle
        tx = sun_config[0] + (planet_r * math.cos(target_angle))
        ty = sun_config[1] + (planet_r * math.sin(target_angle))
        
        # cal the distance and time the planet would travel
        dist = math.sqrt((tx - launcher_pos.x)**2 + (ty - launcher_pos.y)**2)
        
        if planet.owner > -1:
            ships_needed = (t * planet.production) + planet.ships + 1
            
        travel_time = dist / cal_fleet_speed(ships_needed)
        
        if abs(t - travel_time) < INTERCEPT_THRESHOLD: # Threshold for 'close enough'
            break
        t = travel_time
    
    return tx, ty, ships_needed

def cal_intercept_with_sun(start_x, start_y, nearest_x, nearest_y, angle):
    dx = sun_config[0] - start_x
    dy = sun_config[1] - start_y
    
    nx = sun_config[0] - nearest_x
    ny = sun_config[1] - nearest_y
    
    miss_distance = abs(dx * math.sin(angle) - dy * math.cos(angle))
    dot_product = dx * math.cos(angle) + dy * math.sin(angle)
    
    nearest_angle = math.atan2(ny, nx)
    nearest_dot_product = nx * math.cos(nearest_angle) + ny * math.sin(nearest_angle)
    return miss_distance, dot_product, nearest_dot_product

def get_prev_image(initial_targets, nearest):
    for p in initial_targets:
        if p.id == nearest.id and nearest.x != p.x: # break if we found our planet and its revolving
            return p
        elif p.id == nearest.id: # break if we found our planet
            return None
        
def get_angle(dx, dy, initial_target, target, angular_velocity, my_planet):
    angle = math.atan2(dy, dx)
    tx, ty, ships_needed = target.x, target.y, target.ships + 1
    if initial_target == None and target.owner > -1:
        # production, distance, current number of ships
        distance = cal_hypotenus(my_planet.x, my_planet.y, target.x, target.y)
        for t in range(1, 20):
            num_ships = (t * target.production) + target.ships
            speed = cal_fleet_speed(num_ships)
            travel_time = distance / speed
            
            if travel_time <= t:
                ships_needed = num_ships
                break
    elif initial_target != None:
        tx, ty, ships_needed = cal_intercept(target, angular_velocity, my_planet)
        angle = math.atan2(ty - my_planet.y, tx - my_planet.x)
    return angle, tx, ty, ships_needed

def get_inputs(obs):
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    initial_planets = [Planet(*p) for p in obs.get("initial_planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    
    comet_planet_ids = obs.get("comet_planet_ids", [])

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    initial_targets = [p for p in initial_planets if p.owner != player]
    
    fleets_owned = [f for f in fleets if f.owner == player]
    
    return player, my_planets, targets, initial_targets, fleets_owned, comet_planet_ids

def cal_fleet_speed(num_of_ships):
    return 1.0 + (6.0 - 1.0) * (math.log(num_of_ships) / math.log(1000)) ** 1.5

def sync_new_fleets(fleet_owned):
    for f in fleet_owned:
        if f.id not in fleet_target_map:
            match = None
            for i, record in enumerate(launch_history):
                origin_id, target_id, intent_angle, turn = record
                
                if f.from_planet_id == origin_id and math.isclose(f.angle, intent_angle, abs_tol=1e-4):
                    match = target_id
                    launch_history.pop(i)
                    break
                
            if match is not None:
                fleet_target_map[f.id] = match
    
    launch_history[:] = [r for r in launch_history if time - r[3] < 3]

def cal_hypotenus(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)

def get_moves(target, reserved_targets, initial_targets, mine, angular_velocity, comet_planet_ids):
    # reserved_targets - target with launched fleet
    if target.id in comet_planet_ids or target.id in reserved_targets:
        return None
    
    initial_target = get_prev_image(initial_targets, target)

    dx = target.x - mine.x
    dy = target.y - mine.y
        
    angle, tx, ty, ships_needed = get_angle(dx, dy, initial_target, target, angular_velocity, mine)
        
    if mine.ships >= ships_needed + (FLEET_LAUNCH_THRESHOLD * mine.ships):
        distance_r, dot_product, nearest_dot_product = cal_intercept_with_sun(mine.x, mine.y, tx, ty, angle)

        move = [mine.id, angle, ships_needed]
        if not (dot_product > 0 and distance_r <= sun_config[2]) or nearest_dot_product < 0:
            launch_history.append([mine.id, target.id, angle, time])
            reserved_targets.add(target.id)
            return move
            
def agent(obs):
    global time
    global fleet_target_map
    time += 1
    
    moves = []
    player, my_planets, targets, initial_targets, fleets_owned, comet_planet_ids = get_inputs(obs)

    if not targets:
        return []
    
    current_fleet_ids = {f.id for f in fleets_owned}
    
    new_fleet_target_map = {}
    for fid, tid in fleet_target_map.items():
        ownership = next((p.owner for p in targets if p.id == tid and p.owner != player and p.owner > -1), None)
        
        ownership_changed = False
        if tid in planet_ownership_map:
            ownership_changed = planet_ownership_map[tid] != ownership
            
        if fid in current_fleet_ids and not ownership_changed:
            new_fleet_target_map[fid] = tid
        planet_ownership_map[tid] = ownership
    fleet_target_map = new_fleet_target_map

    sync_new_fleets(fleets_owned)
    reserved_targets = set(fleet_target_map.values())

    if len(my_planets) > len(targets):
        for target in targets:
            # get sorted list of nearest planets owned to launch from
            next_nearest_list = sorted(my_planets, key=lambda mine: cal_hypotenus(mine.x, mine.y, target.x, target.y))
            for mine in next_nearest_list:
                move = get_moves(target, reserved_targets, initial_targets, mine, obs.angular_velocity, comet_planet_ids)
                if move != None:
                    moves.append(move)
    else:
        for mine in my_planets:
            # Find nearest 3 planets we don't own
            next_nearest_list = sorted(targets, key=lambda t: cal_hypotenus(mine.x, mine.y, t.x, t.y))[:3]
            for target in next_nearest_list:
                move = get_moves(target, reserved_targets, initial_targets, mine, obs.angular_velocity, comet_planet_ids)
                if move != None:
                    moves.append(move)
    return moves