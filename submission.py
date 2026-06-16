import functools
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS

INTERCEPT_THRESHOLD = 0.1
DAMPING_FACTOR = 0.05
GAMMA = 0.99

# retaining planet ship when launching does not help, occupy as much as possible
FLEET_RETENTION_THRESHOLD = 0.1

sun_config = (50.0, 50.0, 10.0)

# target customisation for each game base on number of planets
planets_num = (20, 40)
target_range = (3, 7)

# retention to avoid depletion and prone to attack
planet_production_range = (1, 5)
percentage_range = (0.5, 0.1)

show_logs = False

class AgentState:
    """Encapsulates mutable state for the agent across turns."""
    
    def __init__(self):
        self.player = -1
        self.time = 0
        self.launch_percentage_range = [0.1, 0.3]
        self.planet_owned_range = [1, None]
        
        self.fleet_target_map = {}  # maps already targeted planets
        self.launch_history = []
        
        self.attack_fleet_target_map = {}
        self.comets_expire_time_map = {}
        self.time_past = None

        self.last_know_size_map = {}  # map of owned planets and size from previous turn
        
        self.my_planet_num = 0
        self.split_ships = {}

        self.targets_count_set = False
        self.targets_count = 3
        
        # environment
        self.planets = []
        self.my_fleet = []
        self.other_fleets = []
        self.comet_planet_ids = []
        self.comets = []
        self.unowned_planets = []
        self.angular_v = -1
        self._my_planets = {}

    @property
    def my_planets(self):
        return self._my_planets

    def update_my_planets(self):
        self._my_planets = {p.id: p for p in _as.planets if p.owner == _as.player}

# Global state instance (persists across turns)
_as = AgentState()

## Helper functions

def log(message: str = None):
    """agent logging

    Args:
        message (str): logged message
    """
    if show_logs:
        print(message if message else "")

def interpolate(input, input_range, output_range):
    ratio = (input - input_range[0]) / (input_range[1] - input_range[0])
    scale = output_range[1] - output_range[0]
    return output_range[0] + ratio * scale

def cal_fleet_speed(num_of_ships):
    ratio = math.log(num_of_ships) / math.log(1000)
    speed = 1.0 + (6.0 - 1.0) * ratio ** 1.5
    return speed

def cal_hypotenus(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)

def angle_to(origin_x, origin_y, target_x, target_y):
    return math.atan2(target_y - origin_y, target_x - origin_x)

def angle_from_sun(tx, ty):
    return angle_to(sun_config[0], sun_config[1], tx, ty)

def cal_time(distance, ships):
    return distance / cal_fleet_speed(ships)

# Retention / Sizing

def min_retention_perc(planet_prod):
    return interpolate(planet_prod, planet_production_range, percentage_range)

@functools.lru_cache(maxsize=1024)
def cal_ships_needed(target_ships):
    additional_ships_perc = interpolate(_as.my_planet_num, _as.planet_owned_range, _as.launch_percentage_range)
    more_ships = (additional_ships_perc * target_ships)
    # log(f"Number of planets owned {_agent_state.my_planet_num} target ships {target_ships} additional perc: {additional_ships_perc} More Ships: {more_ships}")
    return math.ceil(more_ships if more_ships > 0 else 1)

# Intercept / Orbit Prediction

def cal_intercept_2(target, origin_x, origin_y):
    more_ships = cal_ships_needed(target.ships)
    # more_ships = 1
    
    current_angle = angle_from_sun(target.x, target.y)
    planet_r = cal_hypotenus(sun_config[0], sun_config[1], target.x, target.y) # distance fromt the sun's center to the target planet

    dist = cal_hypotenus(target.x, target.y, origin_x, origin_y)
    spd = cal_fleet_speed(target.ships + more_ships)
    tx, ty, t = target.x, target.y, cal_time(dist, spd)

    for _ in range(10):
        next_angle = current_angle + (_as.angular_v * t)
        
        if target.owner > -1 and target.owner != _as.player:
            ships_needed = (t * target.production) + target.ships + more_ships
        else:
            if target.owner == _as.player and target.id in _as.last_know_size_map and _as.last_know_size_map[target.id] > target.ships:
                ships_needed = _as.last_know_size_map[target.id] - target.ships
            else:
                ships_needed = target.ships + more_ships

        ships_needed = math.ceil(ships_needed)
                
        tx = sun_config[0] + (planet_r * math.cos(next_angle))
        ty = sun_config[1] + (planet_r * math.sin(next_angle))

        dist = cal_hypotenus(origin_x, origin_y, tx, ty)
                
        # round down because fleet over shot - this fixed 99 % of launches
        # if math.floor(cal_time(dist, ships_needed)) <= t:
        travel_time = cal_time(dist, ships_needed)
        if abs(t - travel_time) < 0.01:
            return tx, ty, ships_needed, travel_time
        t = travel_time
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
    current_angle = angle_from_sun(target.x, target.y)
    planet_r = cal_hypotenus(sun_config[0], sun_config[1], target.x, target.y) # distance fromt the sun's center to the target planet
    
    initial_dist = cal_hypotenus(origin_x, origin_y, target.x, target.y)

    ships_needed = target.ships + 1 # minimum ships neeeded
    t = cal_time(initial_dist, ships_needed) # minimum time
    
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

        travel_time = cal_time(dist, ships_needed)
        
        # cal dot_product and miss distance from fleet/launch pos to target
        # calculating miss distance is unnecessary because projected coordinates is where the target center will be

        if abs(t - travel_time) < threshold: # Threshold for 'close enough'
            arrival_turn = math.ceil(t)
            fleet_travel_dist = ships_needed * arrival_turn
            return abs(dist - fleet_travel_dist), tx, ty, ships_needed
    
        t = travel_time
    return None

# Collision Checks

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

def collides_with_sun(start_x, start_y, target_x, target_y, angle):
    """calculations to check if fleet will hit the sun

    Args:
        start_x (float): launch position x
        start_y (float): launch position y
        target_x (float): target x
        target_y (float): target y
        angle (float): angle of launch to sun

    Returns:
        boolean: if the fleet will hit the sun
    """
    
    distance_to_sun = cal_hypotenus(start_x, start_y, sun_config[0], sun_config[1])
    distance_to_target = cal_hypotenus(start_x, start_y, target_x, target_y)
    will_hit = collides_with_obstacle(start_x, start_y, sun_config[0], sun_config[1], angle, sun_config[2])
    return will_hit and distance_to_target > distance_to_sun

def search_collisions(start_x, start_y, launch_id, target_id, target_x, target_y, angle, planets):
    for p in planets:
        if p.id == target_id or p.id == launch_id: continue
        
        distance_to_target = cal_hypotenus(start_x, start_y, target_x, target_y)
        distance_to_obstacal = cal_hypotenus(start_x, start_y, p.x, p.y)
        
        if collides_with_obstacle(start_x, start_y, p.x, p.y, angle, p.radius) and distance_to_target > distance_to_obstacal:
            return True
    return False

# Angle / move calculations

def predict_arrival(fleet, target):
    if is_orbiting(target):
        result = cal_intercept_2(target, fleet.x, fleet.y)
        if result:
            _, _, _, eta = result
            return eta
    else:
        if collides_with_obstacle(fleet.x, fleet.y, target.x, target.y, fleet.angle, target.radius):
            dist = cal_hypotenus(fleet.x, fleet.y, target.x, target.y)
            return cal_time(dist, fleet.ships)
    return None

def get_angle(target, my_planet):
    
    more_ships = cal_ships_needed(target.ships)
    # more_ships = 1

    angle = angle_to(my_planet.x, my_planet.y, target.x, target.y)
    if is_orbiting(target):
        result = cal_intercept_2(target, my_planet.x, my_planet.y)
        if result != None:
            tx, ty, ships_needed, t = result
            angle = angle_to(my_planet.x, my_planet.y, tx, ty)
            return angle, ships_needed, tx, ty
    elif target.owner == -1 or target.owner == _as.player:
        if target.owner == _as.player and target.id in _as.last_know_size_map and _as.last_know_size_map[target.id] > target.ships:
            ships_needed = _as.last_know_size_map[target.id] - target.ships
        else:
            ships_needed = target.ships + more_ships
        return angle, math.ceil(ships_needed), target.x, target.y
    else:
        # Calculates time to arrive and compares with travel time
        # production, distance, current number of ships
        distance = cal_hypotenus(my_planet.x, my_planet.y, target.x, target.y)
        for t in range(1, 100):
            if target.owner == _as.player and target.id in _as.last_know_size_map and _as.last_know_size_map[target.id] > target.ships:
                ships_needed = _as.last_know_size_map[target.id] - target.ships
            else:
                ships_needed = (t * target.production) + target.ships + more_ships
            
            ships_needed = math.ceil(ships_needed)
            
            travel_time = cal_time(distance, ships_needed)
            
            if travel_time <= t: # check if the time take is optimal, larger number of ship == less time
                return angle, math.ceil(ships_needed), target.x, target.y
    return None

def get_moves(target, reserved_targets, mine):
    # reserved_targets - target with launched fleet
    if target.id in _as.comet_planet_ids:
        return None
    
    if target.id in reserved_targets:
        return None
    
    if target.id == mine.id:
        return None
    
    # retrieve attacking fleet from cache - if target already reinforced don't add it to cache
    attack_fleet_id = next((at for at, v in _as.attack_fleet_target_map.items() if v[0] == mine.id), None)
    if attack_fleet_id:
        return None
        
    result = get_angle(target, mine)
    if not result: return None
    
    angle, ships_needed, tx, ty = result
    
    if mine.ships >= (ships_needed + (mine.ships * _as.launch_percentage_range[0])):
    # if mine.ships >= (ships_needed + (FLEET_RETENTION_THRESHOLD * mine.ships)):
        ships_needed = math.ceil(ships_needed)
        move = [mine.id, angle, ships_needed]
        
        will_hit_sun = collides_with_sun(mine.x, mine.y, tx, ty, angle)
        will_hit_obstacle = search_collisions(mine.x, mine.y, mine.id, target.id, target.x, target.y, angle, _as.planets)
        if not (will_hit_sun or will_hit_obstacle):
            t = cal_time(cal_hypotenus(tx, ty, mine.x, mine.y), ships_needed)
            _as.launch_history.append([mine.id, target.id, angle, _as.time, t])
            reserved_targets.add(target.id)
            
            fleet_id = next((at for at, v in _as.attack_fleet_target_map.items() if v[0] == target.id), None)
            if fleet_id:
                attack_fleet_sum = sum([v[2] for v in _as.attack_fleet_target_map.values() if v[0] == target.id])
                fleet_sum = sum([v[1] for v in _as.fleet_target_map.values() if v[0] == target.id])
                log(f"{_as.time} [attack detection] Target {target.id} attack fleet sum {attack_fleet_sum} vs my fleet sum {fleet_sum}")
                if fleet_sum > attack_fleet_sum:
                    _as.attack_fleet_target_map[fleet_id] = (_as.attack_fleet_target_map[fleet_id][0], True, _as.attack_fleet_target_map[fleet_id][2])
            return move, tx, ty
    return None

# Comets & attack detection

def deploy_comet_ships(times, mine, target):
    if _as.time == times[0] + times[1] - 2:
        angle = angle_to(mine.x, mine.y, target.x, target.y)
        return [mine.id, angle, mine.ships]
    return None

def target_attacked(planet, attack_fleet):
    """Predict if a planet will be attacked and update attack tracking."""
    eta = predict_arrival(attack_fleet, planet)
    if eta:
        eta = math.ceil(eta)
        # Future defense: current ships + production over eta turns
        future_defense = planet.ships + (planet.production * eta) if planet.owner == _as.player else planet.ships
        if attack_fleet.ships > future_defense and attack_fleet.id not in _as.attack_fleet_target_map:
            planet_id = next((fid for fid, v in _as.attack_fleet_target_map.items() if v[0] == planet.id), None)
            if not planet_id:
                _as.attack_fleet_target_map[attack_fleet.id] = (planet.id, False, attack_fleet.ships)
            return eta
        # elif attack_fleet.ships < future_defense and attack_fleet.id not in _as.attack_fleet_target_map:
        #     _as.attack_fleet_target_map[attack_fleet.id] = (planet.id, True, attack_fleet.ships)
    return None

# State Sync & inputs

def get_inputs(obs):
    _as.player = obs.get("player", 0)
    _as.planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    
    _as.my_fleet = [f for f in fleets if f.owner == _as.player]
    _as.other_fleets = [f for f in fleets if f.owner != _as.player]

    _as.comet_planet_ids = obs.get("comet_planet_ids", [])
    _as.comets = obs.get("comets", [])
    
    _as.update_my_planets()
    
    _as.angular_v = obs.angular_velocity

def sync_new_fleets(fleet_owned):
    for f in fleet_owned:
        if f.id not in _as.fleet_target_map:
            for i, record in enumerate(_as.launch_history):
                origin_id, target_id, intent_angle, turn, t = record # t = turn - time in orbit wars 1-500
                
                # check where the fleet came from
                # check if ownership has change (target id is in planet ownership map)
                if f.from_planet_id == origin_id and math.isclose(f.angle, intent_angle, abs_tol=1e-4):
                    _as.fleet_target_map[f.id] = (target_id, f.ships)
                    _as.launch_history.pop(i)
                    break
    
    _as.launch_history[:] = [r for r in _as.launch_history if _as.time - r[3] <= r[4]]

def is_orbiting(planet):
    sun_planet_distance = cal_hypotenus(sun_config[0], sun_config[1], planet.x, planet.y)
    return sun_planet_distance + planet.radius < 50

def expire_comets(temp_last_know_size_map):
    # if agent owns comet, if last time to see commet, launch all ships to nearest planet
    moves = []
    for id in _as.comets_expire_time_map:
        if id not in _as.my_planets: continue
        
        mine = _as.my_planets[id]
        filtered_planets = [p for p in _as.planets if not is_orbiting(p) and p.id != id]
        target = sorted(filtered_planets, key=lambda t: cal_hypotenus(mine.x, mine.y, t.x, t.y))[0]
        move = deploy_comet_ships(_as.comets_expire_time_map[id], mine, target)
        if move:
            if target.id in temp_last_know_size_map:
                log(f"{_as.time} [comet_targets]: Reinforced {target.id} with {move[2]} ships {target.ships}")
                temp_last_know_size_map[target.id] += move[2]
                
            log(f"{_as.time} [comet_targets]: Planet {id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
            moves.append(move)
    return moves

def get_reserved_targets():
    new_fleet_target_map = {}
    for fid, v in _as.fleet_target_map.items():
        tid = v[0]
        
        # retrieve target in cache
        t = next((p for p in _as.planets if p.id == tid), None)
        if not t : continue
        
        # retrieve fleet in cache
        f = next((f for f in _as.my_fleet if f.id == fid), None)
        if not f: continue
        
        # retrieve attacking fleet from cache - if target already reinforced don't add it to cache
        attack_fleet_id = next((at for at, v in _as.attack_fleet_target_map.items() if v[0] == t.id), None)
        if attack_fleet_id and not _as.attack_fleet_target_map[attack_fleet_id][1]:
            continue
        
        # if not owned and target not being attacked add target to cache
        # else if target not attacked, calculate what is needed to defend
        #   if can defend add to reinforced fleet
        if t.owner == -1:
            new_fleet_target_map[fid] = (tid, f.ships)
        else:
            eta = predict_arrival(f, t)
            if eta:
                eta = math.ceil(eta)
                # If it's our own planet, it's a reinforcement, so it's always "good" to keep it
                # If it's not ours, check if we still win
                future_defense = t.ships + (t.production * eta) if t.owner != _as.player else 0
                if t.owner == _as.player or f.ships > future_defense:
                    new_fleet_target_map[fid] = (tid, f.ships)
    
    # update target map cache
    _as.fleet_target_map = new_fleet_target_map
    sync_new_fleets(_as.my_fleet)
    log(f"{_as.time} [start]: Fleet History {_as.launch_history}")
    return set([v[0] for v in _as.fleet_target_map.values()])

def reinforcement_achieved(target_id, temp_last_know_size_map, ships, ships_needed):
    # logs for reinforcement
    if target_id in temp_last_know_size_map:
        log(f"{_as.time} [my_group]: Reinforced {target_id} with {ships_needed} ships {ships}")
        temp_last_know_size_map[target_id] += ships_needed
    return temp_last_know_size_map

def search_attacking_fleets(ts: list):
    targets = []
    for f in _as.other_fleets:
        for p_id in _as.my_planets:
            my_p = _as.my_planets[p_id]
            result = target_attacked(my_p, f)
            if result:
                eta = result
                planet_ships_eta = (eta * my_p.production) + my_p.ships
                deficit = f.ships - planet_ships_eta
                deficit = (eta * my_p.production) + deficit
                my_p = my_p._replace(ships = deficit)
                targets.append(my_p)
        
        # check neutral planets being attacked
        for p in _as.unowned_planets:
            result = target_attacked(p, f)
            if result:
                eta = result
                deficit = f.ships - p.ships
                deficit = (eta * p.production) + deficit
                p = p._replace(ships = deficit)
                target = next((t for t in ts if t.id == p.id), None)
                if target:
                    ts.remove(target)
                    targets.append(p)
                
                fleet_id = next((fid for fid, v in _as.fleet_target_map.items() if v[0] == p.id), None)
                if fleet_id:
                    _as.fleet_target_map.pop(fleet_id)
                
                index = next((i for i, r in enumerate(_as.launch_history) if r[1] == p.id), None)
                if index:
                    _as.launch_history.pop(index)
    targets += ts
    return targets

def search_owned_by_target(moves, reserved_targets, temp_last_know_size_map, targets):
    for target in targets:
        # get sorted list of nearest owned planets to launch from to avoid long distance travel
        next_nearest_list = sorted(_as.my_planets, key=lambda id: cal_hypotenus(_as.my_planets[id].x, _as.my_planets[id].y, target.x, target.y))
        for id in next_nearest_list:
            # if planet is being attacked skip it
            if is_fleet_attacking(id):
                continue
            
            result = get_moves(target, reserved_targets, _as.my_planets[id])
            
            move = []
            if result:
                move, _, _ = result

                if not (move == None or next((m for m in moves if move[0] == m[0]), None)):                
                    log(f"{_as.time} [my_group]: Planet {id} sending {move[2]} to capture target -> ID: {target.id}  production {target.production} Planet Ships {target.ships} Owner {target.owner}")
                    moves.append(move)

                    temp_last_know_size_map = reinforcement_achieved(target.id, temp_last_know_size_map, target.ships, move[2])
    return moves, temp_last_know_size_map

def search_target_by_owned_planet(moves, my_planet, reserved_targets, temp_last_know_size_map, ships, targets):
    next_nearest_list = sorted(targets, key=lambda t: (cal_hypotenus(my_planet.x, my_planet.y, t.x, t.y), -t.production))[:_as.targets_count]
    for i, target in enumerate(next_nearest_list):
        if ships <= 0: break
        my_planet = my_planet._replace(ships = ships)
        
        # section II - target other nearest planets including largest
        result = get_moves(target, reserved_targets, my_planet)
        
        if result:
            move, _, _ = result
            
            if not (move == None or next((m for m in moves if move[0] == m[0]), None)):            
                log(f"{_as.time} [target_groups]: Planet {my_planet.id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
                ships -= move[2]
                moves.append(move)

                # logs for reinforcement
                temp_last_know_size_map = reinforcement_achieved(target.id, temp_last_know_size_map, target.ships, move[2])
    return moves, temp_last_know_size_map, ships, my_planet

def initialize_agent():
    # set planet_owned_range upper bound
    if not _as.planet_owned_range[1]:
        _as.planet_owned_range[1] = len(_as.planets)

    # set target count for each game
    if not _as.targets_count_set:
        _as.targets_count = math.floor(interpolate(len(_as.planets), planets_num, target_range))
        _as.targets_count_set = True

        log(f"{_as.time} [start]: Number of Planets {len(_as.planets)} Target count {_as.targets_count}")

    temp_my_planet_num = len(_as.my_planets)

    # if _as.time % 3 == 0:
    #     if temp_my_planet_num - _as.my_planet_num > 0:
    #         if _as.launch_percentage_range[0] < 0.5:
    #             _as.launch_percentage_range[0] += 0.01
    #         if _as.launch_percentage_range[1] <= 0.99:
    #             _as.launch_percentage_range[1] += 0.1
    #     elif temp_my_planet_num - _as.my_planet_num < 0 and _as.launch_percentage_range[1] > 0.5:
    #             if _as.launch_percentage_range[0] > 0.1:
    #                 _as.launch_percentage_range[0] -= 0.05
    #             if _as.launch_percentage_range[1] >= 0.3:
    #                 _as.launch_percentage_range[1] -= 0.05
    log(f"{_as.time} [start]: Increamenting Configs: {_as.launch_percentage_range}")
    log(f"{_as.time} [start]: Previous Planets: {_as.my_planet_num} Number of planets: {temp_my_planet_num} launch perc bound: {_as.launch_percentage_range}")
    _as.my_planet_num = temp_my_planet_num

    if _as.comet_planet_ids and not _as.time_past:
        _as.time_past = _as.time

    # time between entering and leaving space
    planet_ids = []
    if _as.time_past and _as.time > _as.time_past and not _as.comets_expire_time_map:
        if _as.comets:
            paths = _as.comets[0]["paths"]
            planet_ids = _as.comets[0]["planet_ids"]
            
            _as.comets_expire_time_map = {planet_ids[i]: (_as.time, len(c)) for i, c in enumerate(paths)}
        else: 
            _as.comets_expire_time_map = {}
            _as.time_past = None

def is_fleet_attacking(planet_id):
    fleet_id = next((fid for fid, v in _as.attack_fleet_target_map.items() if v[0] == planet_id), None)
    if fleet_id:
        return True
    return False

def target_evaluation(reserveed_targets, mine, targets):
    target_evaluation_map = {}
    for target in targets:
        result = get_moves(target, reserveed_targets, mine)
        
        if result:
            move, tx, ty = result
            t = cal_time(cal_hypotenus(tx, ty, mine.x, mine.y), move[2])
            pv = target.production * (GAMMA ** (_as.time + t) - GAMMA ** 500) / (1.0-GAMMA)
            target_evaluation_map[(mine.id, target.id)] = (pv, move)
    return target_evaluation_map

# Main agent function

def agent(obs):
    _as.time += 1
            
    # initialize game input
    moves = []
    get_inputs(obs)
    
    # current size - compares with previous planet size
    temp_last_know_size_map = {_as.my_planets[p].id: _as.my_planets[p].ships for p in _as.my_planets}
    
    initialize_agent()

    targets = [p for p in _as.planets if p.owner != _as.player]

    for t in targets:
        fleet_id = next((fid for fid, v in _as.fleet_target_map.items() if v[0] == t.id), None)
        if fleet_id:
            _as.unowned_planets.append(t)

    if not targets:
        return moves
    
    # updates attacking fleets cache
    _as.attack_fleet_target_map = {fid: t for fid, t in _as.attack_fleet_target_map.items() if fid in [fleet.id for fleet in _as.other_fleets]}
    log(f"{_as.time} [start]: Attack vectors {_as.attack_fleet_target_map}")
    
    # looks for attacking fleets and their targets
    targets = search_attacking_fleets(targets)
    
    for id, my_p in _as.my_planets.items():
        perc = min_retention_perc(my_p.production)
        planet = next((p for p in targets if p.id == my_p.id), None)
        if not planet and id in _as.last_know_size_map and perc * _as.last_know_size_map[id] >= my_p.ships:
            targets.append(my_p)
    
    # cache fleet deployments
    reserved_targets = get_reserved_targets()
    log(f"{_as.time} [start]: My Fleet {_as.fleet_target_map}")
    
    # gracefully expire comet planets
    moves = expire_comets(temp_last_know_size_map)

    # occupied most planets - allows closest owned planet to launch to target
    if len(_as.my_planets) > len(targets):
        moves, temp_last_know_size_map = search_owned_by_target(moves, reserved_targets, temp_last_know_size_map, targets)
    else: # owned planets are few, launch to any nearest
        for id in _as.my_planets:
            # if planet is being attacked skip it
            if is_fleet_attacking(id):
                continue
            
            mine = _as.my_planets[id]
            ships = mine.ships

            # Find nearest {n} planets we don't own, optimize for largest planet first
            # n - determined by the interpolation of current game's number of planets and target range
            next_nearest_list = sorted(targets, key=lambda t: (cal_hypotenus(_as.my_planets[id].x, _as.my_planets[id].y, t.x, t.y), -t.production, t.ships))
            next_nearest_list = [t for t in next_nearest_list if t.production >= 3][:2]
            for i, target in enumerate(next_nearest_list):
                if ships <= 0: break
                mine = mine._replace(ships = ships)
                
                # section I - target large planets,
                # if any was found, accumulate planet's ship to attain enough ships to attack
                result = get_moves(target, reserved_targets, _as.my_planets[id])
                
                if result:
                    move, _, _ = result
                    
                    if not (move == None or next((m for m in moves if move[0] == m[0]), None)):
                        log(f"{_as.time} [nearest_largest]: Planet {id} sending {move[2]} to capture target -> ID: {target.id} production {target.production} ships {target.ships} owner {target.owner}")
                        ships -= move[2]
                        moves.append(move)
                        
                        temp_last_know_size_map = reinforcement_achieved(target.id, temp_last_know_size_map, target.ships, move[2])
            
            moves, temp_last_know_size_map, ships, mine = search_target_by_owned_planet(moves, mine, reserved_targets, temp_last_know_size_map, ships, targets)

    # if not moves and _as.time == 0 and (_as.time % 3 == 0 or _as.time % 11 == 0): # if no move available, launch to any nearest to maintain activity and explore
    if not moves and len(_as.my_planets) > 2: # if no move available, launch to any nearest to maintain activity and explore
        for id in _as.my_planets:
            # if planet is being attacked skip it
            if is_fleet_attacking(id):
                continue
            
            # if is_min_rentetion_threshold_reached(id):
            #     continue
            
            mine = _as.my_planets[id]
            ships = mine.ships
            
            # Find nearest {n} planets we don't own, optimize for largest planet first
            # n - determined by the interpolation of current game's number of planets and target range
            moves, temp_last_know_size_map, ships, mine = search_target_by_owned_planet(moves, mine, reserved_targets, temp_last_know_size_map, ships, targets)
    
    log(f"{_as.time} [overview]: Moves {len(moves)}")
    _as.last_know_size_map = temp_last_know_size_map
    log()
    return moves