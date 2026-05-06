import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, COMET_SPAWN_STEPS

sun_config = (50.0, 50.0, 10.0)
time = 0
combat_planet_ids = []

def find_planet(planets, id):
    for p in planets:
        if p.id == id:
            return p
    return None

def is_fleet_in_transit(combat_planet_id):
    return combat_planet_id in [id[1] for id in combat_planet_ids]

def cal_intercept(planet, angular_v, launcher_pos, time, fleet_speed):
    dx = planet.x - sun_config[0]
    dy = planet.y - sun_config[1]

    current_angle = math.atan2(dy, dx)
    planet_r = math.sqrt(dx**2 + dy**2)
    t = time

    for _ in range(20):
        target_angle = current_angle + (angular_v * t) # calculate the next angle given the time, t
        # cal position from the new angle
        tx = sun_config[0] + (planet_r * math.cos(target_angle))
        ty = sun_config[1] + (planet_r * math.sin(target_angle))

        # cal the distance and time the planet would travel
        dist = math.sqrt((tx - launcher_pos.x)**2 + (ty - launcher_pos.y)**2)
        travel_time = dist / fleet_speed

        if abs(t - travel_time) < 0.1: # Threshold for 'close enough'
            break
        t = travel_time

    return tx, ty

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
            break
        elif p.id == nearest.id: # break if we found our planet
            return None

def get_angle(dx, dy, initial_nearest, nearest, angular_velocity, my_planet, min_dist, fleet_speed):
    angle = math.atan2(dy, dx)
    tx, ty = nearest.x, nearest.y
    if initial_nearest != None:
        tx, ty = cal_intercept(nearest, angular_velocity, my_planet, min_dist/fleet_speed, fleet_speed)
        angle = math.atan2(ty - my_planet.y, tx - my_planet.x)
    return angle, tx, ty

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

    return my_planets, targets, initial_targets, fleets_owned, comet_planet_ids

def cal_fleet_speed(num_of_ships):
    return 1.0 + (6.0 - 1.0) * (math.log(num_of_ships) / math.log(1000)) ** 1.5

def agent(obs):
    global time
    time += 1

    moves = []
    my_planets, targets, initial_targets, fleets_owned, comet_planet_ids = get_inputs(obs)

    to_planet_ids = [id[0] for id in combat_planet_ids]
    from_planet_ids = [id[1] for id in combat_planet_ids]
    fleet_from_planet_ids = [f.from_planet_id for f in fleets_owned]

    if not targets:
        return []

    for mine in my_planets:
        # Find nearest 3 planets we don't own
        next_nearest_list = sorted(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))[:3]
        for nearest in next_nearest_list:
            if nearest.id in to_planet_ids and mine.id in from_planet_ids and not mine.id in fleet_from_planet_ids:
                index = None
                for i in range(len(combat_planet_ids)):
                    combat_id = combat_planet_ids[i]
                    if combat_id[0] == nearest.id and combat_id[1] == mine.id:
                        index = i
                if index != None:
                    combat_planet_ids.pop(index)

            if nearest.id in comet_planet_ids:
                continue

            ships_needed = nearest.ships + 1
            fleet_speed = cal_fleet_speed(ships_needed)
            if mine.ships >= ships_needed:
                initial_nearest = get_prev_image(initial_targets, nearest)

                min_dist = math.sqrt((mine.x - nearest.x)**2 + (mine.y - nearest.y)**2)

                dx = nearest.x - mine.x
                dy = nearest.y - mine.y

                angle, tx, ty = get_angle(dx, dy, initial_nearest, nearest, obs.angular_velocity, mine, min_dist, fleet_speed)
                if not is_fleet_in_transit(nearest.id):
                    distance_r, dot_product, nearest_dot_product = cal_intercept_with_sun(mine.x, mine.y, tx, ty, angle)

                    move = [mine.id, angle, ships_needed]
                    if not (dot_product > 0 and distance_r <= sun_config[2]) or nearest_dot_product < 0:
                        moves.append(move)
                        combat_planet_ids.append((nearest.id, mine.id))
    return moves
