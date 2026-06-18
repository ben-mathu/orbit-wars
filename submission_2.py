import math
import sys
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# --- Constants ---
SUN_POS = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_TURNS = 500

class AgentState:
    def __init__(self):
        self.player = -1
        self.time = 0
        self.angular_v = 0.0
        self.planets = []
        self.my_planets = {}
        self.my_fleets = []
        self.other_fleets = []
        self.comet_ids = []
        self.fleet_target_map = {}
        self.pending_launches = {}

    def update(self, obs):
        step = obs.get("step", 0)
        if step == 0: self.__init__()
        self.time = step
        self.player = obs.get("player", 0)
        self.angular_v = obs.get("angular_velocity", 0.0)
        self.planets = [Planet(*p) for p in obs.get("planets", [])]
        fleets = [Fleet(*f) for f in obs.get("fleets", [])]
        self.my_planets = {p.id: p for p in self.planets if p.owner == self.player}
        self.my_fleets = [f for f in fleets if f.owner == self.player]
        self.other_fleets = [f for f in fleets if f.owner != self.player]
        self.comet_ids = obs.get("comet_planet_ids", [])
        self._sync_fleets()

    def _sync_fleets(self):
        new_map = {}
        sorted_fleets = sorted(self.my_fleets, key=lambda f: f.id)
        for f in sorted_fleets:
            if f.id in self.fleet_target_map:
                new_map[f.id] = self.fleet_target_map[f.id]
                continue
            if f.from_planet_id in self.pending_launches and self.pending_launches[f.from_planet_id]:
                target_id, turn, ships = self.pending_launches[f.from_planet_id].pop(0)
                new_map[f.id] = target_id
        for pid in list(self.pending_launches.keys()):
            self.pending_launches[pid] = [l for l in self.pending_launches[pid] if self.time - l[1] < 15]
        self.fleet_target_map = new_map

_as = AgentState()

# --- Helpers ---

def get_fleet_speed(ships):
    s = max(1, ships)
    ratio = math.log(s) / math.log(1000)
    return 1.0 + (6.0 - 1.0) * max(0, ratio) ** 1.5

def get_dist(p1_pos, p2_pos):
    return math.hypot(p1_pos[0]-p2_pos[0], p1_pos[1]-p2_pos[1])

def is_orbiting(planet):
    return get_dist((planet.x, planet.y), SUN_POS) + planet.radius < 50

def predict_planet_pos(planet, t):
    if not is_orbiting(planet): return planet.x, planet.y
    angle = math.atan2(planet.y - SUN_POS[1], planet.x - SUN_POS[0]) + (_as.angular_v * t)
    r = get_dist((planet.x, planet.y), SUN_POS)
    return SUN_POS[0] + r * math.cos(angle), SUN_POS[1] + r * math.sin(angle)

def find_intercept(source, target, max_ships_available):
    for t in range(1, 400):
        tx, ty = predict_planet_pos(target, t)
        dist = get_dist((source.x, source.y), (tx, ty))
        if target.owner == -1: min_needed = target.ships + 1
        else: min_needed = target.ships + (target.production * t) + 1
        if min_needed > max_ships_available: continue
        required_speed = dist / t
        if get_fleet_speed(max_ships_available) >= required_speed:
            if required_speed <= 1.0: s_for_speed = 1
            else:
                try:
                    log_s = math.log(1000) * ((required_speed - 1.0) / 5.0) ** (1.0 / 1.5)
                    s_for_speed = math.ceil(math.exp(log_s))
                except: s_for_speed = max_ships_available
            final_ships = max(min_needed, s_for_speed)
            if final_ships <= max_ships_available:
                return math.atan2(ty - source.y, tx - source.x), int(final_ships), t
    return None

def project_target_state(target):
    owner, ships, prod = target.owner, target.ships, target.production
    approaching = []
    for f in _as.my_fleets:
        if _as.fleet_target_map.get(f.id) == target.id:
            approaching.append((max(1, math.ceil(get_dist((f.x,f.y),(target.x,target.y))/get_fleet_speed(f.ships))), f.ships, _as.player))
    for pid, launches in _as.pending_launches.items():
        for tid, turn, l_ships in launches:
            if tid == target.id:
                src = _as.my_planets.get(pid)
                if src: approaching.append((max(1, math.ceil(get_dist((src.x,src.y),(target.x,target.y))/get_fleet_speed(l_ships))), l_ships, _as.player))
    for f in _as.other_fleets:
        dx, dy = target.x-f.x, target.y-f.y
        dist = math.hypot(dx, dy)
        if dist < 0.1: continue
        t_ang = math.atan2(dy, dx)
        if abs((f.angle - t_ang + math.pi) % (2*math.pi) - math.pi) < 0.06:
            approaching.append((max(1, math.ceil(dist/get_fleet_speed(f.ships))), f.ships, f.owner))
    approaching.sort()
    curr_t = 0
    for eta, f_ships, f_owner in approaching:
        dt = eta - curr_t
        if owner != -1: ships += prod * dt
        if f_owner == owner: ships += f_ships
        else:
            if f_ships > ships: ships, owner = f_ships - ships, f_owner
            else: ships -= f_ships
        curr_t = eta
    return owner, ships

def check_path_clear(source, target_pos, target_id):
    s, e = (source.x, source.y), target_pos
    dx, dy = e[0]-s[0], e[1]-s[1]
    L = math.hypot(dx, dy)
    if L < 0.1: return True
    ux, uy = dx/L, dy/L
    ox, oy = SUN_POS[0]-s[0], SUN_POS[1]-s[1]
    proj = ox*ux + oy*uy
    d_sun = math.hypot(ox, oy) if proj < 0 else (math.hypot(e[0]-SUN_POS[0], e[1]-SUN_POS[1]) if proj > L else abs(ox*uy - oy*ux))
    if d_sun < SUN_RADIUS + 0.05: return False
    for p in _as.planets:
        if p.id == source.id or p.id == target_id: continue
        ox, oy = p.x-s[0], p.y-s[1]
        proj = ox*ux + oy*uy
        d_p = math.hypot(ox, oy) if proj < 0 else (math.hypot(e[0]-p.x, e[1]-p.y) if proj > L else abs(ox*uy - oy*ux))
        if d_p < p.radius: return False
    return True

# --- Main Agent ---

def agent(obs):
    _as.update(obs)
    moves = []
    
    is_very_early = _as.time < 50
    is_early = _as.time < 120
    is_desperate = len(_as.my_planets) <= 1
    
    if _as.time >= MAX_TURNS - 30:
        rem = MAX_TURNS - _as.time
        for pid, my_p in _as.my_planets.items():
            cur_s = my_p.ships
            if cur_s <= 5: continue
            others = [p for p in _as.planets if p.id != pid and p.owner != _as.player]
            if not others: continue
            others.sort(key=lambda p: get_dist((my_p.x, my_p.y), (p.x, p.y)))
            for target in others:
                if cur_s <= 5: break
                intrcp = find_intercept(my_p, target, cur_s - 5)
                if intrcp:
                    ang, shp, t = intrcp
                    if t <= rem:
                        amt = cur_s - 5
                        moves.append([my_p.id, ang, int(amt)])
                        cur_s -= amt
                        _as.my_planets[pid] = my_p._replace(ships=cur_s)
                        break

    targets = []
    for p in _as.planets:
        p_owner, p_ships = project_target_state(p)
        if p.owner == _as.player:
            def_buf = 10 if is_very_early else (30 if is_early else 50)
            if p_owner != _as.player or p_ships < def_buf:
                targets.append((200000 + (p.production * 300) - p_ships, p))
            continue
        if p_owner == _as.player: continue
        dists = [get_dist((mp.x, mp.y), (p.x, p.y)) for mp in _as.my_planets.values()]
        min_d = min(dists) if dists else 100
        type_m = 300 if p.owner == -1 else 30
        comet_m = 10 if p.id in _as.comet_ids else 1
        targets.append(((p.production * 200 * type_m * comet_m) / (p.ships + 5) / (min_d / 50 + 1), p))
    targets.sort(key=lambda x: x[0], reverse=True)
    
    for score, target in targets:
        sources = []
        for pid, mp in _as.my_planets.items():
            if pid == target.id: continue
            base_ret = 1 if is_very_early else (15 if is_early else 40)
            ret = 1 if is_desperate else max(base_ret, mp.ships * 0.1)
            if mp.ships > ret: sources.append((get_dist((mp.x, mp.y), (target.x, target.y)), mp))
        sources.sort()
        for d, src_obj in sources:
            src = _as.my_planets.get(src_obj.id)
            if not src: continue
            p_owner, p_ships = project_target_state(target)
            target_buffer = 40 if target.owner == _as.player else 10
            if target.owner == _as.player:
                if p_owner == _as.player and p_ships >= target_buffer: break
            elif p_owner == _as.player and p_ships >= target_buffer: break
            base_ret = 1 if is_very_early else (15 if is_early else 40)
            ret = 1 if is_desperate else max(base_ret, src.ships * 0.1)
            can_s = src.ships - ret
            if can_s <= 0: continue
            intrcp = find_intercept(src, target, can_s)
            if intrcp:
                ang, s_needed, t = intrcp
                if _as.time + t >= MAX_TURNS: continue
                if target.owner == _as.player: amt = min(can_s, target_buffer - p_ships)
                else: amt = min(can_s, s_needed + 5)
                amt = min(amt, src.ships - 1)
                if amt <= 0: continue
                tx, ty = predict_planet_pos(target, t)
                if check_path_clear(src, (tx, ty), target.id):
                    amt = int(amt)
                    moves.append([src.id, ang, amt])
                    if src.id not in _as.pending_launches: _as.pending_launches[src.id] = []
                    _as.pending_launches[src.id].append((target.id, _as.time, amt))
                    src = src._replace(ships=src.ships - amt)
                    _as.my_planets[src.id] = src
    return moves
