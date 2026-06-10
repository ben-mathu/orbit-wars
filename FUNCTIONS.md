**Functions Overview — submission.py**

File: [submission.py](submission.py)

Summary: Core agent logic for Orbit Wars — orbit/intercept math, collision checks, move selection, and state sync for fleets.

**Utilities**
- **`log(message: str)`**: Conditional printing when `show_logs` is True.
- **`interpolate(input, input_range, output_range)`**: Linear interpolation between two ranges.
- **`cal_hypotenus(x0, y0, x1, y1)`**: Distance helper (wrapper for `math.hypot`).
- **`cal_fleet_speed(num_of_ships)`**: Speed model for fleets (log-based). Commented caching available.

**Retention / sizing**
- **`min_retention(planet_size, planet_prod)`**: Computes minimum retained ships using `interpolate` and `percentage_range`.
- **`cal_ships_needed(target_ships)`**: Heuristic for additional ships to send based on planetary holdings.

**Intercept / orbit prediction**
- **`cal_intercept(target, angular_v, player, origin_x, origin_y, threshold=..., damping=..., launch_perc=None)`**: Iterative fixed-point solver to estimate intercept coordinates, travel time and required ships; returns (score, tx, ty, ships_needed) or `None`.
- **`cal_intercept_2(target, angular_v, origin_x, origin_y)`**: Alternate discrete-time intercept search; returns (None, tx, ty, ships_needed) or `None` (note: return shape differs from `cal_intercept`).

**Collision checks**
- **`collides_with_obstacle(start_x, start_y, px, py, angle, radius)`**: Geometry check (miss distance and dot product) → boolean collision.
- **`collides_with_sun(start_x, start_y, target_x, target_y, angle)`**: Uses `collides_with_obstacle` to test if a launch would hit the sun (returns boolean; docstring is outdated).
- **`search_collisions(...)`**: Scans other planets for collisions on the projected path.

**Angle / move calculations**
- **`get_angle(dx, dy, target, angular_velocity, my_planet)`**: Decides launch angle and ships for orbiting/neutral/enemy planets; delegates to intercept calculators when needed.
- **`get_moves(target, reserved_targets, mine, angular_velocity, comet_planet_ids, planets)`**: High-level decision to form a `move` (checks retention, collisions, records `launch_history`, updates `reserved_targets`).

**Comet & attack detection**
- **`deploy_comet_ships(times, mine, target)`**: Special-case full-launch when a comet is about to expire.
- **`target_attacked(planet, angular_v, attack_fleet)`**: Predicts future defense level against an incoming fleet; marks `attack_fleet_target_map` when advantageous.

**State syncing & inputs**
- **`get_inputs(obs)`**: Parses `obs` into `player, planets, fleets_owned, other_fleets`.
- **`sync_new_fleets(fleet_owned)`**: Matches observed fleets to prior `launch_history` and updates `fleet_target_map`.
- **`is_orbiting(planet)`**: Boolean check against `sun_config` and `planet.radius`.

**Main**
- **`agent(obs)`**: Orchestrates the turn: updates `time`, synchronizes caches, predicts incoming attacks, handles comet logic, picks moves, and returns the list of `moves`.

**Improvement Areas (high priority)**
- Inconsistent return types and docstrings: `cal_intercept_2` vs `cal_intercept`; `collides_with_sun` docstring wrong. Standardize signatures and docs.
- Percent vs fraction mismatch: `percentage_range` values appear as fractions but `min_retention` divides by 100. Verify intended scale and fix.
- Duplicate constants & orbit logic across `bases/` modules: centralize `sun_config`, thresholds, and `is_orbiting` into a single module.
- Global mutable state: many module-level globals (`time`, `fleet_target_map`, etc.). Encapsulate in a `AgentState` class to improve testability.
- Performance: memoize `cal_fleet_speed` (use `lru_cache`) and avoid repeated expensive geometry calls when possible.
- Testing: expand unit tests for intercept, collision detection, and `get_moves` (only `test_is_orbiting` exists).
- Add types and consistent docstrings for public functions; return explicit types (or use `Optional[...]`).

**Suggested next steps**
- Create `agent_state.py` to centralize constants and state, move duplicated helpers there.
- Add unit tests for `cal_intercept`, `collides_with_obstacle`, and `get_moves`.
- Normalize percent/fraction usage and enable caching for `cal_fleet_speed`.

If you want, I can implement one of the suggested refactors (pick which), or also run tests locally.
