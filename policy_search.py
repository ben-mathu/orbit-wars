import math
import random
import numpy as np

# Use the explicitly corrected reference point for orbits
SUN_CENTER = (50, 50) 

def simulate_intercept(planet, launcher_pos, threshold, damping):
    """
    Runs your cal_intercept logic using the parameterized threshold and damping.
    Returns the final distance between the fleet and the planet at the integer arrival turn.
    """
    # 1. Initial State Setup (Angles from Sun's center)
    dx = planet.x - SUN_CENTER[0]
    dy = planet.y - SUN_CENTER[1]
    planet_r = math.hypot(dx, dy)
    initial_angle = math.atan2(dy, dx)
    
    initial_dist = math.hypot(launcher_pos.x - planet.x, launcher_pos.y - planet.y)
    t = initial_dist / 6.0 # Assume max speed for baseline estimation
    
    # 2. The Parameterized Search Loop
    for _ in range(50):
        target_angle = initial_angle + (planet.angular_v * t)
        tx = SUN_CENTER[0] + (planet_r * math.cos(target_angle))
        ty = SUN_CENTER[1] + (planet_r * math.sin(target_angle))
        
        dist = math.hypot(launcher_pos.x - tx, launcher_pos.y - ty)
        
        # Mocking ships needed for the simulation
        ships_needed = (t * planet.production) + planet.ships + 1
        speed = min(6.0, 1.0 + 5.0 * ((math.log(max(1.1, ships_needed)) / math.log(1000)) ** 1.5))
        
        travel_time = dist / speed
        
        # Policy Parameter 1: The Threshold
        if abs(t - travel_time) < threshold:
            break
            
        # Policy Parameter 2: The Damping Factor
        t = (t * (1 - damping)) + (travel_time * damping)
    
    # 3. Game Engine Simulation (The "Truth")
    # Engines operate on discrete integers, so we ceil the arrival time.
    arrival_turn = math.ceil(t) 
    
    actual_angle = initial_angle + (planet.angular_v * arrival_turn)
    actual_tx = SUN_CENTER[0] + (planet_r * math.cos(actual_angle))
    actual_ty = SUN_CENTER[1] + (planet_r * math.sin(actual_angle))
    
    fleet_travel_dist = speed * arrival_turn
    
    # Return the error: How far is the fleet from the planet's exact center?
    # (Lower is better. 0.0 is a perfect strike).
    return abs(dist - fleet_travel_dist)

class MockEntity:
    def __init__(self, x, y, v=0, p=0, s=0):
        self.x, self.y = x, y
        self.angular_v = v
        self.production = p
        self.ships = s

def evaluate_policy(threshold, damping, episodes=500):
    total_error = 0.0
    
    for _ in range(episodes):
        # Generate random valid scenarios
        planet = MockEntity(
            x=random.uniform(10, 90), 
            y=random.uniform(10, 90),
            v=random.uniform(-0.1, 0.1), # Angular velocity
            p=random.randint(0, 5),
            s=random.randint(10, 100)
        )
        launcher = MockEntity(x=random.uniform(10, 90), y=random.uniform(10, 90))
        
        error = simulate_intercept(planet, launcher, threshold, damping)
        total_error += error
        
    return total_error / episodes # Mean Error

def run_policy_search():
    # Initial Guesses
    best_threshold = 0.1
    best_damping = 0.5
    best_score = float('inf')
    
    print("Starting Policy Search...")
    
    for generation in range(1, 51):
        # Mutate the parameters (add Gaussian noise)
        # Decay the noise over time to "fine-tune" as we get deeper
        noise_scale = max(0.01, 1.0 / generation) 
        
        test_threshold = max(1e-5, best_threshold + random.gauss(0, 0.1 * noise_scale))
        test_damping = max(0.1, min(0.9, best_damping + random.gauss(0, 0.2 * noise_scale)))
        
        # Evaluate fitness
        score = evaluate_policy(test_threshold, test_damping)
        
        # Keep the best policy
        if score < best_score:
            best_score = score
            best_threshold = test_threshold
            best_damping = test_damping
            print(f"Gen {generation} | New Best -> Score: {best_score:.4f} | Threshold: {best_threshold:.5f} | Damping: {best_damping:.3f}")

    print("\n--- Optimal Policy Discovered ---")
    print(f"INTERCEPT_THRESHOLD = {best_threshold:.5f}")
    print(f"DAMPING_FACTOR = {best_damping:.3f}")

# Execute the search
if __name__ == "__main__":
    run_policy_search()