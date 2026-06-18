from kaggle_environments import make
import sys

def get_ships(obs, player):
    # obs can be a dict or an object depending on version
    planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    p_ships = sum(p[5] for p in planets if p[1] == player)
    f_ships = sum(f[6] for f in fleets if f[1] == player)
    return p_ships + f_ships

def run_test():
    env = make("orbit_wars", debug=True)
    
    print("Running Orbit Wars: submission_2 vs base_submission")
    env.run(["submission_2.py", "base_submission.py"])
    
    for i, state in enumerate(env.steps):
        if i % 100 == 0 or i == len(env.steps) - 1:
            obs = state[0].observation
            print(f"Step {i:3}: P0={get_ships(obs, 0):5}, P1={get_ships(obs, 1):5}")
            
    # Check final status
    for i, state in enumerate(env.state):
        if state.status == "ERROR":
            print(f"Agent {i} ERROR: {state.observation.get('info', 'No info')}")

if __name__ == "__main__":
    run_test()
