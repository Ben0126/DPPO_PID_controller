import numpy as np
from dppo_pid_env import make_env
import yaml

def verify_reward():
    print("Verifying Gaussian Reward Implementation...")
    
    # Load config to check if reward type is set correctly
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Reward Type in config: {config['reward']['reward_type']}")
    if config['reward']['reward_type'] != 'gaussian':
        print("ERROR: Reward type is not 'gaussian'!")
        return

    # Create environment
    env = make_env("config.yaml")
    obs, info = env.reset(seed=42)
    
    print("Environment created successfully.")
    print(f"Initial Observation: {obs}")
    
    # Take a step with zero action (PID gains = 0)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step 1 Reward: {reward}")
    print(f"Info: {info}")
    
    # Check if reward is within expected range (0 to 1 + bonus) * n_inner_steps
    # n_inner_steps is 10, so max reward per step is roughly (1 + 0.1) * 10 = 11
    # But since error is likely non-zero, it should be positive but less than max.
    
    if reward > 0:
        print("SUCCESS: Reward is positive as expected for Gaussian reward.")
    else:
        print("WARNING: Reward is not positive. Check calculation.")

    # Take a step with some action
    action = np.array([5.0, 0.1, 0.2], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step 2 Reward (with action): {reward}")

    env.close()

if __name__ == "__main__":
    verify_reward()
