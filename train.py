from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from env import TriplePendulumEnv

if __name__ == "__main__":
    print("Setting up Triple Pendulum Environment (The Robotic Leg)...")
    
    # Clean, single environment
    env = TriplePendulumEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=1000)
    
    check_env(env)
    
    # Larger neural network to handle the 3-joint kinematics
    policy_kwargs = dict(net_arch=[256,256])
    
    model = PPO("MlpPolicy", env, verbose=1, 
                policy_kwargs=policy_kwargs,
                n_steps=2048,          
                batch_size=64,         
                tensorboard_log="./triple_pendulum_tb/")
    
    print("Starting training. This requires immense precision...")
    # 2 Million steps to give it time to coordinate both motors
    model.learn(total_timesteps=300_000, tb_log_name="ppo_triple_balance")
    
    model.save("ppo_triple_pendulum2")
    print("Training finished!")