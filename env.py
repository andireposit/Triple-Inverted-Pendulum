import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time

TRIPLE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".8 .9 .8 1"/>
    
    <body name="base" pos="0 0 1.5">
      <geom type="cylinder" size="0.05 0.1" rgba="0.2 0.2 0.2 1" axisangle="0 1 0 1.57"/>
      
      <body name="calf" pos="0 0 0">
        <joint name="ankle_hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.04" rgba="0 0.3 0.7 1"/>
        
        <body name="thigh" pos="0 0 0.4">
          <joint name="knee_hinge" type="hinge" axis="0 1 0" damping="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.03" rgba="0 0.7 0.3 1"/>
          
          <body name="torso" pos="0 0 0.4">
            <joint name="hip_hinge" type="hinge" axis="0 1 0" damping="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.02" rgba="0.7 0.3 0.3 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint="knee_hinge" name="knee_motor" gear="5"/>
    <motor joint="hip_hinge" name="hip_motor" gear="5"/>
  </actuator>
</mujoco>
"""

class TriplePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(TRIPLE_XML)
        self.data = mujoco.MjData(self.model)
        
        # Action Space: 2 Motors (Knee and Hip)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space: 9 variables -> 3 sets of (cos, sin) + 3 velocities
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        self.viewer = None

    def _get_obs(self):
        # 1. Extract raw RELATIVE angles and velocities from MuJoCo
        theta1, theta2, theta3 = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        vel1, vel2, vel3 = self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]
        
        # 2. Calculate GLOBAL angles
        # The calf is attached to the base, so it is already global
        global_theta1 = theta1 
        # The thigh's global angle is the calf's angle plus the knee's bend
        global_theta2 = theta1 + theta2 
        # The torso's global angle is the sum of all three joints
        global_theta3 = theta1 + theta2 + theta3 
        
        # 3. Calculate GLOBAL velocities
        global_vel1 = vel1
        global_vel2 = vel1 + vel2
        global_vel3 = vel1 + vel2 + vel3
        
        # 4. Convert global angles to continuous trigonometric state space
        obs = np.array([
            np.cos(global_theta1), np.sin(global_theta1),
            np.cos(global_theta2), np.sin(global_theta2),
            np.cos(global_theta3), np.sin(global_theta3),
            global_vel1, global_vel2, global_vel3
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start perfectly upright with microscopic noise
        self.data.qpos[0] += np.random.uniform(-0.1, 0.1)
        #self.data.qpos[1] += np.random.uniform(-0.1, 0.1)
        self.data.qpos[1] += 0.4
        self.data.qpos[2] += -0.8
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Apply the two actions to the two motors
        self.data.ctrl[0] = action[0] # Knee
        self.data.ctrl[1] = action[1] # Hip

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        
        # Extract cosines for the reward function (obs indexes 0, 2, and 4)
        cos_theta1 = obs[0]
        cos_theta2 = obs[2]
        cos_theta3 = obs[4]
        
        # --- REWARD ---
        # Relaxed baseline to 0.5 (60 degrees) for all three joints
        reward = (cos_theta1 - 0.5) + (cos_theta2 - 0.5) + (cos_theta3 - 0.5)
        
        # Tiny penalty to encourage smooth motor usage
        reward -= 0.001 * (action[0]**2 + action[1]**2)

        # --- TERMINATION ---
        # If ANY of the three links drops below 60 degrees, the episode ends
        terminated = bool(
            cos_theta1 < 0.5 or 
            cos_theta2 < 0.5 or
            cos_theta3 < 0.5
        )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)

    def close(self):
        if self.viewer:
            self.viewer.close()