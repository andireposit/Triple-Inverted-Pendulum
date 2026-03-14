import matplotlib.pyplot as plt
import numpy as np
import mujoco
from stable_baselines3 import PPO
from env import TriplePendulumEnv

if __name__ == "__main__":
    print("Loading Viewer...")
    env = TriplePendulumEnv(render_mode="human")

    print("Loading trained model...")
    model = PPO.load("ppo_triple_pendulum")

    obs, info = env.reset()

    # --- Resolve hinge2 dof index once ---
    knee_hinge_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_hinge")
    knee_dof_adr = env.model.jnt_dofadr[knee_hinge_id]
    hip_hinge_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_hinge")
    hip_dof_adr = env.model.jnt_dofadr[hip_hinge_id]

    # --- DATA ARRAYS ---
    t = []
    policy_cmd = []
    ctrl_signal = []
    actuator_force = []
    knee_joint_torque = []
    hip_joint_torque = []

    thigh_angle =[]
    torso_angle = []
    calf_angle = []

    print("Running simulation and recording data...")

    while True:
        action, _states = model.predict(obs, deterministic=True)

        # Save PPO output (policy command)
        policy_cmd.append(float(action[0]))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Log real MuJoCo signals AFTER stepping ---
        raw_ankle = env.unwrapped.data.qpos[0]
        raw_knee = env.unwrapped.data.qpos[1]
        raw_hip = env.unwrapped.data.qpos[2]
        global_ankle = raw_ankle
        global_knee = raw_knee + raw_ankle
        global_hip = raw_hip + global_knee
        t.append(float(env.data.time))
        ctrl_signal.append(float(env.data.ctrl[0]))
        actuator_force.append(float(env.data.actuator_force[0]))
        knee_joint_torque.append(float(env.data.qfrc_actuator[knee_dof_adr]))
        thigh_angle.append(float(np.degrees(global_knee)))
        hip_joint_torque.append(float(env.data.qfrc_actuator[hip_dof_adr]))
        torso_angle.append(float(np.degrees(global_hip)))
        calf_angle.append(float(np.degrees(global_ankle)))

        if terminated or truncated or env.data.time > 20.0:  # Safety stop after 20 seconds
            print(f"Episode finished at {env.data.time:.2f} seconds.")
            break

    env.close()
    #Plotting
    print("Generating Graph...")

    #plt.figure(figsize=(11,5))
    fig, (plt1, plt2) = plt.subplots(2, 1, figsize=(10  , 8), sharex=True)

    #plt.plot(t, policy_cmd, label="Policy Output (action)", linewidth=3.0)
    #plt.plot(t, actuator_force, label="Actuator Force", linewidth=1.0)
    plt1.plot(t, thigh_angle, label="Thigh Angle (deg)",color = 'green', linewidth=1.5)
    plt1.plot(t, torso_angle, label="Torso Angle (deg)",color = 'blue', linewidth=1.5)
    plt1.plot(t, calf_angle, label="Calf Angle (deg)",color = 'black', linewidth=1.5)
    #plt.plot(t, angle1, label="Hinge2 Angle (rad)", linewidth=1.0)
    plt1.axhline(0,color ='black', linestyle="--", linewidth=1)
    #plt1.set_xlabel("Time (seconds)")
    plt1.set_ylabel("Angle (Degrees)")
    plt1.set_title("Global Joint Angles vs Time(s)")
    plt1.legend()
    plt1.grid(True, alpha=0.3)

    plt2.plot(t, knee_joint_torque, label="Knee Joint Torque (Physics)", color='green', linewidth=1.5)
    plt2.plot(t, hip_joint_torque, label="Hip Joint Torque (Physics)", color='blue', linewidth=1.5)
    plt2.axhline(0, color='black', linestyle="--", linewidth=1)
    plt2.set_xlabel("Time (seconds)")
    plt2.set_ylabel("Torque (Nm)")
    plt2.set_title("Torque Applied to Joints vs Time(s)")
    plt2.legend()
    plt2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
