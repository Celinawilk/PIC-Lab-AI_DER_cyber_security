from powergym.env_register import make_env
from stable_baselines3 import PPO

models_dir = "models"

env = make_env("13Bus_PV_Attack", worker_idx=None)
env.reset()

model_path = f"{models_dir}/PPO Seed 3 VVC Normal.zip"
model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        print(rewards)
        print(env.obs['bus_voltages'])
        print(env.obs['true_bus_voltages'])