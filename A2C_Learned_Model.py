from stable_baselines3.common.evaluation import evaluate_policy

from powergym.env_register import make_env
from stable_baselines3 import A2C

models_dir = "models"

env = make_env("13Bus_PV_Attack", worker_idx=None)
env.reset(load_profile_idx=0)
#env.reset()

model_path = f"{models_dir}/A2C Seed 3 VVC Rise.zip"
model = A2C.load(model_path, env=env)

episodes = 1

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes)
#print(mean_reward)
#print(std_reward)

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        #print(rewards)
        print(info['vol_reward'])
        #print(info['violated_nodes'])
        #print('done')
        print(env.obs['bus_voltages'])
        print(env.obs['true_bus_voltages'])

