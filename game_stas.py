from jericho import *
from src.env import JerichoEnv
from tqdm import tqdm

def main():
    cum_reward = 0
    env = JerichoEnv(rom_path="envs/jericho-game-suite/zork1.z5", 
                    seed=0, 
                    step_limit=100, 
                    get_valid = False
                    )
    ob, info = env.reset()
    done = False
    step = 0
    reward_step = []
    walkthrough = env.env.get_walkthrough()
    for act in walkthrough:
        ob, reward, done, info = env.step(act)
        step += 1
        if reward:
            reward_step.append(step)
        cum_reward += reward

        print("step" + str(step))
        print(ob)
        print(act)
        print(cum_reward)
        
    print("maxscore" + str(cum_reward))
    print("reward_step" + str(reward_step))
    diffs = [abs(reward_step[i+1] - reward_step[i]) for i in range(len(reward_step) - 1)]
    print(diffs)
    
    
if __name__ == '__main__':
    main()