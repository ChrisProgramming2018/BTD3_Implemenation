import gym
import os
import json
import numpy as np
import time
import argparse
from replay_buffer2 import ReplayBuffer 
from agent import Agent
import sys 

def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path




def main(args):
    with open (args.param, "r") as f:
        param = json.load(f)
    print("use the env {} ".format(param["env_name"]))
    print(param)
    continue_iql = True
    #continue_iql = False
    param["locexp"] = args.locexp
    env = gym.make(param["env_name"])
    if param["env_name"] == "Taxi-v2":
        state_space = env.observation_space.n
        action_space = env.action_space.n
    else:
        state_space = 10000
        action_space = 1
        
    print("State space ", state_space)
    print("Action space ", action_space)
    agent = Agent(state_space, action_space, param)
    lr = 0.7
    agent.eval_policy(use_expert=True)
    agent.eval_policy(random_agent=True)
    # sys.exit()
    if continue_iql:
        print("Continue")
        #agent.create_expert_policy()
        #agent.memory.save_memory("memory")
        agent.invers_q()
    else:    
        agent.train()
        agent.save_q_table()
        agent.invers_q()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="test", type=str)
    arg = parser.parse_args()
    mkdir("", arg.locexp)
    main(arg)
