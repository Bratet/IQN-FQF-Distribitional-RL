
import torch
from agents import FQF_Agent, IQN_Agent
import numpy as np
import torch.optim as optim
import random
import math
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym
import argparse
import wrapper
import multipro



def evaluate(eps, frame, eval_runs):
    """
    Makes an evaluation run with the current epsilon
    """
    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0), eps, eval=True)
            state, reward, done, _ = eval_env.step(action[0].item())
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
        
    writer.add_scalar("Reward", np.mean(reward_batch), frame)


def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1):
    """
    
    Params
    ======

    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100//worker)  # last 100 scores
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state = envs.reset()
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        next_state, reward, done, _ = envs.step(action)
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, writer)
        state = next_state
        score += np.mean(reward)
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            evaluate(eps, frame*worker, eval_runs)
        
        if done.any():
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage100 Score: {:.2f} '.format(i_episode*worker, frame*worker, np.mean(scores_window)))
            i_episode +=1 
            state = envs.reset()
            score = 0              


class IQN_Config:
    def __init__(self):
        self.agent = "iqn"
        self.env = "CartPole-v0"
        self.frames = 30000
        self.eval_every = 1000
        self.eval_runs = 5
        self.seed = 1
        self.munchausen = 0
        self.batch_size = 4
        self.layer_size = 512
        self.n_step = 1
        self.N = 32
        self.memory_size = int(1e5)
        self.entropy_coeff = 0.001
        self.lr = 5e-4
        self.gamma = 0.99
        self.tau = 1e-3
        self.eps_frames = 5000
        self.min_eps = 0.025
        self.worker = 1
        self.info = "iqn_agent"
        self.save_model = 0

class FQF_Config:
    def __init__(self):
        self.agent = "fqf"
        self.env = "CartPole-v0"
        self.frames = 30000
        self.eval_every = 1000
        self.eval_runs = 5
        self.seed = 1
        self.munchausen = 0
        self.batch_size = 4
        self.layer_size = 512
        self.n_step = 1
        self.N = 32
        self.memory_size = int(1e5)
        self.entropy_coeff = 0.001
        self.lr = 5e-4
        self.gamma = 0.99
        self.tau = 1e-3
        self.eps_frames = 5000
        self.min_eps = 0.025
        self.worker = 1
        self.info = "fqf_agent"  # This can be updated as needed
        self.save_model = 0


if __name__ == "__main__":
    iqn_config = IQN_Config()
    
    fqf_config = FQF_Config()
    
    writer = SummaryWriter("runs/"+iqn_config.info)       
    seed = iqn_config.seed
    BUFFER_SIZE = iqn_config.memory_size
    BATCH_SIZE = iqn_config.batch_size
    GAMMA = iqn_config.gamma
    TAU = iqn_config.tau
    LR = iqn_config.lr
    n_step = iqn_config.n_step
    env_name = iqn_config.env
    device = torch.device("cpu")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if "-ram" in iqn_config.env or iqn_config.env == "CartPole-v0" or iqn_config.env == "LunarLander-v2": 
        envs = multipro.SubprocVecEnv([lambda: gym.make(iqn_config.env) for i in range(iqn_config.worker)])
        eval_env = gym.make(iqn_config.env)
    else:
        envs = multipro.SubprocVecEnv([lambda: wrapper.make_env(iqn_config.env) for i in range(iqn_config.worker)])
        eval_env = wrapper.make_env(iqn_config.env)
    envs.seed(seed)
    eval_env.seed(seed+1)


    action_size = eval_env.action_space.n
    state_size = eval_env.observation_space.shape

    agent = IQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        network=iqn_config.agent,
                        munchausen=iqn_config.munchausen,
                        layer_size=iqn_config.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA,  
                        N=iqn_config.N,
                        worker=iqn_config.worker,
                        device=device, 
                        seed=seed)


    eps_fixed = False
    
    print("Training IQN Agent")

    t0 = time.time()
    run(frames = iqn_config.frames//iqn_config.worker, eps_fixed=eps_fixed, eps_frames=iqn_config.eps_frames//iqn_config.worker, min_eps=iqn_config.min_eps, eval_every=iqn_config.eval_every//iqn_config.worker, eval_runs=iqn_config.eval_runs, worker=iqn_config.worker)
    t1 = time.time()
    
    print("\n Training time: {}min".format(round((t1-t0)/60,2)))
    if iqn_config.save_model:
        torch.save(agent.qnetwork_local.state_dict(), iqn_config.info+".pth")


    writer = SummaryWriter("runs/"+fqf_config.info)     
    torch.autograd.set_detect_anomaly(True)
    env_name = fqf_config.env
    seed = fqf_config.seed
    BUFFER_SIZE = fqf_config.memory_size
    BATCH_SIZE = fqf_config.batch_size
    GAMMA = fqf_config.gamma
    TAU = fqf_config.tau
    LR = fqf_config.lr
    n_step = fqf_config.n_step
    device = torch.device("cpu")
    

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if "-ram" in fqf_config.env or fqf_config.env == "CartPole-v0" or fqf_config.env == "LunarLander-v2": 
        envs = multipro.SubprocVecEnv([lambda: gym.make(fqf_config.env) for i in range(fqf_config.worker)])
        eval_env = gym.make(fqf_config.env)
    else:
        envs = multipro.SubprocVecEnv([lambda: wrapper.make_env(fqf_config.env) for i in range(fqf_config.worker)])
        eval_env = wrapper.make_env(fqf_config.env)   
    envs.seed(seed)
    eval_env.seed(seed+1)
    action_size = eval_env.action_space.n
    state_size = eval_env.observation_space.shape

    agent = FQF_Agent(state_size=state_size,    
                        action_size=action_size,
                        network=fqf_config.agent,
                        layer_size=fqf_config.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA,
                        Munchausen=fqf_config.munchausen,
                        N=fqf_config.N,
                        entropy_coeff=fqf_config.entropy_coeff,
                        worker=fqf_config.worker,
                        device=device, 
                        seed=seed)


    eps_fixed = False
    
    print("Training FQF Agent")

    t0 = time.time()
    run(frames = fqf_config.frames//fqf_config.worker, eps_fixed=eps_fixed, eps_frames=fqf_config.eps_frames//fqf_config.worker, min_eps=fqf_config.min_eps, eval_every=fqf_config.eval_every//fqf_config.worker, eval_runs=fqf_config.eval_runs, worker=fqf_config.worker)
    t1 = time.time()

    print("\nTraining time: {}min".format(round((t1-t0)/60,2)))
    if fqf_config.save_model:
        torch.save(agent.qnetwork_local.state_dict(), fqf_config.info+".pth")
