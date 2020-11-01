import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_utils.dotdic import DotDic
from env.grid_game_flat import GridGame


# reference:
# https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/GridWorld/gridWorld_Q.py


class Agents:
    def __init__(self, world, dim):
        # for training
        # consists of (b x n) agents
        self.actions = [i for i in range(5)]
        self.world = world
        self.info = 1

        # learning params
        self.epsilon = 0.1

        # init table
        self.Q = torch.zeros(dim ** 2, len(self.actions)) + 10
        self.reset_history()

    def choose_action(self):
        s = self.world.get_state(self.info)
        if np.random.uniform(0, 1) <= self.epsilon:
            # explore
            return np.random.choice(self.actions)
        # greedy
        u_vals = self.Q[s]
        u = np.random.choice(np.flatnonzero(u_vals == u_vals.max()))
        return u

    def take_action(self, u):
        self.step_records.actions_taken.append(u)
        r, terminate = self.world.get_reward(u)
        r = r[0, 0].item()
        self.step_records.reward_obtained.append(r)
        # log new state
        s = self.world.get_state(self.info)
        self.step_records.states_visited.append(s)
        self.terminated = bool(terminate[0])
        return r

    def reset_history(self):
        self.step_records = DotDic({
            'states_visited': [],
            'actions_taken': [],
            'reward_obtained': []
        })
        self.terminated = False
        self.world.reset()


class Arena:
    def __init__(self, opt, agents):
        self.a = agents
        self.opt = opt
        # train params
        self.num_episodes = opt.nepisodes
        self.alpha = 0.5
        self.gamma = 0.9
        self.perf_hist = [0 for _ in range(self.num_episodes)]

    def run_episode(self):
        max_steps = 1000
        i = 0
        while not self.a.terminated and i < max_steps:
            u = self.a.choose_action()
            # print(u, end=' -> ')
            r = self.a.take_action(u)
            # print(r)
            i += 1
        return i

    def train(self):
        for e in tqdm(range(self.num_episodes)):
            # generate episode data
            T = self.run_episode()
            self.perf_hist[e] = T
            s_hist = self.a.step_records.states_visited
            u_hist = self.a.step_records.actions_taken
            r_hist = self.a.step_records.reward_obtained
            s_hist.reverse()
            u_hist.reverse()
            r_hist.reverse()
            # all actions at terminal sT = rT
            rT = r_hist[0]
            sT = s_hist[0]
            self.a.Q[sT, :] = rT
            g = rT
            # backpropagate values
            for t in range(len(s_hist[1:])):
                sprime = s_hist[t - 1]
                q_prime = torch.max(self.a.Q[sprime]).item()
                s = s_hist[t]
                u = u_hist[t]
                r = r_hist[t]
                q_i = self.a.Q[s, u]
                q_ii = q_i + self.alpha * (r + self.gamma * q_prime - q_i)
                self.a.Q[s, u] = q_ii
            self.show_table(self.a.Q)
            self.a.reset_history()
        # print(self.a.Q)
        self.show_table(self.a.Q)
        plt.plot(self.perf_hist)
        plt.show()

    def show_table(self, q, vid=False):
        dim = self.opt.world_dim
        if vid:
            os.system('cls' if os.name == 'nt' else "printf '\033c'")
        grid_idx = -1
        for i in range(dim):
            print('+' + ('-' * 9 + '+') * dim)
            out = '| '
            for j in range(dim):
                grid_idx += 1
                data = str(round(torch.mean(q[grid_idx]).item()))
                out += data.ljust(7) + ' | '
            print(out)
        print('+' + ('-' * 9 + '+') * dim)
        print()


opt = DotDic({
    'bs': 8,
    'game_nagents': 1,
    'game_action_space': 5,
    'game_comm_bits': 0,
    'nepisodes': 100,
    'reward_loc': 2,
    'world_dim': 3
})
opt['game_action_space_total'] = 2 ** opt.game_comm_bits + opt.game_action_space


g = GridGame(opt, (opt.world_dim, opt.world_dim))
g.show(vid=False)

a = Agents(g, opt.world_dim)

tr = Arena(opt, a)
tr.train()
