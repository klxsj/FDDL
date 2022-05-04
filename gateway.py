import numpy as np
import torch

from modelsInGateway import AgentDoubleDQN
from cacheEnv import CacheEnv, AVGFCache
from replay import ReplayBuffer
import csv

import os
import socket

import pickle
import struct
import time
import numpy.random as rd
import sys

from copy import deepcopy
import datetime


ENV_CONFIG = {
    "state_size": 9,  # size of state space: state_dim
    "action_size": 2,  # size of action space: action_dim
    "window": [100, 500, 1000],  # window for last 100req, 500req, 1000req
    "bins": np.array([0.1 * x for x in range(10)]),  # quantization of state space
    "gamma_r": 0.9,
    "net_dim": 2 ** 8,  # width of hidden layer
    "freq_discount": 0.9,  # discount for s,m,l freq
}
TRAINING_CONFIG = {
    "max_memo": 2 ** 12,  # capacity of replay buffer
    "batch_size": 2 ** 8,  # num of transitions sampled from replay buffer.
    "repeat_times": 2 ** 0,  # repeatedly update network to keep critic's loss small
    "target_step": 2 ** 11,  # collect target_step, then update network, target_step must be much smaller than len(dataset)
    "gamma": 0.99,  # discount factor of future rewards
    "reward_scale": 2 ** 0,  # an approximate target reward usually be closed to 256
    "if_per": False,  # Prioritized Experience Replay for sparse reward
    "if_on_policy": False,  # whether the method is on-policy or off-policy
}
MSG = {
    0: 'MSG_INIT',
    1: 'MSG_GLO_SYNC',
    2: 'MSG_LOAD_FINISHED',
    3: 'MSG_TRAIN_FINISHED',
    4: 'MSG_NEW_TERM',
}


class Gateway(object):
    '''each gateway have an env, an agent, and a replay buffer'''
    def __init__(self, name, localData):
        self.name = name
        self.local_data = localData
        self.dataset_size = len(self.local_data)
        self.batch_size_for_update_net = TRAINING_CONFIG["batch_size"]
        self.target_step = TRAINING_CONFIG["target_step"]
        self.train_dl = None
        self.total_cache_table = {}
        self.max_size_until_now = 0

        '''init agent'''
        self.agent = AgentDoubleDQN()
        self.agent.init(net_dim=ENV_CONFIG["net_dim"], state_dim=ENV_CONFIG["state_size"], action_dim=ENV_CONFIG["action_size"])

        '''init cache environment'''
        self.env = CacheEnv(ENV_CONFIG)
        self.env.reset()

        '''init replay buffer in double dqn'''
        self.buffer = ReplayBuffer(max_len=TRAINING_CONFIG["max_memo"], state_dim=ENV_CONFIG["state_size"], action_dim=1, if_on_policy=TRAINING_CONFIG["if_on_policy"], if_per=TRAINING_CONFIG["if_per"], if_gpu=True)

        '''init env.hit_count and total train step'''
        self.env.local_hit_count_real = 0
        self.env.other_hit_count_real = 0
        self.env.local_traffic_offload = 0
        self.env.other_traffic_offload = 0
        self.env.total_traffic = 0
        self.total_train_step = 0
        self.env.replace_count = 0

        self.cache_set = None

    def explore_before_training(self):
        action_dim = self.env.action_dim
        dataset_for_explore = self.train_dl[:self.target_step]

        state = self.env.reset()
        steps = 0
        self.env.local_hit_count = 0
        self.env.other_hit_count = 0

        while steps < self.target_step:
            request = dataset_for_explore[steps]
            if self.env.cache.get(request[1], [request[6], request[7], request[8]]):
                self.env.local_hit_count += 1
                steps += 1
                continue
            else:  # which client has the content
                first_other_hit = True
                other_is_cached = []
                hit_cli = None
                for cli, table in self.total_cache_table.items():
                    if first_other_hit and (request[1] in table):
                        self.env.other_hit_count += 1
                        hit_cli = cli
                        other_is_cached.append(1)
                        first_other_hit = False
                    elif not first_other_hit and (request[1] in table):
                        other_is_cached.append(1)
                    else:
                        other_is_cached.append(0)
            state = self.env.update_state(request, other_is_cached)
            action = rd.randint(action_dim)  # explore
            next_state, reward, done, _ = self.env.step(action, request, other_is_cached)
            steps += 1

            scaled_reward = reward * TRAINING_CONFIG["reward_scale"]
            mask = 0.0 if done else TRAINING_CONFIG["gamma"]
            other = (scaled_reward, mask, action)
            self.buffer.append_buffer(state, other)
        return steps

    def localUpdate(self, total_cache_table, global_parameters=None):
        '''load global model params'''
        if global_parameters:
            global_parameters_in_torch = {}
            for key, var in global_parameters.items():
                global_parameters_in_torch[key] = torch.Tensor(var)
            self.agent.cri.load_state_dict(global_parameters_in_torch, strict=True)

        '''load local dataset'''
        self.train_dl = self.local_data.numpy()

        total_step = 0
        first_training_step = 0

        '''prepare for training'''
        self.env.local_hit_count = 0
        self.env.other_hit_count = 0

        with torch.no_grad():
            steps = self.explore_before_training()

        self.agent.update_net(self.buffer, self.target_step, self.batch_size_for_update_net, TRAINING_CONFIG["repeat_times"])  # pre-training and hard update
        self.agent.act_target.load_state_dict(self.agent.act.state_dict()) if getattr(self.agent, 'act_target', None) else None
        self.agent.cri_target.load_state_dict(self.agent.cri.state_dict()) if getattr(self.agent, 'cri_target', None) else None
        total_step = steps
        first_training_step = steps
        self.cache_set = set(self.env.cache.cache.keys())


        '''start training'''
        training_time = 0
        self.env.local_hit_count = 0
        self.env.other_hit_count = 0
        reward_list_in_one_update = []
        if_reach_goal = False  # In CacheEnv, this flag means there is no data left in dataset
        traffic_in_first_cycle = 0
        isFirst = True
        while not (if_reach_goal or total_step > self.dataset_size):
            temp_set = self.cache_set & set(self.env.cache.cache.keys())
            self.cache_set = set(self.env.cache.cache.keys())
            if total_step + self.target_step <= self.dataset_size:
                steps, reward_list_in_one_explore_env = self.agent.explore_env(self.env, self.buffer, self.target_step, TRAINING_CONFIG["reward_scale"], TRAINING_CONFIG["gamma"], self.train_dl[total_step:total_step+self.target_step], total_cache_table)
            else:
                steps, reward_list_in_one_explore_env = self.agent.explore_env(self.env, self.buffer, self.dataset_size - total_step, TRAINING_CONFIG["reward_scale"], TRAINING_CONFIG["gamma"], self.train_dl[total_step:], total_cache_table)
            total_step += steps
            self.total_train_step += steps
            reward_list_in_one_update += reward_list_in_one_explore_env

            if isFirst:
                traffic_in_first_cycle = self.env.total_traffic
                isFirst = False
            else:  # Real cache is empty in the first circulation, the hit ratio is recorded from the second circulation.
                local_hit_rate_real = (self.env.local_hit_count_real / (self.total_train_step - self.target_step))
                print('| explore_env\thit ratio: ', local_hit_rate_real)
            
            start_training_time = time.time()
            obj_a, obj_c = self.agent.update_net(self.buffer, self.target_step, self.batch_size_for_update_net, TRAINING_CONFIG["repeat_times"])
            end_training_time = time.time()
            training_time += end_training_time - start_training_time

            if steps < self.target_step: if_reach_goal = True

            # Update of real cache to the end when in FL iterations.
            '''update real_cache'''  
            self.env.real_cache.cache = deepcopy(self.env.cache.cache)
            self.env.real_cache.occupation = self.env.cache.occupation

        print('training time in one local update: ', training_time)

        return [self.agent.cri.state_dict(), self.env.local_hit_count_real, self.env.other_hit_count_real, self.total_train_step, reward_list_in_one_update, self.env.local_traffic_offload + self.env.other_traffic_offload, self.env.total_traffic]


def load_dataset(load_dataset_path):
    local_dataset = []
    with open(load_dataset_path, 'rt') as fr:
        cr = csv.reader(fr)
        row_count = 0
        for row in cr:
            local_dataset.append(list(map(int, row)))
            row_count += 1
    return local_dataset


if __name__=="__main__":
    if len(sys.argv) != 4:
        print('usage: python gateway.py agent_index dataset_select cache_size')
        exit(0)
    print('Gateway{} starts!'.format(sys.argv[1]))

    '''init model'''
    if sys.argv[2] == 'movielens':
        # movielens
        load_dataset_path = 'movielens25_gateway{}.csv'.format(sys.argv[1])
    dataset = load_dataset(load_dataset_path)

    ENV_CONFIG['cache_size'] = int(sys.argv[3])

    gateway = Gateway('Gateway{}'.format(sys.argv[1]), torch.tensor(dataset))
    local_parameters = {}

    global_parameters = None  # Fill in here if recv global params.
    total_cache_table = {}  # Fill in here if recv total cache table

    '''local training'''
    local_vars, local_hit_count_real, other_hit_count_real, total_train_step, reward_list_in_one_update, total_traffic_offload, total_traffic = gateway.localUpdate(total_cache_table, global_parameters=global_parameters)
    for key, var in local_vars.items():
        local_parameters[key] = var.clone()
    local_cache_table = (gateway.name, tuple(gateway.env.real_cache.cache.keys()))
