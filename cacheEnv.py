import numpy as np
import collections
import csv
from queue import Queue

class AVGFCache(object):
    def __init__(self, size, window, freq_discount):
        self.capacity = size
        self.occupation = 0  # self.occupation should be less than self.capacity
        self.cache = {}
        self.cache_key_list = []
        self.cache_freq_list = []
        self.cache_size_list = []
        self.window = window
        self.freq_discount = freq_discount
 
    def get(self, key, window_freq): # window_freq: [short_freq, middle_freq, long_freq]
        if key in self.cache_key_list:
            p = self.cache_key_list.index(key)
            avg_freq_ = (1 / (2 * self.cache_size_list[p] ** 0.5)) * (window_freq[0] + window_freq[1] * (self.window[0] / self.window[1]) * self.freq_discount + window_freq[2] * (self.window[0] / self.window[2]) * (self.freq_discount ** 2))
            feature_list = self.cache.pop(key)
            del self.cache_key_list[p]
            del self.cache_freq_list[p]
            del self.cache_size_list[p]
            feature_list[1:] = window_freq[:]
            self.cache[key] = feature_list.copy()
            insert_flag = False
            cache_len = len(self.cache_key_list)
            if cache_len > 0:
                for i in range(cache_len):
                    if avg_freq_ >= self.cache_freq_list[i]:
                        self.cache_key_list.insert(i, key)
                        self.cache_freq_list.insert(i, avg_freq_)
                        self.cache_size_list.insert(i, feature_list[0])
                        insert_flag = True
                        break
                if insert_flag == False:
                    self.cache_key_list.append(key)
                    self.cache_freq_list.append(avg_freq_)
                    self.cache_size_list.append(feature_list[0])
                insert_flag = False
            else:
                self.cache_key_list.append(key)
                self.cache_freq_list.append(avg_freq_)
                self.cache_size_list.append(feature_list[0])
        else:
            feature_list = None
            return None
        return [key, feature_list]
 
    def set(self, key, feature_list):  # feature_list: [size, short_freq, middle_freq, long_freq]
        avg_freq_ = (1 / (2 * feature_list[0] ** 0.5)) * (feature_list[1] + feature_list[2] * (self.window[0] / self.window[1]) * self.freq_discount + feature_list[3] * (self.window[0] / self.window[2]) * (self.freq_discount ** 2))
        if key in self.cache_key_list:
            feature_list_old = self.cache.pop(key)
            p = self.cache_key_list.index(key)
            del self.cache_key_list[p]
            del self.cache_freq_list[p]
            del self.cache_size_list[p]
            self.cache[key] = feature_list.copy()
            insert_flag = False
            cache_len = len(self.cache_key_list)
            if cache_len > 0:
                for i in range(cache_len):
                    if avg_freq_ >= self.cache_freq_list[i]:
                        self.cache_key_list.insert(i, key)
                        self.cache_freq_list.insert(i, avg_freq_)
                        self.cache_size_list.insert(i, feature_list[0])
                        insert_flag = True
                        break
                if insert_flag == False:
                    self.cache_key_list.append(key)
                    self.cache_freq_list.append(avg_freq_)
                    self.cache_size_list.append(feature_list[0])
                insert_flag = False
            else:
                self.cache_key_list.append(key)
                self.cache_freq_list.append(avg_freq_)
                self.cache_size_list.append(feature_list[0])
        else:
            size = feature_list[0]
            if self.occupation + size > self.capacity:  # If cache is filled，then delete items
                evict_item_list = []
                evict_item_index = -1
                while self.occupation + size > self.capacity:
                    self.occupation -= self.cache_size_list[evict_item_index]
                    evict_item_index -= 1
                evict_item_index += 1
                # Delete old items
                evict_item_keys = self.cache_key_list[evict_item_index:]
                evict_item_freqs = self.cache_freq_list[evict_item_index:]
                for k in evict_item_keys:
                    evict_item_list.append(self.cache.pop(k))
                self.cache_key_list = self.cache_key_list[:evict_item_index]
                self.cache_freq_list = self.cache_freq_list[:evict_item_index]
                self.cache_size_list = self.cache_size_list[:evict_item_index]
                # Add new item
                self.cache[key] = feature_list.copy()
                self.occupation += size
                insert_flag = False
                cache_len = len(self.cache_freq_list)
                for i in range(cache_len):
                    if avg_freq_ >= self.cache_freq_list[i]:
                        self.cache_key_list.insert(i, key)
                        self.cache_freq_list.insert(i, avg_freq_)
                        self.cache_size_list.insert(i, feature_list[0])
                        insert_flag = True
                        break
                if insert_flag == False:
                    self.cache_key_list.append(key)
                    self.cache_freq_list.append(avg_freq_)
                    self.cache_size_list.append(feature_list[0])
                insert_flag = False
                return [evict_item_list, evict_item_freqs]  # evict_item_list: [size, short_freq, middle_freq, long_freq]
            else:
                self.cache[key] = feature_list.copy()
                self.occupation += size
                insert_flag = False
                cache_len = len(self.cache_freq_list)
                if cache_len > 0:
                    for i in range(cache_len):
                        if avg_freq_ >= self.cache_freq_list[i]:
                            self.cache_key_list.insert(i, key)
                            self.cache_freq_list.insert(i, avg_freq_)
                            self.cache_size_list.insert(i, feature_list[0])
                            insert_flag = True
                            break
                    if insert_flag == False:
                        self.cache_key_list.append(key)
                        self.cache_freq_list.append(avg_freq_)
                        self.cache_size_list.append(feature_list[0])
                    insert_flag = False
                else:
                    self.cache_key_list.append(key)
                    self.cache_freq_list.append(avg_freq_)
                    self.cache_size_list.append(feature_list[0])
        
        return [None, None]

class RealCache(object):
    def __init__(self, size):
        self.capacity = size
        self.occupation = 0
        self.cache = {}

    def get(self, key):
        if key in self.cache.keys():
            return [key, self.cache[key]]
        else:
            return None


'''cache environment for each client for FL, add real_cache'''

class CacheEnv(object):
    def __init__(self, config):
        super().__init__()
        self.env_name = 'CacheEnv'
        self.target_return = None
        self.action_size = config["action_size"]
        self.action_dim = self.action_size
        self.state_size = config["state_size"]
        self.state_dim = self.state_size
        self.cache_size = config["cache_size"]
        self.window = config["window"]
        self.bins = config["bins"]
        self.gamma_r = config["gamma_r"]
        self.freq_discount = config["freq_discount"]
        self.miu_r_l = [10, 10]  # Adjust the weights at r_local, here we give an example of single point cache
        self.miu_r_o = [1, 0]  # Adjust the weights of r_local and r_other, here we give an example of single point cache
        self.local_hit_count = 0
        self.local_hit_count_real = 0
        self.other_hit_count = 0
        self.other_hit_count_real = 0
        self.local_traffic_offload = 0
        self.other_traffic_offload = 0
        self.total_traffic = 0
        self.req_to_other_count = 0
        self.max_content_size = 5
        self.if_discrete = True
        self.max_step = None
        self.window_queue = [Queue(maxsize=self.window[0]), Queue(maxsize=self.window[1]), Queue(maxsize=self.window[2])]
        self.replace_count = 0

    def update_window_queue(self, content):
        queue_pop_item = [None, None, None]
        if self.window_queue[0].full():
            queue_pop_item[0] = self.window_queue[0].get()
            if self.window_queue[1].full():
                queue_pop_item[1] = self.window_queue[1].get()
                if self.window_queue[2].full():
                    queue_pop_item[2] = self.window_queue[2].get()

        self.window_queue[0].put(content)
        self.window_queue[1].put(content)
        self.window_queue[2].put(content)
        return queue_pop_item

    def update_cache_freq(self, content, queue_pop_item):
        if queue_pop_item[0] != None:
            cache_list = list(self.cache.cache.keys())
            if (queue_pop_item[0] in cache_list) and (queue_pop_item[0] != content):
                self.cache.cache[queue_pop_item[0]][1] -= 1
                # AVGFCache needs to update cache_freq_list
                if isinstance(self.cache, AVGFCache):
                    p0 = self.cache.cache_key_list.index(queue_pop_item[0])
                    feature_list_0 = self.cache.cache[queue_pop_item[0]].copy()
                    avg_freq_new0 = feature_list_0[1] + feature_list_0[2] * (self.window[0] / self.window[1]) * self.freq_discount + feature_list_0[3] * (self.window[0] / self.window[2]) * (self.freq_discount ** 2)
                    del self.cache.cache_key_list[p0]
                    del self.cache.cache_freq_list[p0]
                    del self.cache.cache_size_list[p0]
                    insert_flag_0 = False
                    cache_freq_len = len(self.cache.cache_freq_list)
                    if cache_freq_len > 0:
                        for i0 in range(cache_freq_len):
                            if avg_freq_new0 >= self.cache.cache_freq_list[i0]:
                                self.cache.cache_key_list.insert(i0, queue_pop_item[0])
                                self.cache.cache_freq_list.insert(i0, avg_freq_new0)
                                self.cache.cache_size_list.insert(i0, feature_list_0[0])
                                insert_flag_0 = True
                                break
                        if insert_flag_0 == False:
                            self.cache.cache_key_list.append(queue_pop_item[0])
                            self.cache.cache_freq_list.append(avg_freq_new0)
                            self.cache.cache_size_list.append(feature_list_0[0])
                    else:
                        self.cache.cache_key_list.append(queue_pop_item[0])
                        self.cache.cache_freq_list.append(avg_freq_new0)
                        self.cache.cache_size_list.append(feature_list_0[0])
            if queue_pop_item[1] != None:
                if (queue_pop_item[1] in cache_list) and (queue_pop_item[1] != content):
                    self.cache.cache[queue_pop_item[1]][2] -= 1
                    # AVGFCache needs to update cache_freq_list
                    if isinstance(self.cache, AVGFCache):
                        p1 = self.cache.cache_key_list.index(queue_pop_item[1])
                        feature_list_1 = self.cache.cache[queue_pop_item[1]].copy()
                        avg_freq_new1 = feature_list_1[1] + feature_list_1[2] * (self.window[0] / self.window[1]) * self.freq_discount + feature_list_1[3] * (self.window[0] / self.window[2]) * (self.freq_discount ** 2)
                        del self.cache.cache_key_list[p1]
                        del self.cache.cache_freq_list[p1]
                        del self.cache.cache_size_list[p1]
                        insert_flag_1 = False
                        cache_freq_len = len(self.cache.cache_freq_list)
                        if cache_freq_len > 0:
                            for i1 in range(cache_freq_len):
                                if avg_freq_new1 >= self.cache.cache_freq_list[i1]:
                                    self.cache.cache_key_list.insert(i1, queue_pop_item[1])
                                    self.cache.cache_freq_list.insert(i1, avg_freq_new1)
                                    self.cache.cache_size_list.insert(i1, feature_list_1[0])
                                    insert_flag_1 = True
                                    break
                            if insert_flag_1 == False:
                                self.cache.cache_key_list.append(queue_pop_item[1])
                                self.cache.cache_freq_list.append(avg_freq_new1)
                                self.cache.cache_size_list.append(feature_list_1[0])
                        else:
                            self.cache.cache_key_list.append(queue_pop_item[1])
                            self.cache.cache_freq_list.append(avg_freq_new1)
                            self.cache.cache_size_list.append(feature_list_1[0])
                if queue_pop_item[2] != None:
                    if (queue_pop_item[2] in cache_list) and (queue_pop_item[2] != content):
                        self.cache.cache[queue_pop_item[2]][3] -= 1
                        # AVGFCache needs to update cache_freq_list
                        if isinstance(self.cache, AVGFCache):
                            p2 = self.cache.cache_key_list.index(queue_pop_item[2])
                            feature_list_2 = self.cache.cache[queue_pop_item[2]].copy()
                            avg_freq_new2 = feature_list_2[1] + feature_list_2[2] * (self.window[0] / self.window[1]) * self.freq_discount + feature_list_2[3] * (self.window[0] / self.window[2]) * (self.freq_discount ** 2)
                            del self.cache.cache_key_list[p2]
                            del self.cache.cache_freq_list[p2]
                            del self.cache.cache_size_list[p2]
                            insert_flag_2 = False
                            cache_freq_len = len(self.cache.cache_freq_list)
                            if cache_freq_len > 0:
                                for i2 in range(cache_freq_len):
                                    if avg_freq_new2 >= self.cache.cache_freq_list[i2]:
                                        self.cache.cache_key_list.insert(i2, queue_pop_item[2])
                                        self.cache.cache_freq_list.insert(i2, avg_freq_new2)
                                        self.cache.cache_size_list.insert(i2, feature_list_2[0])
                                        insert_flag_2 = True
                                        break
                                if insert_flag_2 == False:
                                    self.cache.cache_key_list.append(queue_pop_item[2])
                                    self.cache.cache_freq_list.append(avg_freq_new2)
                                    self.cache.cache_size_list.append(feature_list_2[0])
                            else:
                                self.cache.cache_key_list.append(queue_pop_item[2])
                                self.cache.cache_freq_list.append(avg_freq_new2)
                                self.cache.cache_size_list.append(feature_list_2[0])

    def update_state(self, request, other_is_cached):
        self.ori_state[0] = (self.cache_size - self.cache.occupation) / self.cache_size  # Free space in cache
        self.ori_state[1] = request[2] / self.max_content_size  # Content size
        self.ori_state[2] = (request[4] / 3600) if (request[4] < 36000) else 10  # Time difference, normalized in intervals of hours
        self.ori_state[3] = (request[5] / 1000) if (request[5] < 10000) else 10  # Frequency difference, normalized by the length of the maximum time window
        self.ori_state[4] = request[6] / self.window[0]  # short
        self.ori_state[5] = request[7] / self.window[1]  # middle
        self.ori_state[6] = request[8] / self.window[2]  # long
        self.ori_state[7] = 0  # Is cached locally

        self.state[0] = np.digitize(self.ori_state[0], self.bins)
        self.state[1] = np.digitize(self.ori_state[1], self.bins)
        self.state[2] = np.digitize(self.ori_state[2], self.bins)
        self.state[3] = np.digitize(self.ori_state[3], self.bins)
        self.state[4] = np.digitize(self.ori_state[4], self.bins)
        self.state[5] = np.digitize(self.ori_state[5], self.bins)
        self.state[6] = np.digitize(self.ori_state[6], self.bins)
        self.state[7] = 0
        self.state[8] = 1 if sum(other_is_cached) >= 1 else 0
        return self.state

    def step(self, action, request, other_is_cached):
        evict_item_avg_freq_list = []
        avg_freq_of_cache_items = []
        other_cache_rate = sum(other_is_cached) / len(other_is_cached) if len(other_is_cached) > 0 else 0
        other_cache_flag = 1 if other_cache_rate > 0 else 0
        '''execute action and update next_state'''
        evict_item_list = []
        if action:
            evict_item_list, evict_item_freqs = self.cache.set(request[1], [request[2], request[6], request[7], request[8]])
            if evict_item_list:
                self.replace_count += 1
            self.next_state = self.state.copy()
            self.next_state[0] = np.digitize((self.cache_size - self.cache.occupation) / self.cache_size, self.bins)
            self.next_state[7] = 1
            # Adjust r1 and r2 for different demands, here we give an example of single point cache
            r1 = (self.ori_state[4] + self.gamma_r * self.ori_state[5]  + self.gamma_r * self.gamma_r * self.ori_state[6])
            r2 = (request[2] / self.max_content_size) * (self.cache.occupation / self.cache_size)
            self.reward = self.miu_r_o[0] * (self.miu_r_l[0] * r1 - self.miu_r_l[0] * r2) - self.miu_r_o[1] * other_cache_rate
        else:
            self.next_state = self.state.copy()
            # Adjust r1 and r2 for different demands, here we give an example of single point cache
            r1 = request[2] * (self.cache.occupation / self.cache_size)
            r2 = (self.ori_state[4] + self.gamma_r * self.ori_state[5]  + self.gamma_r * self.gamma_r * self.ori_state[6]) * request[2]
            self.reward = self.miu_r_o[0] * (self.miu_r_l[1] * r1 - self.miu_r_l[0] * r2) + self.miu_r_o[1] * other_cache_rate
        self.total_episode_score_so_far += self.reward
        return self.next_state, self.reward, False, {}

    def reset(self):
        """Resets the game information so we are ready to play a new episode"""
        self.ori_state = np.zeros(self.state_size)
        self.state = np.zeros(self.state_size)
        self.cache = AVGFCache(self.cache_size, self.window, self.freq_discount)
        self.real_cache = RealCache(self.cache_size)
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        return self.state

def load_data_set(data_path=None):
    req = []
    with open(data_path, 'rt') as fr:
        cr = csv.reader(fr)
        for row in cr:
            req.append(list(map(int, row)))
    return req

