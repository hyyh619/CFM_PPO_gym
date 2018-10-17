# ----------------------------------------------
# Project: Proximal Policy Optimization
# Author: benethuang
# Date: 2018.9.30
# ----------------------------------------------

from collections import deque
import numpy as np
import random

GAMMA = 0.99

class ReplayBuffer(object):

    def __init__(self, logger, buffer_size):
        self.__logger         = logger
        self.buffer_size      = buffer_size
        self.num_experiences  = 0
        self.state_buffer     = []
        self.action_buffer    = []
        self.pred_buffer      = []
        self.reward_buffer    = []
        self.reward_over_time = []

    def get_batch(self):
        batch = [[], [], [], []]
        self.transform_reward()

        for i in range(len(self.state_buffer)):
            obs     = self.state_buffer[i]
            action  = self.action_buffer[i]
            pred    = self.pred_buffer[i]
            r       = self.reward_buffer[i]

            batch[0].append(obs)
            batch[1].append(action)
            batch[2].append(pred)
            batch[3].append(r)

        self.erase()
        return batch

    def size(self):
        return self.num_experiences

    def get_total_reward(self):
        if len(self.reward_over_time) < 100:
            mean_over_time_reward = np.mean(self.reward_over_time)
        else:
            mean_over_time_reward = np.mean(self.reward_over_time[-100:])

        current_reward_sum = np.array(self.reward_buffer).sum()
        #    print('Episode #', self.episode, '\tfinished with reward', ,
        #          '\tAverage reward of last 100 episode :', )
        return current_reward_sum, mean_over_time_reward

    def transform_reward(self):
        self.reward_over_time.append(np.array(self.reward_buffer).sum())
        #print("len: {0}".format(len(self.reward_buffer)))

        # orig
        for j in range(len(self.reward_buffer) - 2, -1, -1):
            #print("j: {0}".format(j))
            self.reward_buffer[j] += self.reward_buffer[j + 1] * GAMMA

        #for j in range(len(self.reward_buffer)):
        #    reward = self.reward_buffer[j]
        #    for k in range(j + 1, len(self.reward_buffer)):
        #        reward += self.reward_buffer[k] * GAMMA ** k
        #    self.reward_buffer[j] = reward

        current_reward_sum, mean_over_time_reward = self.get_total_reward()
        #self.__logger.info(self.reward_buffer)
        if len(self.reward_buffer) > 0:
            self.__logger.info("reward sum: {0:2.4f}, reward mean: {1:2.4f}".format(
                self.reward_buffer[0], mean_over_time_reward))

    def add(self, state, action, pred_action, reward, done):
        self.num_experiences += 1
        self.state_buffer.append(state) 
        self.action_buffer.append(action)
        self.pred_buffer.append(pred_action)
        self.reward_buffer.append(reward)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.state_buffer     = []
        self.action_buffer    = []
        self.pred_buffer      = []
        self.reward_buffer    = []
        self.num_experiences  = 0

    def clear_all(self):
        self.state_buffer     = []
        self.action_buffer    = []
        self.pred_buffer      = []
        self.reward_buffer    = []
        self.reward_over_time = []
        self.num_experiences  = 0
