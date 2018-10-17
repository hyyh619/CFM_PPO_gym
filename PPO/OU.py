# ----------------------------------------------
# Project: Proximal Policy Optimization
# Author: benethuang
# Date: 2018.9.30
# ----------------------------------------------

import random
import numpy as np 

class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

if __name__ == '__main__':
    #生成正态分布
    x = np.round(np.random.normal(1.75, 0.2, 5000), 2)
    y = np.round(np.random.normal(100, 10, 5000), 2)

    print(x)
    print(y)