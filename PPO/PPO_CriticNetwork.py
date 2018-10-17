# ----------------------------------------------
# Project: Proximal Policy Optimization
# Author: benethuang
# Date: 2018.9.30
# ----------------------------------------------

import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, LSTM, Reshape, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS    = 300
HIDDEN2_UNITS    = 600
LSTM_OUTPUT_SIZE = 128
HIDDEN_SIZE      = 256

class CriticNetwork(object):
    def __init__(self, logger, sess, state_size, action_size, batch_size, learning_rate, training, lstm_mode = 0):
        self.__logger        = logger
        self.__sess          = sess
        self.__batch_size    = batch_size
        self.__learning_rate = learning_rate
        self.__action_size   = action_size
        self.__state_size    = state_size

        K.set_session(sess)

        # Now create the model
        self.model = self.create_critic_network(self.__state_size, self.__action_size, training)
        return

    def create_critic_network(self, state_size, action_dim, training):
        state_input = Input(shape=(state_size, ))
        x = Dense(300, activation = 'relu')(state_input)
        if training == 1:
            x = Dropout(0.5)(x)
        x = Dense(600, activation = 'relu')(x)
        if training == 1:
            x = Dropout(0.5)(x)

        out_value = Dense(1)(x)

        model = Model(inputs = [state_input], outputs = [out_value])
        model.compile(optimizer = Adam(lr = self.__learning_rate), loss = 'mse')
        model.summary()
        return model

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        self.__logger.info("Successfully saved network. %s" %(path))

    def load_network(self, path):
        self.model.load_weights(path)
        self.__logger.info("Succesfully loaded network. %s" %(path))