# ----------------------------------------------
# Project: Proximal Policy Optimization
# Author: benethuang
# Date: 2018.9.30
# ----------------------------------------------

import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, LSTM, Reshape, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN_SIZE   = 256
LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
NOISE         = 0.5

def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred)/ (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction)/ (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss

class ActorNetwork(object):
    def __init__(self, logger, sess, state_size, action_size, batch_size, learn_rate, training, lstm_mode = 0):
        self.__logger        = logger
        self.__sess          = sess
        self.__batch_size    = batch_size
        self.__learning_rate = learn_rate
        self.__state_size    = state_size
        self.__action_size   = action_size

        K.set_session(sess)

        self.DUMMY_ACTION = np.zeros((1, action_size))
        self.DUMMY_VALUE  = np.zeros((1, 1))

        # Now create the model
        self.model = self.create_actor_network(state_size, action_size, training)
        return

    def create_actor_network(self, state_size, action_dim, training):
        # Input definition
        state_input    = Input(shape=(state_size,))
        advantage      = Input(shape=(1,))
        old_prediction = Input(shape=(action_dim,))

        # network layer.
        x = Dense(300, activation='relu')(state_input)
        if training == 1:
            x = Dropout(0.5)(x)
        x = Dense(600, activation='relu')(x)
        if training == 1:
            x = Dropout(0.5)(x)

        # out_actions = Dense(NUM_ACTIONS, name='output')(x)
        # Rotate(-1~1)/Move Direction(0~1)/Move Distance(0~1)/Fire(0~1, >0.5 means fire)
        rotation          = Dense(1, activation='tanh')(x)
        move_direction    = Dense(1, activation='sigmoid')(x)
        move_accelaration = Dense(1, activation='sigmoid')(x)
        fire              = Dense(1, activation='sigmoid')(x)
        out_actions       = merge([rotation, move_direction, move_accelaration, fire], mode = 'concat')

        model = Model(inputs = [state_input, advantage, old_prediction], outputs = [out_actions])
        model.compile(optimizer = Adam(lr = self.__learning_rate),
                      loss = [proximal_policy_optimization_loss_continuous(
                              advantage = advantage,
                              old_prediction = old_prediction)])
        model.summary()
        return model

    def get_action(self, observation):
        p = self.model.predict([observation.reshape(1, self.__state_size), self.DUMMY_VALUE, self.DUMMY_ACTION])

        #if training == 1:
        #    action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        #else:
        #    action = action_matrix = p[0]

        return p

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        self.__logger.info("Successfully saved network. %s" %(path))

    def load_network(self, path):
        self.model.load_weights(path)
        self.__logger.info("Succesfully loaded network. %s" %(path))