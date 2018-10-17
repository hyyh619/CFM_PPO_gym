# ---------------------------------------
# Project: Proximal Policy Optimization
# Author: benethuang
# Date: 2018.9.30
# ---------------------------------------

import os
import json
import csv
import re
import cv2
import time
import logging
import configparser
import random
import numpy as np
import math
from collections import deque

from aimodel.AIModel import AIModel
from CFM_PPO.PPO.PPO_ReplayBuffer import ReplayBuffer
from CFM_PPO.PPO.PPO_ActorNetwork import ActorNetwork
from CFM_PPO.PPO.PPO_CriticNetwork import CriticNetwork
from CFM_PPO.PPO.OU import OU
from AgentAPI import AgentAPIMgr

import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array

COMMON_CFG_FILE  = '../cfg/common.json'
AI_CFG_FILE      = '../data/AI/AI.json'
FINAL_EPSILON    = 0.001
INITIAL_EPSILON  = 0.3
NUM_FRAMES       = 4
EPSILON_DECAY    = 300000
MIN_OBSERVATION  = 2000
BUFFER_SIZE      = 10000
MINIBATCH_SIZE   = 16

ACTION_NAME_LIST = ['forward',
                    'backward',
                    'move_left',
                    'move_right',
                    'turn_left',
                    'turn_right',
                    'no_action']

ACTION_DIM = 4   # Rotate(-1~1)/Move Direction(0~1)/Move Distance(0~1)/Fire(0~1, >0.5 means fire)
STATE_DIM  = 41  # It includes 1 + 4*6(humanInfo) + 4(myPos, myViewPos) + 1(attackVector) + 3(blood, ammo, kills) + 2*4(our guys' position in small map).
GAMMA      = 0.99
NOISE      = 0.01

class PPOAI(AIModel):
    def __init__(self):
        AIModel.__init__(self)

    def Init(self, agentEnv):
        self.agentAPI = AgentAPIMgr.AgentAPIMgr()
        ret = self.agentAPI.Initialize("../cfg/task.json")

        self.__sess = self._GPUSetting()

        self.__screenWidth  = 0
        self.__screenHeight = 0

        self.__lastFrameTime = time.time()
        self.__agentEnv      = agentEnv
        self.__timePerFrame  = 0.2 # based on training time, we will do action on 10FPS.

        self.__bEposideStart = False
        self.__nSkipCounter  = 0

        commonArgs = self._LoadCommonParams()
        self.logger.info('commonArgs : {0}'.format(commonArgs))
        self._ProcArgs(commonArgs)

        # Load hyper parameters and fixed strategy parameters.
        self._LoadAIParams()

        # Create Proximal Policy Optimization model.
        self._CreatePPO()

        # For fixed strategy
        self._InitFixedStrategy()

        return True

    def _LoadAIParams(self):
        # This parameters can be setting by AI.json based on different phone.
        self.__attack_turn_left_step  = 4
        self.__attack_turn_right_step = 2
        self.__attack_init_step       = 2
        self.__enemy_init_step        = 1
        self.__enemy_turn_step        = 10

        # If we turn left or right when being attacked, we will don't check attacked flag for a while.
        self.__attack_Waiting         = 10

        # Using jumping randomly to work around obstacle.
        self.__enable_random_jump     = 1

        # There is intertia, if there are a lot of turn left or turn right.
        # We will turn the opposite direction after the successive turn left/right.
        self.__enable_compensate_turn = 1
        self.__compensateTurn         = 5

        # action sleep.
        self.__action_sleep = 0.03

        # Default Dueling DQN parameters
        self.__decayRate             = 0.99
        self.__learnRate             = 1e-4
        self.__numActions            = 4 # forward/backward/left/right/turn left/turn right/no action
        self.__numFrames             = 8 # send 8 frames to network
        self.__numReplaySamples      = BUFFER_SIZE
        self.__inputX                = 512
        self.__inputY                = 288
        self.__finalEpsilon          = FINAL_EPSILON
        self.__initEpsilon           = INITIAL_EPSILON
        self.__epsilonDecay          = EPSILON_DECAY
        self.__minObservation        = MIN_OBSERVATION
        self.__minBatchSize          = MINIBATCH_SIZE
        self.__updateTarget          = 500
        self.__updateAgent           = 2000
        self.__idleTrainNum          = 100  # This parameter cannot be set too large, because it will cause loss boom.
        self.__path                  = "../../AI/train-ppo/"
        self.__modelNamePrefix       = "car_clone_"
        self.__saveModelNum          = 2000
        self.__saveModelPersistNum   = 10000
        self.__usingTCP              = 0
        self.__tcpModelFilePath      = '../data/AI/TCP'
        self.__tcpReceivedActorFile  = ''
        self.__tcpReceivedCriticFile = ''
        self.__bTraining             = 1
        self.__bLocalTraining        = 1
        self.__localTrainingInterval = 2
        self.__localTrainingPath     = '../data/AI/train-ppo'
        self.__ppoEpoch              = 10

        if os.path.exists(AI_CFG_FILE):
            with open(AI_CFG_FILE, 'rb') as file:
                jsonstr = file.read()
                commonCfg = json.loads(str(jsonstr, encoding='utf-8'))

                self.__attack_turn_left_step  = commonCfg.get('attack_turn_left')
                self.__attack_turn_right_step = commonCfg.get('attack_turn_right')
                self.__attack_init_step       = commonCfg.get('attack_init')
                self.__attack_Waiting         = commonCfg.get('attack_waiting')
                self.__enemy_init_step        = commonCfg.get('enemy_init')
                self.__enemy_turn_step        = commonCfg.get('enemy_turn')
                self.__enable_random_jump     = commonCfg.get('enable_random_jump')
                self.__enable_compensate_turn = commonCfg.get('enable_compensate_turn')
                self.__action_sleep           = commonCfg.get('action_sleep')

                self.__decayRate             = commonCfg.get('decay_rate')
                self.__learnRate             = commonCfg.get('learn_rate')
                self.__numActions            = commonCfg.get('num_actions')
                self.__numFrames             = commonCfg.get('num_frames')
                self.__numReplaySamples      = commonCfg.get('num_replay_samples')
                self.__inputX                = commonCfg.get('input_x')
                self.__inputY                = commonCfg.get('input_y')
                self.__finalEpsilon          = commonCfg.get('final_epsilon')
                self.__initEpsilon           = commonCfg.get('init_epsilon')
                self.__epsilonDecay          = commonCfg.get('epsilon_decay')
                self.__minObservation        = commonCfg.get('min_observation')
                self.__minBatchSize          = commonCfg.get('minibatch')
                self.__updateTarget          = commonCfg.get('update_target')
                self.__updateAgent           = commonCfg.get('update_agent')
                self.__idleTrainNum          = commonCfg.get('idle_train_num')
                self.__path                  = commonCfg.get('model_path')
                self.__modelNamePrefix       = commonCfg.get('model_name_prefix')
                self.__saveModelNum          = commonCfg.get('save_model_num')
                self.__saveModelPersistNum   = commonCfg.get('save_model_persist_num')
                self.__usingTCP              = commonCfg.get('using_tcp')
                self.__bTraining             = commonCfg.get('training')
                self.__bLocalTraining        = commonCfg.get('loacl_training')
                self.__localTrainingPath     = commonCfg.get('local_training_path')
                self.__localTrainingInterval = commonCfg.get('local_training_interval')
                self.__ppoEpoch              = commonCfg.get('ppo_epoch')

        if not os.path.exists(self.__tcpModelFilePath):
            os.mkdir(self.__tcpModelFilePath)

        return

    def _CreatePPO(self):
        # Update current agent's eval network
        self.__actorNetwork     = ''
        self.__criticNetwork    = ''
        self.__targetNetwork    = ''
        self.__syncNetworkCount = 0
        self.__epsilon          = self.__initEpsilon
        self.__observeNum       = 0
        self.__gradient_steps   = 0

        # A buffer that keeps the last 3 images
        self.__processBuffer = []

        stateSize           = STATE_DIM * self.__numFrames
        self.__actor        = ActorNetwork(self.logger, self.__sess, stateSize, ACTION_DIM, self.__minBatchSize, self.__learnRate, self.__bLocalTraining)
        self.__critic       = CriticNetwork(self.logger, self.__sess, stateSize, ACTION_DIM, self.__minBatchSize,  self.__learnRate, self.__bLocalTraining)
        self.__replayBuffer = ReplayBuffer(self.logger, self.__minBatchSize)
        self.__OU           = OU()

        # Load Model
        if self.__bLocalTraining == 1:
            self._loadModelConfigFile = self.__localTrainingPath + '/' + "checkpoint.json"
        else:
            self._loadModelConfigFile = self.__path + '/' + "checkpoint.json"

        self._lastActorModel1           = ""
        self._lastActorModel2           = ""
        self._lastActorModel3           = ""
        self._lastCriticModel1          = ""
        self._lastCriticModel2          = ""
        self._lastCriticModel3          = ""
        self.__modelIndex               = 0
        self.__lastSavedActorModelFile  = ""
        self.__lastSavedCriticModelFile = ""

        # Training counter for save model
        self.__trainingCounter = 0

        if self.__bTraining == 1 and self.__bLocalTraining == 0:
            if self.__usingTCP == 0:
                self._LoadModel(self._loadModelConfigFile)
            elif self.__usingTCP == 1:
                self._SyncNetworkFromTCP()
        elif self.__bLocalTraining == 1:
            self._SyncNetworkFromFile(self.__localTrainingPath)
        else:
            self._SyncNetworkFromFile(self.__localTrainingPath)

    def _InitFixedStrategy(self):
        # For switch ammo, we should skip some frames after clicking switch ammo button.
        self.__switchAmmoCount   = 0
        self.__switchAmmoWaiting = 40

        # Counter forward, If we step forward enough, we will turn right or left randomly.
        # This is for breaking stuck at the corner.
        self.forwardCount = 0

        # If there are a lot of same action, we will count it.
        self.__lastActions         = [0, 0, 0, 0]
        self.__lastSameActionCount = 0
        self.__changeAction        = 4

        # If we turn left or right when being attacked, we will don't check attacked flag.
        self.attackedCount = 0

        # If using DQN to make decision, we should use it at least 10 times
        self.__useDQNMin   = 20
        self.__bUseDQN     = False

    def _ProcArgs(self, commonArgs):
        taskID = commonArgs.get('TaskID')
        if taskID is not None:
            path = ''

        runType = commonArgs.get('RunType')
        return

    def _ConvertProcessBuffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        if len(self.__processBuffer) < self.__numFrames:
            return None

        # Inverse
        invProcessBuffer = []
        len1 = self.__numFrames - 1
        for i in range(self.__numFrames):
            invProcessBuffer.append(self.__processBuffer[len1 - i])

        state = np.concatenate(invProcessBuffer, axis=0)
        #state = state.reshape(1, state.shape[0] * self.__numFrames)
        #state = state[0]
        return state

    def _AddProcessBuffer(self, state):
        if len(self.__processBuffer) >= self.__numFrames:
            self.__processBuffer.pop(0)

        self.__processBuffer.append(self.__nextObservation)

    def _LoadCommonParams(self):
        if os.path.exists(COMMON_CFG_FILE):
            with open(COMMON_CFG_FILE, 'rb') as file:
                jsonstr = file.read()
                commonCfg = json.loads(str(jsonstr, encoding='utf-8'))
                return commonCfg
        else:
            self.logger.error('No common param file.')

        return {}

    def _GPUSetting(self):
        # Set memory allocation of GPU for this AI model
        config = tf.ConfigProto(device_count={'gpu':0})
        # config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        return sess

    def _ConvertState(self, state):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""

        return state

    def _LoadModel(self, loadModelConfigFile):
        if not os.path.exists(loadModelConfigFile):
            return

        with open(loadModelConfigFile, 'rb') as file:
            jsonstr                 = file.read()
            commonCfg               = json.loads(str(jsonstr, encoding='utf-8'))
            actorModelFile          = commonCfg.get('actor_model_checkpoint_path')
            criticModelFile         = commonCfg.get('critic_model_checkpoint_path')
            self.__lastActorModel1  = commonCfg.get('last_actor_model1')
            self.__lastActorModel2  = commonCfg.get('last_actor_model2')
            self.__lastActorModel3  = commonCfg.get('last_actor_model3')
            self.__lastCriticModel1 = commonCfg.get('last_critic_model1')
            self.__lastCriticModel2 = commonCfg.get('last_critic_model2')
            self.__lastCriticModel3 = commonCfg.get('last_critic_model3')

            if actorModelFile == '' or criticModelFile == '':
                return

            [shotname, extension] = os.path.splitext(actorModelFile)
            actorModelFile  = self.__path + actorModelFile
            criticModelFile = self.__path + criticModelFile
            self._SetModelFile(actorModelFile, criticModelFile)
            self.__actor.load_network(actorModelFile)
            self.__critic.load_network(criticModelFile)

            # Get training count
            count = int(re.sub("\D", "", shotname))
            self.__trainingCounter = count

    def _SetModelFile(self, actorModelFile, criticModelFile):
        self.__currentActorModelFile  = actorModelFile
        self.__currentCriticModelFile = criticModelFile

    def _SaveModel(self, loadModelConfigFile, savedPath):
        actorFileName      = ''
        criticFileName     = ''
        actorFileFullName  = ''
        criticFileFullName = ''

        if self.__trainingCounter % self.__saveModelNum == 0:
            actorFileName = '%s_actor_%d.h5' % (self.__modelNamePrefix, self.__trainingCounter)
            actorFileFullName = self.__path + '/' + actorFileName
            self.logger.info("Saving Network: %s", actorFileFullName)
            self.__actor.save_network(actorFileFullName)

            criticFileName = '%s_critic_%d.h5' % (self.__modelNamePrefix, self.__trainingCounter)
            criticFileFullName = self.__path + '/' + criticFileName
            self.logger.info("Saving Network: %s", criticFileFullName)
            self.__critic.save_network(criticFileFullName)

            self._SetModelFile(actorFileFullName, criticFileFullName)

        if self.__trainingCounter % self.__saveModelPersistNum == 0:
            saveCfg = {}
            with open(loadModelConfigFile, 'rb') as file:
                jsonstr = file.read()
                saveCfg = json.loads(str(jsonstr, encoding='utf-8'))
                self.__lastActorModel1  = saveCfg.get('last_actor_model1')
                self.__lastActorModel2  = saveCfg.get('last_actor_model2')
                self.__lastActorModel3  = saveCfg.get('last_actor_model3')
                self.__lastCriticModel1 = saveCfg.get('last_critic_model1')
                self.__lastCriticModel2 = saveCfg.get('last_critic_model2')
                self.__lastCriticModel3 = saveCfg.get('last_critic_model3')

                currentActorFile  = ''
                currentCriticFile = ''
                if self.__modelIndex == 0:
                    currentActorFile              = self.__lastActorModel1
                    currentCriticFile             = self.__lastCriticModel1
                    self.__lastActorModel1        = actorFileName
                    self.__lastCriticModel1       = criticFileName
                    saveCfg['last_actor_model1']  = actorFileName
                    saveCfg['last_critic_model1'] = criticFileName
                elif self.__modelIndex == 1:
                    currentActorFile              = self.__lastActorModel2
                    currentCriticFile             = self.__lastCriticModel2
                    self.__lastActorModel2        = actorFileName
                    self.__lastCriticModel2       = criticFileName
                    saveCfg['last_actor_model2']  = actorFileName
                    saveCfg['last_critic_model2'] = criticFileName
                elif self.__modelIndex == 2:
                    currentActorFile              = self.__lastActorModel3
                    currentCriticFile             = self.__lastCriticModel3
                    self.__lastActorModel3        = actorFileName
                    self.__lastCriticModel3       = criticFileName
                    saveCfg['last_actor_model3']  = actorFileName
                    saveCfg['last_critic_model3'] = criticFileName

                self.__modelIndex += 1
                if self.__modelIndex > 2:
                    self.__modelIndex = 0

                if currentActorFile != '':
                    currentActorFile = self.__path + '/' + currentActorFile
                    if os.path.exists(currentActorFile):
                        os.remove(currentActorFile)

                if currentCriticFile != '':
                    currentCriticFile = self.__path + '/' + currentCriticFile
                    if os.path.exists(currentCriticFile):
                        os.remove(currentCriticFile)

                saveCfg['actor_model_checkpoint_path']  = actorFileName
                saveCfg['critic_model_checkpoint_path'] = criticFileName

                self._SetModelFile(actorFileFullName, criticFileFullName)

            with open(loadModelConfigFile, 'w') as file:
                json.dump(saveCfg, file)

        elif self.__trainingCounter % self.__saveModelNum == 0:
            if os.path.exists(self.__lastSavedActorModelFile):
                os.remove(self.__lastSavedActorModelFile)

            if os.path.exists(self.__lastSavedCriticModelFile):
                os.remove(self.__lastSavedCriticModelFile)

            self.__lastSavedActorModelFile  = actorFileFullName
            self.__lastSavedCriticModelFile = criticFileFullName

    def _DoAction(self, actions, sleep):
        self.__agentEnv.DoActionForImitationLearning(actions, sleep, True)
        #time.sleep(sleep)

    def _GetActionsFromPPO(self, curState):
        # Rotate(-1~1)/Move Direction(0~1)/Move Distance(0~1)/Fire(0~1, >0.5 means fire)
        # Epsilon decay.
        if self.__epsilon > FINAL_EPSILON:
            self.__epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

        #inputState = curState.reshape(1, curState.shape[0])
        actions = self.__actor.get_action(curState)
        #actions = [0.1, 0.1, 0.1, 0.1]

        #self.logger.info(curState)
        noise_t    = np.zeros([1, ACTION_DIM])
        newActions = np.zeros([1, ACTION_DIM])

        if self.__bTraining == 1 or self.__bLocalTraining == 1:
            # OU noise
            #noise_t[0][0] = max(self.__epsilon, 0) * self.__OU.function(actions[0][0],  0.0 , 0.60, 0.40)
            #noise_t[0][1] = max(self.__epsilon, 0) * self.__OU.function(actions[0][1],  0.5 , 0.10, 0.40)
            #noise_t[0][2] = max(self.__epsilon, 0) * self.__OU.function(actions[0][2],  0.5 , 0.10, 0.40)
            #noise_t[0][3] = max(self.__epsilon, 0) * self.__OU.function(actions[0][3],  0.5 , 0.10, 0.40)

            #newActions[0][0] = actions[0][0] + noise_t[0][0]
            #newActions[0][1] = actions[0][1] + noise_t[0][1]
            #newActions[0][2] = actions[0][2] + noise_t[0][2]
            #newActions[0][3] = actions[0][3] + noise_t[0][3]
            
            #orig:
            #newActions[0] = actions[0] + np.random.normal(loc=0, scale=NOISE, size=actions[0].shape)

            # different normal noise, for rotation standard deviation is 0.4, for the others standard deviation is 1.0
            #rotateNoise = np.random.normal(loc=0, scale=0.4)
            #otherNoise  = np.random.normal(loc=0, scale=1.0, size=3)
            #sign = np.random.randint(2, size=4)
            #if sign[0] == 1:
            #    rotateNoise = -rotateNoise

            #if sign[1] == 1:
            #    otherNoise[0] = -otherNoise[0]

            #if sign[2] == 1:
            #    otherNoise[1] = -otherNoise[1]

            #if sign[3] == 1:
            #    otherNoise[2] = -otherNoise[2]

            #self.logger.info("Noise: {0:1.3f}, {1:1.3f}, {2:1.3f}, {3:1.3f}".format(
            #    rotateNoise, otherNoise[0], otherNoise[1], otherNoise[2]))

            #newActions[0][0] = actions[0][0] + max(self.__epsilon, 0) * rotateNoise
            #newActions[0][1] = actions[0][1] + max(self.__epsilon, 0) * otherNoise[0]
            #newActions[0][2] = actions[0][2] + max(self.__epsilon, 0) * otherNoise[1]
            #newActions[0][3] = actions[0][3] + max(self.__epsilon, 0) * otherNoise[2]

            rotateNoise = np.random.normal(loc=0, scale=1.0)
            otherNoise  = np.random.normal(loc=0, scale=0.5, size=3)
            self.logger.info("Noise: {0:1.2f}, {1:1.2f}, {2:1.2f}, {3:1.2f}".format(
                rotateNoise, otherNoise[0], otherNoise[1], otherNoise[2]))

            newActions[0][0] = actions[0][0] + max(self.__epsilon, 0) * rotateNoise
            newActions[0][1] = actions[0][1] + max(self.__epsilon, 0) * otherNoise[0]
            newActions[0][2] = actions[0][2] + max(self.__epsilon, 0) * otherNoise[1]
            newActions[0][3] = actions[0][3] + max(self.__epsilon, 0) * otherNoise[2]
        else:
            newActions = actions

        #self.logger.info(curState)
        self.logger.info("PPO actions: {0:1.2f}, {1:1.2f}, {2:1.2f}, {3:1.2f}".format(
            actions[0][0], actions[0][1], actions[0][2], actions[0][3]))

        #if newActions[0][0] != actions[0][0]:
        #    self.logger.info("PPO_noise actions: {0:1.3f}, {1:1.3f}, {2:1.3f}, {3:1.3f}".format(
        #        newActions[0][0], newActions[0][1], newActions[0][2], newActions[0][3]))

        if newActions[0][0] < -1:
            newActions[0][0] = -1
        elif newActions[0][0] > 1:
            newActions[0][0] = 1

        if newActions[0][1] < 0:
            newActions[0][1] = 0
        elif newActions[0][1] > 1:
            newActions[0][1] = 1

        if newActions[0][2] < 0:
            newActions[0][2] = 0
        elif newActions[0][2] > 1:
            newActions[0][2] = 1

        if newActions[0][3] < 0:
            newActions[0][3] = 0
        elif newActions[0][3] > 1:
            newActions[0][3] = 1

        if newActions[0][0] != actions[0][0]:
            self.logger.info("PPO_noise actions: {0:1.2f}, {1:1.2f}, {2:1.2f}, {3:1.2f}".format(
                newActions[0][0], newActions[0][1], newActions[0][2], newActions[0][3]))

        return newActions[0], actions[0]

    def _SwitchAmmo(self, currentState):
        if self.__switchAmmoCount < self.__switchAmmoWaiting:
            return

        # Switch ammo if ammo is less than 5.
        # Todo: the total ammo should be taken into consideration.
        if currentState.currentAmmo < 10 and currentState.lastAmmo != 0:
            self.__agentEnv.SwitchAmmo()
            self.__switchAmmoCount = 0

    def _FrameStep(self, actions):
        # Release touch if the last action is forward/move left/move right/backward, beacuse
        # their actions are down, not click.
        # self._ReleaseTouch(action)

        self._DoAction(actions, self.__action_sleep)
        
        timeNow = time.time()
        timePassed = timeNow - self.__lastFrameTime
        if timePassed < self.__timePerFrame:
            timeDelay = self.__timePerFrame - timePassed
            time.sleep(timeDelay)
            #self.logger.info("Sleep: %fms" %(timeDelay))
        else:
            overdTime = timePassed - self.__timePerFrame
            #self.logger.info("Over: %fms" %(overdTime))
            # if overdTime > self.__timePerFrame/5.0:
            #     self.logger.warn('frame overtime: {0} ms'.format(overdTime * 1000))

        ppoState, reward, terminal, curGameState = self.__agentEnv.GetStateForImitationLearning()

        self.__lastFrameTime = time.time()

        return ppoState, reward, terminal, curGameState

    def _ReleaseTouch(self, action):
        # Only turn left and turn right need release.
        if self.__lastActions >= 0 and self.__lastActions <= 3:
            if action > 3:
                self.__agentEnv.ReleaseTouch()

    def _GetBatch(self):
        while self.__replayBuffer.count() < self.__minBatchSize:
            curState = self._ConvertProcessBuffer()
            actions, pred_actions = self._GetActionsFromPPO(curState)

            if self.__bEposideStart == True:
                self.__nSkipCounter += 1

            # Get next state
            self.__nextObservation, self.__reward, self.__terminal, curGameState = self._FrameStep(actions)
            self._AddProcessBuffer(self.__nextObservation)

            if self.__terminal is True:
                break

            sample                = dict()
            sample['state']       = curState
            sample['action']      = actions
            sample['pred_action'] = pred_actions
            sample['reward']      = self.__reward
            sample['terminal']    = self.__terminal
            #sample['next_state'] = self._ConvertProcessBuffer() # self.__nextObservation

            # Skip 10 frames at the beginning
            if self.__nSkipCounter < 10:
                self.logger.info("discard ********* Using PPO action")
                self.__lastActions = actions
                continue

            if self.__bTraining == 1 and self.__bLocalTraining == 0:
                if self.__usingTCP == 1:
                    self._SendSampleToTrainingAIByTCP(sample)
                else:
                    self._SendSampleToTrainingAIByTbus(sample)

            self.__replayBuffer.add(
                sample['state'],
                sample['action'],
                sample['pred_action'],
                sample['reward'],
                sample['terminal']
                )

            self.__lastActions = actions

        batch = self.__replayBuffer.get_batch()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        #print(pred)
        #print(pred.shape)
        #pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def _RunOneStep(self):
        self.__observeNum += 1
        self.__syncNetworkCount += 1

        obs, action, pred, reward = self._GetBatch()

        if len(obs) <= 0:
            self.logger.info("There is no samples for training.")
            return

        if self.__bLocalTraining == 1:
            self._LocalTraining(obs, action, pred, reward)

    def _LocalTraining(self, obs, action, pred, reward):
        self.__trainingCounter += self.__ppoEpoch

        old_prediction = pred
        pred_values    = self.__critic.model.predict(obs)
        #advantage      = pred_values - old_prediction
        advantage      = reward - pred_values

        actor_loss  = []
        critic_loss = []
        for e in range(self.__ppoEpoch):
            actor_loss.append(self.__actor.model.train_on_batch([obs, advantage, old_prediction], [action]))
            critic_loss.append(self.__critic.model.train_on_batch([obs], [reward]))

        self.__gradient_steps += 1

        #if (self.__trainingCounter % 10) == 0:
        current_reward_sum, mean_over_time_reward = self.__replayBuffer.get_total_reward()
        self.logger.info("Training {0}: actor_loss({1:2.4f}) critic_loss({2:2.4f}), mean_reward({3:2.4f})".format(
            self.__trainingCounter, np.mean(actor_loss), np.mean(critic_loss),
            mean_over_time_reward, current_reward_sum))

        self._SaveModel(self._loadModelConfigFile, self.__localTrainingPath)
        return

    def _SendSampleToTrainingAIByTCP(self, sample):
        self.__agentEnv.SendSampleToTrainingByTCP(sample)

    def _SendSampleToTrainingAIByTbus(self, sample):
        self.agentAPI.SendSampleToTrainingAI(sample)

    def _FindLatestNetworkFile(self, modelFilePath):
        maxActorCount  = 0
        maxCriticCount = 0
        actorFile      = ''
        criticFile     = ''

        list = os.listdir(modelFilePath)
        for i in range(0,len(list)):
            path = os.path.join(modelFilePath, list[i])

            jsonFile = path.find('json')
            if jsonFile >= 0:
                continue

            if os.path.isfile(path):
                countStr = re.sub("\D", "", path)
                if countStr == '':
                    continue

                count = int(countStr)
                count = int((count-5) / 10)

                isActor = path.find('actor')
                if isActor >= 0:
                    if count > maxActorCount:
                        actorFile = path
                        maxActorCount = count

                isCritic = path.find('critic')
                if isCritic >= 0:
                    if count > maxCriticCount:
                        criticFile = path
                        maxCriticCount = count

        return actorFile, criticFile, maxActorCount

    def _SyncNetworkFromLocal(self, modelFilePath):
        # Update target network and eval network from trained weights
        if self.__syncNetworkCount < self.__updateAgent:
            return

        self._SyncNetworkFromFile(modelFilePath)

    def _SyncNetworkFromFile(self, modelFilePath):
        actorNewNetwork, criticNewNetwork, count = self._FindLatestNetworkFile(modelFilePath)

        if self.__actorNetwork != None and self.__actorNetwork != actorNewNetwork:
            self.__actorNetwork = actorNewNetwork
            if self.__actorNetwork != None:
                self.__actor.load_network(self.__actorNetwork)
                self.logger.info("observe{0} update actor AI: {1} \n".format(
                    self.__observeNum, self.__actorNetwork))
                self.__trainingCounter = count

            self.__syncNetworkCount = 0

        if self.__criticNetwork != None and self.__criticNetwork != criticNewNetwork:
            self.__criticNetwork = criticNewNetwork
            if self.__criticNetwork != None:
                self.__critic.load_network(self.__criticNetwork)
                self.logger.info("observe{0} update critic AI: {1} \n".format(
                    self.__observeNum, self.__criticNetwork))
                self.__trainingCounter = count

            self.__syncNetworkCount = 0

    def _SyncNetworkFromTCP(self):
        receivedActorFile, receivedCriticFile = self.__agentEnv.ReceiveModelFile()

        if receivedActorFile == '' or receivedCriticFile == '':
            return

        self._SyncNetworkFromFile(self.__tcpModelFilePath)

        if receivedActorFile != self.__tcpReceivedActorFile:
            if os.path.exists(self.__tcpReceivedActorFile):
                os.remove(self.__tcpReceivedActorFile)

        [dirname, filename]=os.path.split(receivedActorFile)
        self.__tcpReceivedActorFile = self.__tcpModelFilePath + '/' + filename

        if receivedActorFile != self.__tcpReceivedCriticFile:
            if os.path.exists(self.__tcpReceivedCriticFile):
                os.remove(self.__tcpReceivedCriticFile)

        [dirname, filename]=os.path.split(receivedCriticFile)
        self.__tcpReceivedCriticFile = self.__tcpModelFilePath + '/' + filename
        return

    def _SyncNetwork(self):
        self.logger.info("Sync network begin...")
        if self.__usingTCP == 0:
            self._SyncNetworkFromLocal(self.__path)
        elif self.__usingTCP == 1:
            self._SyncNetworkFromTCP()
        self.logger.info("Sync network finished!")

    def OnEpsiodeStart(self):
        actions = [0, 0, 0, 0]

        self.__bHasEnemy       = False
        self.__switchAmmoCount = 0
        self.__attackedCount   = 0
        self.__lastFrameTime   = time.time()

        # Skip the beginning 10 frames.
        minFrames = max(10, self.__numFrames)
        for i in range(minFrames):
            self.__nextObservation, self.__reward, self.__terminal, _ = self._FrameStep(actions)
            self._AddProcessBuffer(self.__nextObservation)

        self.__screenWidth, self.__screenHeight = self.__agentEnv.GetSreenSize()
        self.__bEposideStart = True

    def OnEpsiodeOver(self):
        # Do turn right to release down.
        actions = [0, 0, 0, 0]
        self._DoAction(actions, self.__action_sleep)
        self.__agentEnv.Reset()

        if self.__bTraining == 1:
            self._SyncNetwork()

        self.__bEposideStart = False
        self.__nSkipCounter = 0

    def TrainOneStep(self):
        self._RunOneStep()

    def TestOneStep(self):
        self._RunOneStep()

    def IsUsingTCP(self):
        return self.__usingTCP

    def _LoadTbusConfig(self, cfgPath):
        tbusArgs = {}
        LOG.info('cfgPath is %s' % (cfgPath))

        if os.path.exists(cfgPath):
            config = Configparser.ConfigParser(strict=False)
            config.read(cfgPath)
            tbusArgs['GameRecognizeAddr'] = config.get('BusConf', 'GameRecognizeAddr')
            tbusArgs['GameUIAutoAddr'] = config.get('BusConf', 'GameUIAutoAddr')
            tbusArgs['GameAgentAddr'] = config.get('BusConf', 'GameAgentAddr')
        else:
            LOG.error('Tbus Config File not exist in {0}'.format(cfgPath))

        return tbusArgs

    def _DumpPicture(self, image, width, height, name):
        img = cv2.resize(image, (width, height))
        cv2.imwrite(name, img)
