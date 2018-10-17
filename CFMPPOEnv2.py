# ----------------------------------------------
# Project: Deep Deterministic Policy Gradient
# Author: benethuang
# Date: 2018.8.17
# ----------------------------------------------

import time
import os
import configparser
import logging
import numpy as np
import cv2
import json
import math

from actionmanager import ActionController
from agentenv.GameEnv import GameEnv
from AgentAPI import AgentAPIMgr
from CFM_DDPG.TCPComm import TCPThread

TaskGrenadeButton = 1
TaskMissileButton = 2
TaskTrackGunButton = 3
TaskHeroBlood = 4
TaskBossBlood = 5
TaskBossBloodNum = 6
TaskScoreNum = 7
TaskJumpButton = 8
TaskHeroHurt = 9 

ACTION_CFG_FILE  = '../cfg/action.json'
ENV_CFG_FILE     = '../cfg/env.ini'
COMMON_CFG_FILE  = '../cfg/common.json'
TCP_CFG_FILE     = '../cfg/tcp.json'
TCP_MODEL_PATH   = '../data/AI/TCP'
CFM_ENV_CFG_FILE = '../data/AI/CFMEnv.json'

path = '../data/CFMGame/'

LOG = logging.getLogger('agent')

class CurrentState:
    def __init__(self):
        self.currentAmmo     = -1
        self.currentKills    = -1
        self.currentHealth   = -1

        # For reward function, we will save ammo/kills/health of last frame.
        self.lastAmmo = -1
        self.lastKills = -1
        self.lastHealth = -1

        # Check if firing.
        self.bFire = False

        # direction Attacked by enemy.
        self.attackDir = 'None'
        self.lastAttackDir = 'None'

        # my group, 0 terrorist, 1 counter-terrorist, -1 invalid
        self.groupId = -1
        self.classGR = 0
        self.classBL = 0

        # human info in current scene
        self.humanInfo = None
        self.lastHumanInfo = None

        # small map info
        self.smallMapInfo = None
        self.lastSmallMapInfo = None

        # Swap counter
        self.swapCounter = 0


    def SetSmallMap(self, info):
        self.smallMapInfo = info

    def SetAmmo(self, ammo):
        self.currentAmmo = ammo

    def SetKills(self, kills):
        self.currentKills = kills

    def SetHealth(self, health):
        self.currentHealth = health

    def SetAttackDir(self, dir):
        self.attackDir = dir

    def SetGroup(self, groupId):
        if groupId == -1 and self.currentHealth == 0:
            return

        # 1 for GR, 0 for BL
        if groupId == 1:
            self.classGR += 1
        elif groupId == 0:
            self.classBL += 1

        #if self.classGR > self.classBL:
        #    self.groupId = 1
        #elif self.classBL > self.classGR:
        #    self.groupId = 0
        #else:
        #    self.groupId = groupId

        self.groupId = groupId

    def SetHumanInfo(self, info):
        self.humanInfo = info

    def SwapFrame(self):
        self.swapCounter += 1

        if self.currentAmmo < self.lastAmmo:
            self.bFire = True
        else:
            self.bFire = False

        self.lastAmmo      = self.currentAmmo
        self.lastHealth    = self.currentHealth
        self.lastKills     = self.currentKills
        self.lastAttackDir = self.attackDir
        self.lastHumanInfo = self.humanInfo

        #swapSmall = self.swapCounter % 5
        #if swapSmall == 0:
        #    self.lastSmallMapInfo   = self.smallMapInfo
        self.lastSmallMapInfo = self.smallMapInfo

    def Clear(self):
        self.classBL = 0
        self.classGR = 0


class CFMGameMapSearchEnv(GameEnv):
    def __init__(self):
        GameEnv.__init__(self)

        # Save current state, such as blood/ammo/killnums.
        self.currentState = CurrentState()

        self.__actionController = ActionController.ActionController()
        self.__actionController.Initialize(ACTION_CFG_FILE)

        self._LoadEnvParams()
        self._LoadCommonParams()
        self.Reset()

        self.terminal = True

        # Save last action. If current action is the same last action and current current action
        # is forward/backward/move left/move right, we do nothing in DoAction.
        self.__lastActions = None

        # Load env parameters
        self._LoadCFMEnvParams()

        # last frame sequence for calculating reward.
        self.__lastRewardFrameSeq = 0

    def _CreateTCPComm(self, logger):
        self.__tcp = TCPThread(logger, TCP_CFG_FILE, TCP_MODEL_PATH, 'TCP communicator')
        self.__tcp.start()

    def Init(self):
        self.agentAPI = AgentAPIMgr.AgentAPIMgr()
        ret = self.agentAPI.Initialize("../cfg/task.json")
        if not ret:
            self.logger.error('Agent API Init Failed')
            return

        ret = self.agentAPI.SendCmd(AgentAPIMgr.MSG_SEND_GROUP_ID, 1)
        if not ret:
            self.logger.error('send message failed')

        # Create thread of tcp communication.
        self._CreateTCPComm(self.logger)

        return True

    def _LoadCFMEnvParams(self):
        # Caculate reward based on map position.
        self.__origPosition      = None
        self.__tmpPosition       = None
        self.__posDecayInit      = 100
        self.__posDecayCounter   = self.__posDecayInit
        self.__stuckDecayInit    = 100
        self.__stuckDecay        = self.__stuckDecayInit
        self.__stuckOrigPosition = None

        # enemy and attack decay that is used by reward caculation.
        self.__enemyDecayInit  = 3
        self.__enemyDecay      = 0
        self.__enemyLastState  = None
        self.__attackDecayInit = 6
        self.__attackDecay     = 0
        self.__attackLastState = None

        # health decrease decay.
        self.__healthDecayInit = 5
        self.__healthDecay     = 0

        # Action for DDPG
        self.__turnDistance = 200

        if os.path.exists(CFM_ENV_CFG_FILE):
            with open(CFM_ENV_CFG_FILE, 'rb') as file:
                jsonstr = file.read()
                commonCfg = json.loads(str(jsonstr, encoding='utf-8'))

                self.__enemyDecayInit  = commonCfg.get('enemy_decay')
                self.__attackDecayInit = commonCfg.get('attack_decay')
                self.__posDecayInit    = commonCfg.get('pos_decay')
                self.__turnDistance    = commonCfg.get('turn_distance')

        # Load mask bitmap of small map for yingdi.
        self.__mapObstacle = cv2.imread("../data/CFM_Data/MapTemplate/SmallMapMask.jpg")
        pixel1 = self.__mapObstacle[165][48]
        pixel2 = self.__mapObstacle[171][46]

        self.__mapScreenWidth  = 210
        self.__mapScreenHeight = 216

        return

    def SendSampleToTrainingByTCP(self, sample):
        self.__tcp.PushSampleToSendQueue(sample)

    def SendMsgToTrainingByTCP(self, msg):
        self.__tcp.SendMsg(msg)

    def ReceiveModelFile(self):
        return self.__tcp.ReceiveModelFile()

    def Finish(self):
        self.__tcp.SetExitFlag(True)
        self.__tcp.join()
        self.agentAPI.Release()

    def GetActionSpace(self):
        return self.__actionController.GetActionNum()

    def _CalcTurnActions(self, action, bPrint = False):
        # turn left/right is start from (900, 378). Max distance is 200
        turnDistance = action * self.__turnDistance

        if action > 0:
            startTurnX = 900
            startTurnY = 378
            endTurnX   = startTurnX + turnDistance
            endTurnY   = startTurnY
        else:
            endTurnX   = 900
            endTurnY   = 378
            startTurnX = endTurnX - turnDistance
            startTurnY = endTurnY

        if bPrint == True:
            self.logger.info("turn: {0}, ({1}, {2})->({3}, {4})".format(action, startTurnX, startTurnY, endTurnX, endTurnY))

        return startTurnX, startTurnY, endTurnX, endTurnY

    def _CalcMoveActions(self, action1, action2, bPrint = False):
        # action1 Move Direction(0~1)
        # action2 Move Distance(0~1)

        # start xy for direction controller
        directionCenterX = 273
        directionCenterY = 527

        # Max up (273, 430), so the distance is about 100.
        maxDistance = 100
        moveDist    = action2 * maxDistance

        # Calculate move direction
        moveAngle = action1 * math.pi * 2

        pi_half     = math.pi / 2
        pi_one      = math.pi
        pi_one_half = math.pi * 3/2
        pi_two      = math.pi * 2
        if moveAngle < pi_half:
            beta = moveAngle
            endX = moveDist * math.cos(beta)
            endY = moveDist * math.sin(beta)
        elif moveAngle == pi_half:
            endX = 0
            endY = moveDist
        elif moveAngle > pi_half and moveAngle < pi_one:
            beta = pi_one - moveAngle
            endX = - moveDist * math.cos(beta)
            endY =   moveDist * math.sin(beta)
        elif moveAngle == pi_one:
            endX = - moveDist
            endY = 0
        elif moveAngle > pi_one and moveAngle < pi_one_half:
            beta = moveAngle - pi_one
            endX = - moveDist * math.cos(beta)
            endY = - moveDist * math.sin(beta)
        elif moveAngle == pi_one_half:
            endX = 0
            endY = -moveDist
        elif moveAngle > pi_one_half and moveAngle < pi_two:
            beta = pi_two - moveAngle
            endX =   moveDist * math.cos(beta)
            endY = - moveDist * math.sin(beta)
        elif moveAngle == pi_two or moveAngle == 0:
            endX = moveDist
            endY = 0
        else:
            self.logger.info("There is wrong angle: {0}".format(moveAngle))
            endX = 0
            endY = 0

        endX = endX + directionCenterX
        endY = endY + directionCenterY

        if bPrint == True:
            self.logger.info("move: {0}, {1}, ({1}, {2})->({3}, {4})".format(action1, action2, directionCenterX, directionCenterY, endX, endY))

        return directionCenterX, directionCenterY, endX, endY

    def DoAction(self, action):
        actionIndex = np.argmax(action)

        if self.__lastActions == actionIndex:
            # last action is between forward and move right.
            if self.__lastActions >= 0 and self.__lastActions <= 3:
                return

        self.__actionController.DoAction(actionIndex)
        self.__lastActions = actionIndex

    def DoActionForImitationLearning(self, actions, sleep=0.1, bPrint = False):
        # action[0] is Rotate(-1~1)
        # action[1] Move Direction(0~1)
        # action[2] Move Distance(0~1)
        # action[3] Fire(0~1, >0.5 means fire)

        # Fire is first. Fire button is on (1101, 555).
        # If using auto firing, just comment out the following code.
        #if actions[3] > 0.5:
        #    self.__actionController.Click(1101, 555, 1, self.__lastRewardFrameSeq)

        startX, startY, endX, endY = self._CalcTurnActions(actions[0])

        # using contact 1 for turn left/right, 100ms duration, need up.
        self.__actionController.SwipeOnce(startX,
                                          startY,
                                          endX,
                                          endY,
                                          1,
                                          self.__lastRewardFrameSeq,
                                          100,
                                          True)
        time.sleep(0.1)

        startX, startY, endX, endY = self._CalcMoveActions(actions[1], actions[2])

        # using contact 0 for move, 100ms duration, don't need up.
        self.__actionController.SwipeOnce(startX,
                                          startY,
                                          endX,
                                          endY,
                                          0,
                                          self.__lastRewardFrameSeq,
                                          50,
                                          False)
        time.sleep(0.1)
        self.__lastActions = actions

    def _ContructStateVector(self, state):
        humanDetect = [[-1, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0]]

        humanNum = len(state.humanInfo)
        # if humanNum > 0:
        #     print('')

        maxHumanNum = min(humanNum, len(humanDetect))
        for i in range(maxHumanNum):
            # normalize, x, y, w, h
            state.humanInfo[i][2] = state.humanInfo[i][2] / self.__screenWidth
            state.humanInfo[i][3] = state.humanInfo[i][3] / self.__screenHeight
            state.humanInfo[i][4] = state.humanInfo[i][4] / self.__screenWidth
            state.humanInfo[i][5] = state.humanInfo[i][5] / self.__screenHeight

            humanDetect[i] = state.humanInfo[i]

        humanDetectVector = np.hstack((humanDetect[0], humanDetect[1], humanDetect[2], humanDetect[3]))

        # My position and viewport position
        myGuysPos = [[0, 0], [0, 0], [0, 0], [0, 0]]
        if state.smallMapInfo != None:
            myPos     = state.smallMapInfo[2]
            myViewPos = state.smallMapInfo[1]

            if state.smallMapInfo[3] != None:
                mapFriendsLoc = state.smallMapInfo[3]
                guysNum = min(len(mapFriendsLoc['x']), 4)
                for i in range(guysNum):
                    myGuysPos[i][0] = (mapFriendsLoc['x'][i] / self.__mapScreenWidth)
                    myGuysPos[i][1] = (mapFriendsLoc['y'][i] / self.__mapScreenHeight)
        else:
            myPos     = None
            myViewPos = None

        # Normalize myPos and myViewPos
        if myPos != None:
            myPos['x'] = myPos['x'] / self.__mapScreenWidth
            myPos['y'] = myPos['y'] / self.__mapScreenHeight
        else:
            myPos      = {}
            myPos['x'] = 0
            myPos['y'] = 0
        
        if myViewPos != None:
            myViewPos['x'] = myViewPos['x'] / self.__mapScreenWidth
            myViewPos['y'] = myViewPos['y'] / self.__mapScreenHeight
        else:
            myViewPos      = {}
            myViewPos['x'] = 0
            myViewPos['y'] = 0

        smallMapVector = [myPos['x'], myPos['y'], myViewPos['x'], myViewPos['y']]
        smallMapMyGuysVector = np.hstack((myGuysPos[0], myGuysPos[1], myGuysPos[2], myGuysPos[3]))

        # attacked direction
        if state.attackDir == 'HurtUpLeft': 
            attackVector = 1
        elif state.attackDir == 'HurtLeftUp':
            attackVector = 2
        elif state.attackDir == 'HurtLeft':
            attackVector = 3
        elif state.attackDir == 'HurtLeftDown':
            attackVector = 4
        elif state.attackDir == 'HurtDownLeft':
            attackVector = 5
        elif state.attackDir == 'HurtDown':
            attackVector = 6
        elif state.attackDir == 'HurtUpRight':
            attackVector = 7
        elif state.attackDir == 'HurtRightUp':
            attackVector = 8
        elif state.attackDir == 'HurtRight':
            attackVector = 9
        elif state.attackDir == 'HurtRightDown':
            attackVector = 10
        elif state.attackDir == 'HurtDownRight':
            attackVector = 11
        elif state.attackDir == 'HurtUp':
            attackVector = 12
        else:
            attackVector = 13

        attackVector = attackVector / 13

        # Noramlize blood, ammo and kills
        normBlood = state.currentHealth / 100
        normAmmo  = state.currentAmmo / 40
        normKills = state.currentKills / 30

        # Construct all info, vector has 41 elements.
        # It includes 1 + 4*6(humanInfo) + 4(myPos, myViewPos) + 
        # 1(attackVector) + 3(blood, ammo, kills) + 2*4(our guys' position in small map).
        vector = np.hstack((state.groupId, humanDetectVector, smallMapVector, 
                            attackVector, normBlood, normAmmo, normKills, smallMapMyGuysVector))

        return vector

    def _ParseGameResult(self, resultDic):
        currentAmmo         = 0
        currentBlood        = 0
        currentKills        = 0
        attackDir           = None
        groupId             = -1
        humanInfo           = None
        smallMapInfo        = None
        gameState           = AgentAPIMgr.GAME_STATE_INVALID
        currentCrossFire    = None
        currentMyGroupState = None

        for taskID, value in resultDic.items():
            if value is not None:
                self._UpdateRegResult(taskID, value)

                # state check task
                if taskID == 1:
                    
                    # Check if win.
                    if value[0][0] == True:
                        gameState = AgentAPIMgr.GAME_STATE_WIN
                    elif value[1][0] == True:
                        gameState = AgentAPIMgr.GAME_STATE_WIN

                    # Check if lose.
                    elif value[2][0] == True:
                        gameState = AgentAPIMgr.GAME_STATE_LOSE
                    elif value[3][0] == True:
                        gameState = AgentAPIMgr.GAME_STATE_LOSE

                    # Check if running game.
                    elif value[4][0] == True:
                        gameState = AgentAPIMgr.GAME_STATE_RUN

                    # Set default
                    else:
                        gameState = AgentAPIMgr.GAME_STATE_START

                # Get blood num
                elif taskID == 3: 
                    # self.logger.error ("Blood: %d" %(value[2]))
                    currentBlood = value[2]
                    if self.currentState.currentHealth < currentBlood and self.currentState.currentHealth > 30:
                        currentBlood = self.currentState.currentHealth

                    if currentBlood > 100:
                        currentBlood = 100

                    if currentBlood < self.currentState.currentHealth:

                        # Because we cannot get correct blood number when blood number is less than 30.
                        if self.currentState.currentHealth > 30:
                            self.__healthDecay = self.__healthDecayInit

                # kill num
                elif taskID == 4: 
                    # self.logger.error ("Kills: %d" %(value[2]))
                    currentKills = value[2]

                # Get my group id, 0 terrorist, 1 couter-terrorist
                #elif taskID == 5:
                #    groupId = value[0]['loc0']
                #    if groupId == 'GR':
                #        groupId = 1
                #    elif groupId == 'BL':
                #        groupId = 0
                #    else:
                #        groupId = -1

                # Get state of CrossFire
                elif taskID == 5:
                    currentCrossFire = value[0]

                # value[0]: direction
                # value[1]: view center position
                # value[2]: my position
                # value[3]: our guys' position
                elif taskID == 6:
                    smallMapInfo = value

                # direction of attacked.
                elif taskID == 7:
                    colorMeanVar = value[0]
                    colorStdVar = value[1]
                    colorMeanVarBesides = value[2]

                    minColorMeanVar = min(colorMeanVar.items(), key=lambda x: x[1])
                    minName = minColorMeanVar[0]
                    if colorMeanVar[minName] < 50 and colorMeanVarBesides[minName] > 50 and colorStdVar[minName] < 50:
                        attackDir = minName
                    else:
                        attackDir = 'None'

                    if self.__healthDecay > 0 and attackDir == 'None':
                        attackDir = minName
                    elif self.__healthDecay <= 0:
                        attackDir = 'None'

                # Get the position of humans in the current scene.
                elif taskID == 8:
                    humanInfo = value[0]

                elif taskID == 9: # ammo num
                    # self.logger.error ("Ammo: %d" %(value[2]))
                    currentAmmo = value[2]

                # Get group id
                elif taskID == 10:
                    currentMyGroupState = value[0]

        self.logger.info('Ammo: %d, Blood: %d, kills: %d, attackDir: %s' %(currentAmmo, currentBlood, currentKills, attackDir))

        self.__healthDecay -= 1

        self.currentState.SetAmmo(currentAmmo)
        self.currentState.SetHealth(currentBlood)
        self.currentState.SetKills(currentKills)
        self.currentState.SetAttackDir(attackDir)
        self.currentState.SetHumanInfo(humanInfo)
        self.currentState.SetSmallMap(smallMapInfo)

        # We should set groupId in the end, because we use blood to help check if
        # the group id is valid.
        self._SetGroup(currentCrossFire, currentMyGroupState)

        if gameState == AgentAPIMgr.GAME_STATE_RUN and currentBlood <= 0:
            gameState = AgentAPIMgr.GAME_STATE_INVALID

        stateVector = self._ContructStateVector(self.currentState)

        # if self.currentState.lastHumanInfo is not None and len(self.currentState.lastHumanInfo) > 0:
        #     print(self.currentState.lastHumanInfo)

        #self.logger.info(stateVector)
        return gameState, stateVector

    # Set Group
    def _SetGroup(self, currentCrossFire, currentMyGroupState):
        if currentCrossFire is None or currentMyGroupState is None:
            return

        for value in currentCrossFire.values():
            if value is None:
                return

        for keyTer in currentMyGroupState.keys():
            if currentMyGroupState[keyTer] is not None:
                if keyTer == 'BL':
                    self.currentState.SetGroup(0)
                if keyTer == 'GR':
                    self.currentState.SetGroup(1)

    def ChangeGameState(self, gameState):
        if gameState == AgentAPIMgr.GAME_STATE_LOSE:
            self.terminal = True
        elif gameState == AgentAPIMgr.GAME_STATE_WIN:
            self.terminal = True
        elif gameState == AgentAPIMgr.GAME_STATE_RUN:
            self.terminal = False
        elif gameState == AgentAPIMgr.GAME_STATE_START:
            self.terminal = True
        elif gameState == AgentAPIMgr.GAME_STATE_INVALID:
            self.terminal = True
        else:
            self.logger.error('error game state')

    def ReleaseTouch(self):
        self.__actionController.Click(271, 524, 0)
        self.__actionController.Click(271, 524, 1)

    def SwitchAmmo(self):
        self.__actionController.Click(854, 684)
        self.logger.info('switch ammo')

    def Jump(self):
        self.__actionController.Click(1231, 371)
        self.logger.info('jump')

    def Forward(self):
        self.__actionController.Click(275, 448)

    def _DumpPicture(self, image, frameSeq, reward=0):
        img = cv2.resize(image, (1280, 720))

        path = "../data/AI/frame/"
        if os.path.exists(path) == False:
            os.mkdir(path)

        fileName = "%sframe%d_reward%.3f.jpg" % (path, frameSeq, reward)
        self.logger.info("%s" %(fileName))
        cv2.imwrite(fileName, img)

    def GetStateForImitationLearning(self):
        gameResult = None
        gameState = AgentAPIMgr.GAME_STATE_START

        while True:
            gameResult = self.agentAPI.GetInfo(AgentAPIMgr.GAME_RESULT_INFO)
            if gameResult is not None:
                break
            else:
                time.sleep(0.002)

        resultDic = gameResult['result']
        gameState, stateVector = self._ParseGameResult(resultDic)

        image = gameResult['image']
        img = image

        reward = self._CalculateReward(
            self.__lastActions,
            self.currentState.lastHumanInfo,
            self.currentState.groupId,
            self.currentState.currentHealth,
            self.currentState.currentKills,
            self.currentState.currentAmmo,
            self.currentState.lastHealth,
            self.currentState.lastKills,
            self.currentState.lastAmmo,
            self.currentState.lastAttackDir,
            smallMapInfo = self.currentState.smallMapInfo,
            lastSmallMapInfo = self.currentState.lastSmallMapInfo
        )

        frameSeq = gameResult['frameSeq']
        if (self.__lastRewardFrameSeq + 100) < frameSeq:
            reward = 0.0

        self.__lastRewardFrameSeq = frameSeq
        #self._DumpPicture(img, frameSeq, reward)

        self.ChangeGameState(gameState)
        self.currentState.SwapFrame()

        return stateVector, reward, self.terminal, self.currentState

    def GetState(self):
        gameResult = None
        gameState = AgentAPIMgr.GAME_STATE_START

        while True:
            gameResult = self.agentAPI.GetInfo(AgentAPIMgr.GAME_RESULT_INFO)
            if gameResult is not None:
                break
            else:
                time.sleep(0.002)

        resultDic = gameResult['result']
        gameState, stateVector = self._ParseGameResult(resultDic)

        # Switch ammo if ammo is less than 5.
        # Todo: the total ammo should be taken into consideration.
        # Todo: this code should be executed in AI model, not here.
        # But current DQN doesn't has interface to receive current state
        # of CFM.
        # if currentAmmo < 5 and self.currentState.lastAmmo != 0:
        #     self.SwitchAmmo()

        image = gameResult['image']
        img = self._PreprocessImg(image)
        # img = image
        reward = self._CalculateReward(
            self.__lastActions,
            self.currentState.lastHumanInfo,
            self.currentState.groupId,
            self.currentState.currentHealth,
            self.currentState.currentKills,
            self.currentState.currentAmmo,
            self.currentState.lastHealth,
            self.currentState.lastKills,
            self.currentState.lastAmmo,
            self.currentState.lastAttackDir,
            smallMapInfo = self.currentState.smallMapInfo,
            lastSmallMapInfo = self.currentState.lastSmallMapInfo
        )

        self.ChangeGameState(gameState)

        if reward != 0:
            self.logger.info('Reward: {0}, group: {1}'.format(reward, self.currentState.groupId))
        
        self.currentState.SwapFrame()
        return stateVector, reward, self.terminal

    def Reset(self):
        self.__isTrainable = True
        self.__actionController.Reset()
        self.currentState.Clear()

    def IsTrainable(self):
        return self.__isTrainable

    def GetSreenSize(self):
        return self.__screenWidth, self.__screenHeight

    def _PreprocessImg(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img[self.__beginRow:self.__endRow, self.__beginColumn:self.__endColumn]
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]
        if imgWidth < imgHeight:
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)

        img = cv2.resize(img, (176, 108))
        return img

    def _UpdateRegResult(self, taskID, resultValue):
        pass

    def _LoadEnvParams(self):
        if os.path.exists(ENV_CFG_FILE):
            config = configparser.ConfigParser()
            config.read(ENV_CFG_FILE)

            self.__beginColumn = config.getint('ImageCut', 'cut_cols_begin')
            self.__beginRow = config.getint('ImageCut', 'cut_rows_begin')
            self.__cutWidth = config.getint('ImageCut', 'image_cut_width')
            self.__cutHeight = config.getint('ImageCut', 'image_cut_height')
            self.__endColumn = self.__beginColumn + self.__cutWidth
            self.__endRow = self.__beginRow + self.__cutHeight

            self.__initScore = config.getfloat('RewardRule', 'init_score')

            self.__winReward = config.getfloat('RewardRule', 'win_reward')
            self.__loseReward = config.getfloat('RewardRule', 'lose_reward')

    def _LoadCommonParams(self):
        if os.path.exists(COMMON_CFG_FILE):
            with open(COMMON_CFG_FILE, 'rb') as file:
                jsonstr = file.read()
                commonCfg = json.loads(str(jsonstr, encoding='utf-8'))
                self.__screenWidth = commonCfg.get('ScreenCaptureWidth')
                self.__screenHeight = commonCfg.get('ScreenCaptureHeight')
        else:
            self.logger.error('No common param file.')

        return

    def _CalculateMoveReward1(self, smallMapInfo, lastSmallMapInfo):
        reward = 0.0
        if smallMapInfo != None and lastSmallMapInfo != None:
            myPos     = smallMapInfo[2]
            lastMyPos = lastSmallMapInfo[2]

            if myPos != None and lastMyPos != None:
                shiftPos  = math.fabs(myPos['x'] - lastMyPos['x']) + math.fabs(myPos['y'] - lastMyPos['y'])
                shiftPos  = shiftPos / 10
                reward    += shiftPos
                # self.logger.info ("shiftPos: %f" %(shiftPos))
        return reward

    def _CalculateMoveReward2(self, action, smallMapInfo, lastSmallMapInfo, bPrint = False):
        # Rotate(-1~1)/Move Direction(0~1)/Move Distance(0~1)/Fire(0~1, >0.5 means fire)

        reward = 0.0

        # if action is no action/turn left/turn right there is no reward.
        if action == 6 or action == 4 or action == 5:
            return reward

        if self.__posDecayCounter <= 0:
            self.__posDecayCounter = self.__posDecayInit # reset decay parameter.
            self.__origPosition = self.__tmpPosition

        self.__posDecayCounter -= 1

        # First position
        if smallMapInfo != None and self.__origPosition == None:
            self.__origPosition = smallMapInfo

        if smallMapInfo != None and self.__origPosition != None:
            myPos     = smallMapInfo[2]
            lastMyPos = self.__origPosition[2]

            if myPos != None and lastMyPos != None:
                shiftPos    = math.fabs(myPos['x'] - lastMyPos['x']) + math.fabs(myPos['y'] - lastMyPos['y'])
                decay       = self.__posDecayInit - self.__posDecayCounter
                shiftReward = shiftPos / 100 / (decay)
                if shiftReward > 0.002:
                    reward += shiftReward

                if bPrint == True:
                    self.logger.info ("shiftPos: %f, reward: %f, (%.2f, %.2f) -> (%.2f, %.2f), decay: %d" %
                                      (shiftPos, reward, lastMyPos['x'], lastMyPos['y'], myPos['x'], myPos['y'], decay))

                # Record valid position.
                self.__tmpPosition = smallMapInfo

        if smallMapInfo != None and lastSmallMapInfo != None:
            myPos     = smallMapInfo[2]
            lastMyPos = lastSmallMapInfo[2]

            if myPos != None and lastMyPos != None:
                shiftPos    = math.fabs(myPos['x'] - lastMyPos['x']) + math.fabs(myPos['y'] - lastMyPos['y'])

                # If stuck, there is no effect of move action.
                if shiftPos < 4 and action < 4:
                    reward = -0.01

                    if bPrint == True:
                        self.logger.info ("movePos: %f, reward: -0.01, (%.2f, %.2f) -> (%.2f, %.2f)" %(shiftPos, lastMyPos['x'], lastMyPos['y'], myPos['x'], myPos['y']))

        return reward

    def _CalculateAttackedDirReward(self, action, attackDir, bLastAttack, bPrint = False):
        reward = 0.0

        if attackDir == '' or attackDir == 'None':
            return reward, False

        distUnit = self.__turnDistance / 6
        actionDist = abs(self.__turnDistance * action)

        if action < -0.1:
            turn_action = 4
            reward 
        elif action > 0.1:
            turn_action = 5
        else:
            turn_action = 0

        if attackDir == 'HurtUp':
            expectedAction = 0
            expectedDist   = distUnit * 0
            self.__attackDecayInit = 0
        elif attackDir == 'HurtUpLeft': 
            expectedAction = 4
            expectedDist   = distUnit * 3
            self.__attackDecayInit = 1
        elif attackDir == 'HurtLeftUp':
            expectedAction = 4
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 2
        elif attackDir == 'HurtLeft':
            expectedAction = 4
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 3
        elif attackDir == 'HurtLeftDown':
            expectedAction = 4
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 4
        elif attackDir == 'HurtDownLeft':
            expectedAction = 4
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 5
        elif attackDir == 'HurtDown':
            expectedAction = 4
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 6
        elif attackDir == 'HurtUpRight':
            expectedAction = 5
            expectedDist   = distUnit * 3
            self.__attackDecayInit = 1
        elif attackDir == 'HurtRightUp':
            expectedAction = 5
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 2
        elif attackDir == 'HurtRight':
            expectedAction = 5
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 3
        elif attackDir == 'HurtRightDown':
            expectedAction = 5
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 4
        elif attackDir == 'HurtDownRight':
            expectedAction = 5
            expectedDist   = distUnit * 6
            self.__attackDecayInit = 5
        else:
            expectedAction = -1
            expectedDist   = 0
            return reward, False

        if expectedAction == turn_action:
            distGap = abs(actionDist - expectedDist)
            if distGap < 30:
                reward = 0.2
            else:
                reward = 0.2 * (1 - distGap / self.__turnDistance)

            if bPrint == True:
                self.logger.info ("attack: %s, exp: %d, action: %.3f, reward: %.2f" %(attackDir, expectedAction, action, reward))

        elif expectedAction == 4 or expectedAction == 5 or expectedAction == 0:
            reward = -0.2
            if bPrint == True:
                self.logger.info ("attack: %s, exp: %d, action: %.3f, reward: %.2f" %(attackDir, expectedAction, action, reward))

        if bLastAttack is False:
            self.__attackDecay        = self.__attackDecayInit
            self.__attackLastState    = attackDir
            self.__lastExpectedAction = expectedAction
            self.__lastExpectedDist   = expectedDist
        else:
            if bPrint == True:
                self.logger.info ("****last attack")

        return reward, True

    def _CalculateEnemyReward(self, action, humanInfo, myGroupId, bLastEnemy, bPrint = False):
        reward = 0.0
        if humanInfo is None or myGroupId == -1:
            return reward, False

        num = len(humanInfo)

        if num <= 0:
            return reward, False

        if action <= -0.1:
            turn_action = 4
        elif action >= 0.1:
            turn_action = 5
        else:
            turn_action = 0

        #actionDist = abs(self.__turnDistance * action)

        # Unit of distance is pixel.
        distance = 10000.0
        highRiskIndex = -1
        highRiskX = -1
        highRiskY = -1

        # Find the highest risk of enemy
        for i in range(num):
            if humanInfo[i][0] != myGroupId:
                x = humanInfo[i][2] * self.__screenWidth
                y = humanInfo[i][3] * self.__screenHeight
                w = humanInfo[i][4] * self.__screenWidth
                h = humanInfo[i][5] * self.__screenHeight

                x = int(x + w / 2)
                y = int(y + h / 2)

                i_dist = math.sqrt(x*x + y*y)
                if i_dist < distance:
                    distance = i_dist
                    highRiskIndex = i
                    highRiskX = x
                    highRiskY = y

        if highRiskIndex == -1:
            return reward, False

        # There is enemy detected by YOLO.
        # self.__enemyDecay = self.__enemyDecayInit

        # Normalize
        x_normal = float(highRiskX) / float(self.__screenWidth)

        # We use the last state of enemies, but there is no enemy.
        if bLastEnemy is True:
            if x_normal <= 0.45 and turn_action == 4:
                reward = abs(0.5 + (action / 2))
            elif x_normal >= 0.55 and turn_action == 5:
                reward = abs(action - 0.5)
            elif x_normal > 0.45 and x_normal < 0.55 and turn_action == 0:
                reward = 0.5
            else:
                reward -= 0.2

            if bPrint:
                self.logger.info ("****last x_normal: %.3f, action: %.2f, reward: %.2f" %(x_normal, action, reward))

            return reward, False

        # 4 means turn-left, 5 means turn-right, 0 means no turn.
        if x_normal <= 0.45 and turn_action == 4:
            actionNormal = abs(0.5 + (action / 2))
            gap = abs(x_normal - actionNormal)
        elif x_normal >= 0.55 and turn_action == 5:
            actionNormal = abs(0.5 + (action / 2))
            gap = abs(x_normal - actionNormal) 
        elif x_normal > 0.45 and x_normal < 0.55 and turn_action == 0:
            gap = 0.1
        else:
            gap = -1

        baseReward = 0.05
        if gap >= 0 and gap <= 0.1:
            reward = baseReward * 10
        elif gap > 0.1:
            reward = baseReward / gap
        else:
            reward += 0.01 # found enemy.

        if bPrint == True:
            self.logger.info ("****curr x_normal: %.3f, action: %.2f, reward: %.2f" %(x_normal, action, reward))

        #if expectedAction == turn_action:
        #    reward += 0.01
        #    if bPrint == True:
        #        self.logger.info ("x_normal: %.2f, reward: %.2f" %(x_normal, reward))
        #else:
        #    reward -= 0.01
        #    if bPrint == True:
        #        self.logger.info('x_normal: %.2f, reward: %.2f' %(x_normal, reward))

        return reward, True

    # If it is stuck in some position, we will get penalty.
    def _CalculateStuckReward(self, actions, smallMapInfo, lastSmallMapInfo, bPrint = False):
        reward = 0.0

        # First position
        if smallMapInfo != None and self.__stuckOrigPosition == None:
            self.__stuckOrigPosition = smallMapInfo

        # Update orig position using saved valid position.
        if self.__stuckDecay <= 0:
            self.__stuckDecay = self.__stuckDecayInit # reset decay parameter.
            self.__stuckOrigPosition = self.__tmpStuckPosition

        # We process frame about 10FPS, therefore we will wait about 50 frames to
        # check if there is stuck.
        if smallMapInfo != None and\
           self.__stuckOrigPosition != None and\
           (self.__stuckDecay <= (self.__stuckDecayInit / 2)):
            myPos     = smallMapInfo[2]
            origMyPos = self.__stuckOrigPosition[2]

            if myPos != None and origMyPos != None:
                x_diff          = math.fabs(myPos['x'] - origMyPos['x']) * self.__mapScreenWidth
                y_diff          = math.fabs(myPos['y'] - origMyPos['y']) * self.__mapScreenHeight
                shiftPos        = math.sqrt(x_diff * x_diff + y_diff * y_diff)

                # x and y is in pixel. If shift distance is less than
                if shiftPos < 9:
                    reward = -0.01

                    if bPrint == True:
                        self.logger.info ("stuck: shiftPos: %f, reward: %f, (%.2f, %.2f) -> (%.2f, %.2f), decay: %d" %
                                          (shiftPos, reward, origMyPos['x'], origMyPos['y'], myPos['x'], myPos['y'], self.__stuckDecay))


        # We don't always get our position information from small map, so we should
        # save the valid small map information.
        if smallMapInfo != None:
            self.__tmpStuckPosition = smallMapInfo

        self.__stuckDecay -= 1
        return reward

    # If viewport is toward wall, it will get penalty.
    def _CalculateViewReward(self, smallMapInfo, bPrint = False):
        reward = 0.0

        if smallMapInfo == None:
            return reward

        myViewPos = smallMapInfo[1]
        myPos     = smallMapInfo[2]

        if myViewPos == None or myPos == None:
            return reward

        diffX = (myPos['x'] - myViewPos['x']) / 2
        diffY = (myPos['y'] - myViewPos['y']) / 2

        newPosX = diffX + myViewPos['x']
        newPosY = diffY + myViewPos['y']

        newPosX = int(newPosX * self.__mapScreenWidth)
        newPosY = int(newPosY * self.__mapScreenHeight)

        if newPosX >= self.__mapScreenWidth or newPosY >= self.__mapScreenHeight:
            return reward

        pixel = self.__mapObstacle[newPosY][newPosX]

        myPosX     = int(myPos['x'] * self.__mapScreenWidth)
        myPosY     = int(myPos['y'] * self.__mapScreenHeight)
        myViewPosX = int(myViewPos['x'] * self.__mapScreenWidth)
        myViewPosY = int(myViewPos['y'] * self.__mapScreenHeight)

        if pixel[0] < 10 and pixel[1] < 10 and pixel[2] < 10:
            reward = -0.1

            if bPrint == True:
                self.logger.info("Viewport: (%d, %d)->(%d, %d) (%d, %d) reward: -0.1" % 
                                 (myPosX, myPosY, myViewPosX, myViewPosY, newPosX, newPosY))

        diffX = abs(myPosX - myViewPosX)
        diffY = abs(myPosY - myViewPosY)
        if diffX < 4 and diffY < 4:
            reward = -0.1

            if bPrint == True:
                self.logger.info("Viewport bottom-top: (%d, %d)->(%d, %d) (%d, %d) reward: -0.1" % 
                                 (myPosX, myPosY, myViewPosX, myViewPosY, newPosX, newPosY))

        return reward

    def _CalculateMoveReward3(self, actions, smallMapInfo, lastSmallMapInfo, bPrint = False):
        reward = 0.0

        if smallMapInfo != None and lastSmallMapInfo != None:
            myPos     = smallMapInfo[2]
            lastMyPos = lastSmallMapInfo[2]

            if myPos != None and lastMyPos != None:
                myPosX     = int(myPos['x'] * self.__mapScreenWidth)
                myPosY     = int(myPos['y'] * self.__mapScreenHeight)
                myLastPosX = int(lastMyPos['x'] * self.__mapScreenWidth)
                myLastPosY = int(lastMyPos['y'] * self.__mapScreenHeight)

                diffX = abs(myPosX - myLastPosX)
                diffY = abs(myPosY - myLastPosY)
                dist  = math.sqrt(diffX * diffX + diffY * diffY)

                baseReward = 0.001
                limit = 10
                if dist > limit:
                    reward = baseReward
                elif dist > 1 and dist <= limit:
                    reward = (baseReward/limit) * dist

                if reward > 0 and bPrint == True:
                    self.logger.info ("move : (%d, %d) -> (%d, %d), reward: %.5f" %
                                      (myLastPosX, myLastPosY, myPosX, myPosY, reward))

        return reward

    def _CalculateBuddyReward(self, actions, smallMapInfo, bPrint = False):
        reward = 0.0

        if smallMapInfo is None:
            return reward

        myPos = smallMapInfo[2]
        if myPos is None:
            return reward

        mapFriends = smallMapInfo[3]
        if mapFriends is None:
            return reward

        myPosX = int(myPos['x'] * self.__mapScreenWidth)
        myPosY = int(myPos['y'] * self.__mapScreenHeight)

        guysNum = min(len(mapFriends['x']), 4)
        for i in range(guysNum):
            buddyPosX = int(mapFriends['x'][i])
            buddyPosY = int(mapFriends['y'][i])

            diffX = abs(myPosX - buddyPosX)
            diffY = abs(myPosY - buddyPosY)
            dist  = math.sqrt(diffX * diffX + diffY * diffY)

            if diffX < 4 and diffY < 4:
                reward += 0.002

                if bPrint is True:
                    self.logger.info ("buddy : (%d, %d) -> (%d, %d), reward: %.5f" %
                                      (myPosX, myPosY, buddyPosX, buddyPosY, reward))

        return reward

    def _CalculateFireReward(self, fireAction, lastAmmo, ammo, bPrint = False):
        reward = 0.0

        ammoBaseReward = 1
        if lastAmmo == 0 or lastAmmo < ammo:
            ammoReward = 0.0
        elif lastAmmo > 10 and ammo == 0:
            ammoReward = 0.0
        else:
            ammoGap = lastAmmo - ammo
            if ammoGap < 15 and ammoGap > 0:
                ammoReward = ammoGap * ammoBaseReward
            else:
                ammoReward = 0

        if ammoReward > 0 and bPrint == True:
            self.logger.info ("ammoGap: %d, reward: %.2f" %(ammoGap, ammoReward))

        reward += ammoReward

        ## there is no fire action
        #if fireAction > 0.5 and ammoReward > 0:
        #    reward += ammoReward
        #    if ammoReward > 0 and bPrint == True:
        #        self.logger.info ("fire: %.2f, reward: %.2f" % (fireAction, ammoReward))

        ## If fire action is true but there is no fire.
        #if fireAction > 0.5 and ammoReward == 0:
        #    reward -= 0.01
        #    if bPrint == True:
        #        self.logger.info ("fire true: %.2f, reward: -0.01" % (fireAction))

        ## If fire action is false but there is fire.
        #if fireAction <= 0.5 and ammoReward > 0:
        #    reward -= 0.01
        #    if bPrint == True:
        #        self.logger.info ("fire false: %.2f, reward: -0.01" % (fireAction))

        return reward

    def _CalculateReward(self,
                         actions,
                         humanInfo,
                         myGroupId,
                         blood,
                         kills,
                         ammo,
                         lastBlood,
                         lastKills,
                         lastAmmo,
                         attackDir,
                         smallMapInfo = None,
                         lastSmallMapInfo = None):
        # Rotate(-1~1)/Move Direction(0~1)/Move Distance(0~1)/Fire(0~1, >0.5 means fire)

        bPrint = False
        reward = 0.0

        if bPrint == True:
            self.logger.info ("************begin reward***************")

        # -0.02 for one blood.
        if lastBlood <= blood:
            reward += 0.0
        else:
            bloodGap = lastBlood - blood
            reward -= bloodGap * 0.02

            if bPrint == True:
                self.logger.info ("bloodGap: %d, reward: %f" %(bloodGap, -bloodGap * 0.01))

        # 10 for one kill
        if lastKills < kills:
            killsGap = kills - lastKills
            reward += killsGap * 10

            if bPrint == True:
                self.logger.info ("killsGap: %d, reward: %f" %(killsGap, killsGap * 0.5))

        if actions is None:
            return reward

        # -0.01 for one ammo of manual firing
        # +0.01 for one ammo of auto firing
        reward += self._CalculateFireReward(actions[3], lastAmmo, ammo, True)

        # If do nothing, we will give penalty to agent.
        #if reward == 0.0 and action == 6:
        #    reward -= 0.01
        #    if bPrint == True:
        #        self.logger.info ("no action penalty: -0.01")

        # Get my position
        # reward += self._CalculateMoveReward1(smallMapInfo, lastSmallMapInfo)
        # reward += self._CalculateMoveReward2(actions, smallMapInfo, lastSmallMapInfo, bPrint)

        # If stuck in some position.
        reward += self._CalculateStuckReward(actions, smallMapInfo, lastSmallMapInfo, True)

        # Viewport
        reward += self._CalculateViewReward(smallMapInfo, True)

        # Move reward
        reward += self._CalculateMoveReward3(actions, smallMapInfo, lastSmallMapInfo, True)

        # Get reward based on position of enemies.
        enemyReward, bEnemy = self._CalculateEnemyReward(actions[0], humanInfo, myGroupId, False, True)

        if bEnemy is True:
            self.__enemyDecay = self.__enemyDecayInit
            self.__enemyLastState = humanInfo

        reward += enemyReward

        if bEnemy == False:
            # There is no enemy detected.
            # Get reward from attacked direction
            attackReward, bAttack = self._CalculateAttackedDirReward(actions[0], attackDir, False, True)
            reward += attackReward

            # If there is no enemy and no attck tip, we use the previous enemy state.
            # If there is no previous enemy state, we use the previous attack tip.
            if bAttack == False:
                if self.__enemyDecay > 0:
                    enemyReward, bEnemy = self._CalculateEnemyReward(actions[0], self.__enemyLastState, myGroupId, True, True)
                    reward += enemyReward

                if self.__attackDecay > 0:
                    attackReward, bAttack = self._CalculateAttackedDirReward(actions[0], self.__attackLastState, True, True)
                    reward += attackReward

        # Decay for enemy detection and attack tip.
        if self.__enemyDecay > 0:
            self.__enemyDecay -= 1

        if self.__attackDecay > 0:
            self.__attackDecay -= 1

        # Get reward if there are my buddies nearby.
        reward += self._CalculateBuddyReward(actions, smallMapInfo, True)

        # If being attacked by enemy, we should give minus reward.
        #if attackDir != 'None':
        #    reward -= 0.05

        #if action == self.__lastActions:
        #    if action == 4 or action == 5:
        #        reward += 0.005
        #        # self.logger.info ("action: %d, reward: 0.005" %(action))

        if reward != 0.0:
            self.logger.info ("reward: %f" %(reward))

        if bPrint == True:
            self.logger.info ("************end reward***************\n")

        return reward

    def InitTCP(self):
        res = self.__tcp.InitTCPClient()
        #if res == True:
        #    self.__tcp.start()

    def IsEpsiodeOver(self):
        return self.terminal

    def IsEpsiodeStart(self):
        state, reward, terminal, _ = self.GetStateForImitationLearning()
        self.__lastActions = None
        if terminal != True:

            # Test for actions.
            #actions = [-1.0, 0.4, 0.7, 0.6]
            #for i in range(20):
            #    actions[0] += 0.1
            #    self.DoActionForImitationLearning(actions, True)

            #actions = [-0.5, 0.0, 0.7, 0.6]
            #for i in range(10):
            #    actions[1] += 0.1
            #    self.DoActionForImitationLearning(actions, True)

            #actions = [-0.5, 0.2, 0.0, 0.6]
            #for i in range(10):
            #    actions[2] += 0.1
            #    self.DoActionForImitationLearning(actions, True)

            return True
        else:
            self.__origPosition      = None
            self.__stuckOrigPosition = None
            return False

    def SendSampleToTrainingAI(self, sample):
        self.agentAPI.SendSampleToTrainingAI(sample)

def Run(gameEnv, aiModel, isTestMode):
    while True:
        if gameEnv.IsEpsiodeStart() is not True:
            time.sleep(0.001)
            continue

        LOG.error('Epsiode start')
        aiModel.OnEpsiodeStart()

        RunEpsiode(gameEnv, aiModel, isTestMode)

        LOG.error('Epsiode over')
        aiModel.OnEpsiodeOver()

        if aiModel.IsUsingTCP() == 1:
            gameEnv.InitTCP()

def RunEpsiode(gameEnv, aiModel, isTestMode):
    while True:
        if isTestMode is True:
            aiModel.TestOneStep()
        else:
            aiModel.TrainOneStep()
        if gameEnv.IsEpsiodeOver() is True:
            break
    return

if __name__ == '__main__':
    # action[0] is Rotate(-1~1)
    # action[1] Move Direction(0~1)
    # action[2] Move Distance(0~1)
    # action[3] Fire(0~1, >0.5 means fire)

    gameEnv = CFMGameMapSearchEnv()

    for i in range(20):
        actions = [-1.0, 0.4, 0.7, 0.6]
        actions[0] += 0.1
        gameEnv.DoActionForImitationLearning(actions, True)

    for i in range(10):
        actions = [-0.5, 0.0, 0.7, 0.6]
        actions[1] += 0.1
        gameEnv.DoActionForImitationLearning(actions, True)
