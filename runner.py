from agent import AgentDDQN
from collections import namedtuple
from environment import Environment
from itertools import count
from time import time

import matplotlib.pyplot as plt

import torch

EpisodeLog = namedtuple('EpisodeLog', ('runDuration', 'finalScore', 'cumulativeReward'))

class Runner(object):

    def __init__(self, agentParams, envParams, logFile, checkpointName='policyCheckpoint', checkpointFreq=10):
        self.agent = AgentDDQN(**agentParams)
        self.environment = Environment(**envParams)
        self.checkpointName = checkpointName
        self.checkpointFreq = checkpointFreq
        self.logFile = logFile
        self.log = []

    def run(self, num_episodes, verbose=True):
        for eps_idx in range(num_episodes):
            print(f'\nEpisode {eps_idx:>5,} Start! ', end='')
            self.environment.startQWOP()
            last_screen = self.environment.getState()
            current_screen = self.environment.getState()
            state = current_screen - last_screen
            sTime = time()
            for t in count():
                action = self.agent.selectAction(state)
                reward = self.environment.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float, device=self.agent.device)

                last_screen = current_screen
                current_screen = self.environment.getState()
                gameOver = self.environment.isDone()
                if not gameOver:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                self.agent.memory.push(state, action, next_state, reward)
                self.agent.optimizeModel()
                state = next_state
                if gameOver:
                    break

            runDuration = time()-sTime
            finalScore = self.environment.getFinalScore()
            cumulativeReward = self.environment.cumulativeReward
            self.agent.updateTarget(eps_idx)

            self.log.append(EpisodeLog(runDuration, finalScore, cumulativeReward))
            if eps_idx % self.checkpointFreq == 0:
                torch.save(self.agent.policy_net.state_dict(), self.checkpointName)
                self.writeLog()
            if verbose:
                print(f'Episode End. Duration: {runDuration:>6.1f}; Final Score: {finalScore:>6.1f}; Cumulative Reward: {cumulativeReward:.1f}', end='')
        torch.save(self.agent.policy_net.state_dict(), self.checkpointName)
        self.writeLog()
        self.environment.webdriver.quit()
        print("\nRun complete! 'Ctrl+C' to safely end server & script.")

    def loadParameters(self, checkPointFilePath):
        self.agent.loadParameters(checkPointFilePath)

    def writeLog(self):
        with open(self.logFile, 'w+') as w:
            for i in self.log:
                w.write(f'{i}\n')
        self.log = []







# if __name__ == '__main__':
#     agentParams = {
#         'stateSize': 40,
#         'batch_size': 128,
#         'gamma': 0.99,
#         'eps_start': 0.9,
#         'eps_end': 0.05,
#         'eps_decay': 300,
#         'target_update': 10,
#         'device': 'cpu'
#     }
#
#     environmentParams = {
#         'stateSize': 40,
#         'rewardTimePenalty': 0.0005,
#         'rewardScore': 0.5,
#         'rewardHiScore': 2,
#         'shortDuration': 0.08,
#         'longDuration': 0.14,
#         'port': 8080
#     }
#
#
#     agent = Agent(**agentParams)
#     env = Environment(**environmentParams)
#     runner = Runner(agent, env)
#     runner.run(1000)