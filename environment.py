import http.server
import socketserver
import numpy as np
import pytesseract
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from pynput.keyboard import Key, Controller
from threading import Thread
from PIL import Image
from time import sleep, time
from mss import mss

import torch
import torchvision.transforms as T



class Environment:

    def __init__(self, stateSize, comboMax, comboGain, comboDecay, noopPenalty,
                 shortDuration, longDuration, port=8080):
        self.stateSize = stateSize
        self.comboMax = comboMax
        self.comboGain = comboGain
        self.comboDecay = comboDecay
        self.currCombo = 1
        self.noopPenalty = noopPenalty

        self.actionLookup = {
            0: (None, None), # no-operation,
            1: ('q', shortDuration),
            2: ('q', longDuration),
            3: ('w', shortDuration),
            4: ('w', longDuration),
            5: ('o', shortDuration),
            6: ('o', longDuration),
            7: ('p', shortDuration),
            8: ('p', longDuration),
        }

        Thread(target=self._startServer, args=(port,)).start()
        sleep(0.008)
        self.webdriver = webdriver.Chrome()
        self.webdriver.get(f'http://localhost:{port}/Athletics.html')
        self.webdriver.set_window_position(x=10, y=10)
        self.webdriver.set_window_size(width=660, height=560)
        self.gameCanvas = self.webdriver.find_element(By.TAG_NAME, 'canvas')
        self.gameCanvasLoc = self.gameCanvas.location
        self.gameCanvasSize = self.gameCanvas.size
        self.gameCanvasParams = {
            'top': 150, # only need to adjust top
            'left': 15, # only need to adjust left
            'width': 645,
            'height': 410
        }
        self.screenShotParams = { # square screen shot
            'top': 150,
            'left': self.gameCanvasParams['left'] + 110,
            'width': self.gameCanvasParams['height'],
            'height': self.gameCanvasParams['height']
        }
        self.keyboard = Controller()
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(stateSize, interpolation=Image.CUBIC),
                                 T.ToTensor()])
        # self.sTime = None
        self.prevScore = 0
        self.currScore = 0
        self.currHiScore = 0
        self.cumulativeReward = 0

    def _startServer(self, port):
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(('', port), Handler) as httpd:
            print('Serving at port', port)
            httpd.serve_forever()

    def startQWOP(self) -> None:
        self.gameCanvas.click()
        self.keyboard.tap(Key.space)
        self.captureScreen()
        self.prevScore = 0
        self.currScore = 0
        self.currHiScore = 0
        self.cumulativeReward = 0

    def captureScreen(self) -> None:
        with mss() as sct:
            self.screenCap = np.array(sct.grab(self.screenShotParams))[..., :3]

    def captureScore(self, errorThreshold=3) -> None:
        imgScore = self.screenCap[20:60, 80:-100, 0] < 125
        score = pytesseract.image_to_string(image=imgScore)
        result = re.compile("-?[0-9]+.?[0-9]").findall(score)
        if result:
            try:
                newScore = float(result[0])
                if abs(self.currScore - newScore) < errorThreshold:
                    self.prevScore = self.currScore
                    self.currScore = newScore
                    if newScore > self.currHiScore:
                        # self.newHiScore = True
                        self.currHiScore = newScore
                        self.currCombo = min(self.currCombo * self.comboGain, self.comboMax)
                    elif self.currCombo > 1:
                        self.currCombo *= self.comboDecay
                    else:
                        self.currCombo = 1
            except ValueError:
                pass


    def getScore(self) -> float:
        self.captureScore()
        return self.currScore

    def getScreen(self) -> np.ndarray:
        self.captureScreen()
        return self.screenCap.copy()

    def getFinalScore(self) -> float:
        imgScore = self.screenCap[190:220, 100:-100, 0] < 125
        score = pytesseract.image_to_string(image=imgScore)
        result = re.compile("-?[0-9]+.?[0-9]").findall(score)
        if result:
            try:
                return float(result[0])
            except ValueError:
                print('failed to get final score!')
                pass
        return self.currScore

    def isDone(self) -> bool:
        return (self.screenCap[100:-100, :, 0] > 200).mean() > 0.55

    def step(self, keyCode) -> float:
        def asyncKeyPress(key, duration):
            if key is not None:
                self.keyboard.press(key)
                sleep(duration)
                self.keyboard.release(key)
        Thread(target=asyncKeyPress, args=self.actionLookup[keyCode]).start()
        self.noopKey = (keyCode == 0)
        return self.getReward()

    def getState(self):
        screen = self.getScreen().transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0)

    def getReward(self):
        self.captureScore()
        scoreGain = self.currScore - self.prevScore
        if self.noopKey and (scoreGain == 0):
             return self.noopPenalty / self.currCombo
        currReward = scoreGain * self.currCombo
        self.cumulativeReward += currReward
        self.noopKey = False
        return currReward

