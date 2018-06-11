import numpy as np
import cv2
import time as Time
import matplotlib.pyplot as plt
from PIL import Image
from dqn.utils import misc
from dqn.utils.bean import State, Sample


class Game:
    def __init__(self):
        self.state = None
        self.time_tag = 0
        self.score = 0
        self.prev_score = 0
        self.terminal = False
        self.test = False
        self.update_env()

    def get_state(self):
        return State(img=misc.preprocess(self.state), time_tag=self.time_tag)

    def update_env(self):
        self.time_tag = round(Time.time())
        if self.test:
            self.state = np.array(Image.open('state.png'))
        else:
            self.state = misc.pull_screenshot()
        self.score, self.terminal = self.get_score()

    # takes $action$ after $immediate$ ms, returns the resulting state
    def act(self, action):
        prev_score = self.score
        misc.press_hold(action)
        Time.sleep(2.5 + action / 1000.0)
        self.update_env()
        self.prev_score = prev_score
        prev_score = prev_score if not self.terminal else 0
        return Sample(p_state=None,
                      state=State(img=self.state, time_tag=self.time_tag),
                      action=None,
                      reward=min(self.score - prev_score, 2),
                      terminal=self.terminal
                      )

    def reset(self):
        misc.press_hold(10)  # replay button
        Time.sleep(0.4)
        self.state = None
        self.score = 0
        self.update_env()
        Time.sleep(0.1)

    # Adapted from and inspired by Abid
    # https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
    def get_score(self):
        im = self.state[100:400, 0:600, :]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ave_pixel = 0.
        for _ in range(100):
            ave_pixel += im[np.random.randint(0, 300)][np.random.randint(0, 600)]
        ave_pixel /= 100.
        if ave_pixel < 130:
            return -1, True
        if self.test:
            print(ave_pixel)
            plt.imshow(im)
            plt.show()
        im = cv2.GaussianBlur(im, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(im, 255, 1, 1, 29, 2)
        if self.test:
            plt.imshow(thresh)
            plt.show()
        # finding Contours
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        model = cv2.ml.KNearest_create()
        samples = np.loadtxt('samples.data', np.float32)
        responses = np.loadtxt('responses.data', np.float32)

        responses = responses.reshape((responses.size, 1))
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)

        digits = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 70.0 * self.state.shape[0] / 1920:
                    if self.test:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        roi = thresh[y:y + h, x:x + w]
                        plt.imshow(roi)
                        plt.show()

                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    _, results, _, _ = model.findNearest(roismall, k=1)
                    digit = str(int((results[0][0])))
                    if digit is not None:
                        digits.insert(0, digit)
        # no valid digits
        if not digits:
            return -1
        score = 0
        for i in digits:
            score = score * 10 + int(i)
        return score, False


def rand_state():
    return {
         'img': np.random.rand(240, 240, 3),
         'score': np.random.randint(0, 100),
         'reward': np.random.randint(0, 3),
         'time': np.random.randint(200, 1000),
         'terminal': bool(np.random.rand() > 0.5)
     }


if __name__ == '__main__':
    g = Game()
    g.test = True
    g.update_env()
    print('score = ' + str(g.get_score()))
