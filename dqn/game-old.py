import numpy as np
import sys
import cv2
import time as Time
import matplotlib.pyplot as plt
from PIL import Image
from dqn.utils import misc
from dqn.utils.bean import State, Sample


class Game:
    def __init__(self):
        self.state = None
        self.n_action_taken = 0  # the number of actions (waits) taken since last jmp
        self.step_weight = [1, 4, 16, 64, 256]  # see below
        self.time_accu = 0
        self.time_tag = 0
        self.score = 0
        self.prev_score = 0
        self.terminal = False
        self.test = False
        self.update_env()

    def get_state(self):
        return State(img=misc.preprocess(self.state),
                     time_tag=self.time_tag,
                     time=self.time_accu,
                     step=self.n_action_taken
                     )

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
        # wait
        self.time_accu += action * self.step_weight[self.n_action_taken]
        self.n_action_taken += 1
        if self.n_action_taken >= 5:
            # jmp
            misc.press_hold(self.time_accu)
            Time.sleep(2.5 + self.time_accu / 1000.0)
            self.update_env()
            self.prev_score = prev_score
            prev_score = prev_score if not self.terminal else 0
            self.n_action_taken = 0
            self.time_accu = 0
        #if action == 0:
        #    # jmp immediately
        #    utils.press_hold(self.time)
        #    Time.sleep(2.5 + self.time / 1000.0)
        #    self.update_env()
        #    # reward only depends on how many waits have been taken
        #    # but not how high the prev score before failure is
        #    prev_score = prev_score if not self.terminal else 0
        #    penalty = 0.05 * self.n_action_taken  # for not acting quickly
        #    self.n_action_taken = 0
        #else:
        #    # or wait for these ms
        #    self.time += action
        #    self.n_action_taken += 1
        #    penalty = 0.05 * self.n_action_taken  # for not acting quickly
        return Sample(p_state=None,
                      state=State(img=self.state,
                                  time=self.time_accu,
                                  time_tag=self.time_tag,
                                  step=self.n_action_taken
                                  ),
                      action=None,
                      reward=min(self.score - prev_score, 2),
                      terminal=self.terminal
                      )
        # return {'img': self.state,
        #         'time_tag': self.time_tag,
        #         'score': self.score,
        #         'reward': min(self.score - prev_score, 2),
        #         'time': self.time_accu,
        #         'step': self.n_action_taken,
        #         'terminal': self.terminal}

    def reset(self):
        misc.press_hold(10)  # replay button
        Time.sleep(0.4)
        self.state = None
        self.time_accu = 0
        self.score = 0
        self.n_action_taken = 0
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
        if False:
            samples = np.empty((0, 100))
            responses = []
            keys = [i for i in range(48, 58)]

            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    [x, y, w, h] = cv2.boundingRect(cnt)

                    if h > 28:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        roi = thresh[y:y + h, x:x + w]
                        roismall = cv2.resize(roi, (10, 10))
                        cv2.imshow('norm', im)
                        key = cv2.waitKey(0)

                        if key == 27:  # (escape to quit)
                            sys.exit()
                        elif key in keys:
                            responses.append(int(chr(key)))
                            sample = roismall.reshape((1, 100))
                            samples = np.append(samples, sample, 0)

            responses = np.array(responses, np.float32)
            responses = responses.reshape((responses.size, 1))
            print("training complete")

            np.savetxt('generalsamples1.data', samples)
            np.savetxt('generalresponses1.data', responses)

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
