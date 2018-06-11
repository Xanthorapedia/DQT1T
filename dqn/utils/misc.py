import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMG_SIZE = {'h': 1920, 'w':1080}
output_size = 224


def pull_screenshot():
    cmd_scr = 'adb shell screencap -p /sdcard/t1t.png'
    cmd_pull = 'adb pull /sdcard/t1t.png ./state.png'
    os.popen(cmd_scr).close()
    os.popen(cmd_pull).close()
    state = np.array(Image.open('./state.png'))
    return state


def preprocess(state):
    state = cv2.resize(state, (output_size, int(state.shape[0] * (output_size / state.shape[1]))))
    return state[int((state.shape[0] - output_size) * 0.5):int((state.shape[0] + output_size) * 0.5), :, 0:3]  # get rid of alpha chanel


def press_hold(time, state=None):
    rand = np.random.random_integers(0, 10, [2])
    y, x = int(IMG_SIZE['h'] * 0.82 + rand[0]), int(IMG_SIZE['w'] * 0.5 + rand[1])
    cmd = 'adb shell input swipe {} {} {} {} {}'.format(x, y, x, y, time)
    if state is not None:
        return ImageDraw.Draw(Image.fromarray(state)).ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0))
    # print('hold for', time)
    os.system(cmd)


if __name__ == '__main__':
    img = preprocess(pull_screenshot())
    img = Image.fromarray(img)
    img.save('./test1.png')
    press_hold(20)
    imgplot = plt.imshow(img)
    plt.show()
