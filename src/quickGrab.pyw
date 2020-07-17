import time
import os
import numpy as np
from PIL import ImageGrab
from cv2 import cvtColor, COLOR_RGB2BGR, imwrite, COLOR_BGR2GRAY, calcHist
from win32 import win32api
from src import settings
import d3dshot

screenshot_folder = settings.SCREEN_GRABS_PATH

def empty_folder():
    for file in os.listdir(screenshot_folder):
        file_path = os.path.join(screenshot_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def save(img):
    img_name = screenshot_folder + '\\zfull_snap__' + str(int(time.time())) + '.png'
    img.save(img_name, 'PNG')

def save_cv(img):
    img_name = screenshot_folder + '\\zfull_snap__' + str(int(time.time())) + '.png'
    imwrite(img_name, img)

def screenGrab(save_images = False):
    box = (*settings.SCREEN_START, *settings.SCREEN_END)
    img = ImageGrab.grab(box)

    if save_images:
        save(img)  # Slows performance

    img = cvtColor(np.array(img), COLOR_RGB2BGR)
    # img_grey = cvtColor(img, COLOR_BGR2GRAY)
    # print("Color:", img_grey[p[1]][p[0]])

    return img


def screen_grab(save = False):
    t = time.time()

    """
    Your primary display is selected by default but if you have a multi-monitor setup, 
    you can select another entry in d.displays:  d.display = d.displays[1]
    """

    d = d3dshot.create(capture_output="numpy")
    d.screenshot(region=(*settings.SCREEN_START, *settings.SCREEN_END))

    print(time.time()-t)

def pixel_color(p):
    img = screenGrab()
    img_grey = cvtColor(img, COLOR_BGR2GRAY)
    print("Color:", img_grey[p[1]-settings.SCREEN_START[1]][p[0]-settings.SCREEN_START[0]])

def main():

    p = win32api.GetCursorPos()
    screenGrab(True)

    while True:
        time.sleep(0.1)
        p = win32api.GetCursorPos()
        t = time.time()
        screen_grab(True)
        print(time.time()-t)
        print(p)
        pixel_color(p)

if __name__ == '__main__':
    main()
