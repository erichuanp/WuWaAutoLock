import time
import cv2
import numpy as np
from PIL import ImageGrab
from skimage.metrics import structural_similarity as ssim
import pyautogui

itemPos = (172, 137, 561, 1004)
lockPos = (1804, 235, 1819, 253)
needPos = (1400, 250, 1575, 350)
scrollAmount = -3000  # -3000 to -5000
LabelPos = [None, (324, 90), None, (423, 90), (521, 90)]
C = [None, 'c1attack.png', None, 'annihilation.png', 'c4crit.png']
gap = 0.2
numUseful = 0
queue = [1, 4]  # 用户添加

def lockScreen(region):
    s1 = ImageGrab.grab(bbox=region)
    gray = cv2.cvtColor(np.array(s1), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('lock.png', gray)
def UnlockScreen(region):
    s1 = ImageGrab.grab(bbox=region)
    gray = cv2.cvtColor(np.array(s1), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('unlock.png', gray)
def isLocked():
    lock = ImageGrab.grab(bbox=lockPos)
    lock = cv2.cvtColor(np.array(lock), cv2.COLOR_BGR2GRAY)
    unlock_image = cv2.imread('unlock.png', cv2.IMREAD_GRAYSCALE)
    lock_image = cv2.imread('lock.png', cv2.IMREAD_GRAYSCALE)
    score_unlock, _ = ssim(lock, unlock_image, full=True)
    score_lock, _ = ssim(lock, lock_image, full=True)
    return score_lock > score_unlock
def checkIfUseful(template_file):
    template = cv2.imread(template_file, 0)

    screenshot = ImageGrab.grab(bbox=needPos)
    screenshot_np = np.array(screenshot)
    gray_screenshot = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    res = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    # 返回是否找到了模板
    return len(loc[0]) > 0
def capture_screenshot(region):
    screenshot = ImageGrab.grab(bbox=region)
    return np.array(screenshot)
def images_are_equal(image1, image2, threshold=0.99):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_image1, gray_image2, full=True)
    return score >= threshold
template = cv2.imread('plus_sign.png', 0)
w, h = template.shape[::-1]
previous_screenshot = np.zeros_like(capture_screenshot(itemPos))


for label in queue:
    template_file = C[label]
    pyautogui.click(LabelPos[label][0], LabelPos[label][1])

    while True:
        grid_coordinates = []
        time.sleep(3)
        current_screenshot = capture_screenshot(itemPos)
        gray_screenshot = cv2.cvtColor(current_screenshot, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(current_screenshot, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            grid_coordinates.append(pt)

        unique_coordinates = []
        for coord in grid_coordinates:
            if not any(np.linalg.norm(np.array(coord) - np.array(existing_coord)) < 10 for existing_coord in
                       unique_coordinates):
                unique_coordinates.append(coord)

        for coord in unique_coordinates:
            time.sleep(gap)
            pyautogui.click(x=coord[0] + itemPos[0], y=coord[1] + itemPos[1])
            time.sleep(gap)
            if checkIfUseful(template_file):
                numUseful += 1
                if not isLocked():
                    pyautogui.click(lockPos[0], lockPos[1])
                    print('Locked')
            else:
                if isLocked():
                    pyautogui.click(lockPos[0], lockPos[1])
                    print('It is useless. Unlocked')

        if images_are_equal(previous_screenshot, current_screenshot):
            print('Mission Complete')
            break

        previous_screenshot = current_screenshot.copy()
        pyautogui.moveTo(367, 566)
        pyautogui.scroll(scrollAmount)


print(f'Total: {numUseful}')
