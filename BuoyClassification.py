import cv2
import numpy as np
from collections import namedtuple


colorBounds = namedtuple('cbound', 'color name ub lb')
yellowBound = colorBounds(color=0, name='yellow', ub=(40, 255, 255), lb=(20, 30, 20))
orangeBound = colorBounds(color=1, name='orange', ub=(35, 255, 255), lb=(10, 30, 20))
greenBound  = colorBounds(color=2, name='green',  ub=(70, 255, 255), lb=(60, 30, 20))


def ComputeMaskArea(patch, bound):
    '''
    generate confidence score of buoy class
    :param patch: rgb image patch
    :param bound: color threshold bounds
    :return: confidence score
    '''
    hsv_img = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask_img = cv2.inRange(hsv_img, bound.lb, bound.ub)
    size = mask_img.shape

    area = size[0] * size[1]
    whitePixels = np.where(mask_img == 255)[0]
    score = whitePixels.shape[0] / area
    return score

def BuoyClass(patch):
    '''
    generate class for input image patch
    The key idea is to enumerate over all three colors and return the one which has maximum score
    :param patch: RGB image patch
    :return: patch class with confidence score
    '''
    bounds = [yellowBound, orangeBound, greenBound]
    allColors = {bound.color : ComputeMaskArea(patch.copy(), bound) for bound in bounds}

    maxScore = max(allColors.values())

    totalScore = sum(allColors.values())

    for key, value in allColors.items():
        if value == maxScore:
            return {key : value / totalScore}




if __name__ == '__main__':
    patchFile = 'templates/Template2.jpg'
    patch = cv2.imread(patchFile)
    # patch = BuoyClass(patch)
    resClass = BuoyClass(patch)
    print(resClass)
    # cv2.imshow('frame', patch)
    # cv2.waitKey(0)

