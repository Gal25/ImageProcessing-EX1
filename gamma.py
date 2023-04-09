"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np
import cv2
from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import imReadAndConvert, imDisplay




def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    # callback function for createTrackbar function
    def gamma(g):
        # integer value to float
        g = g / 100.0

        # gamma to the image
        gImage = np.power(img / 255.0, g)
        gImage = np.uint8(gImage * 255)

        cv2.imshow('GAMMA', gImage)

    img = cv2.imread(img_path)
    if rep == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # create a window
    cv2.imshow('GAMMA', img)


    # function that create a trackbar and allows us to adjust a gamma value.
    cv2.createTrackbar('Gamma', 'GAMMA', 100, 200, gamma)

    # user input
    cv2.waitKey(0)
    #close the window when done
    cv2.destroyAllWindows()

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
