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
from typing import List
import numpy as np
import cv2
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316138411


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    img = cv2.imread(filename)
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img.astype(np.float64) / 255.0  # Normalize intensity values to [0, 1]
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.astype(np.float64) / 255.0  # Normalize intensity values to [0, 1]
    else:
        raise ValueError("Invalid representation value, should be 1 or 2")
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(img, cmap='gray')  # Display grayscale image
    elif representation == 2:
        plt.imshow(img)  # Display RGB image
    else:
        raise ValueError("Invalid representation value, should be 1 or 2")
    plt.show()  # Show the image in a new figure window


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    T = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])
    # Reshape the input image to a 2D array of pixels (height*width)x3
    imRGB_reshaped = imgRGB.reshape(-1, 3)
    # Apply the transformation matrix T to convert RGB to YIQ
    imYIQ_reshaped = np.dot(imRGB_reshaped, T.T)
    imYIQ = imYIQ_reshaped.reshape(imgRGB.shape)
    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    T_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.106, 1.703]])
    # Reshape the input image to a 2D array of pixels (height*width)x3
    imYIQ_reshaped = imgYIQ.reshape(-1, 3)
    # Apply the inverse transformation matrix T_inv to convert YIQ to RGB
    imRGB_reshaped = np.dot(imYIQ_reshaped, T_inv.T)
    imRGB = imRGB_reshaped.reshape(imgYIQ.shape)
    return imRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # Check if the input image is grayscale or RGB
    global histOrg, histEq
    is_grayscale = (imgOrig.ndim == 2) or (imgOrig.ndim == 3 and imgOrig.shape[2] == 1)

    if not is_grayscale:
        # Convert RGB to YIQ and equalize the Y channel
        imYIQ = transformRGB2YIQ(imgOrig)
        imYIQ[:, :, 0] = hsitogramEqualize(imYIQ[:, :, 0])[0]
        imEq = transformYIQ2RGB(imYIQ)

    else:
        # Normalize the image to [0,255] range and convert to integer
        imOrig = (imgOrig * 255).astype(np.uint8)
        # Calculate the histogram of the original image
        histOrg, bins = np.histogram(imOrig, bins=256, range=(0, 255))
        # Calculate the cumulative sum of the histogram and normalize it
        cumsum = histOrg.cumsum()
        cumsum_norm = (cumsum - cumsum.min()) / (cumsum.max() - cumsum.min())
        # Create the lookup table
        lut = (cumsum_norm * 255).astype(np.uint8)
        # Apply the lookup table to the original image
        imEq = lut[imOrig]
        # Calculate the histogram of the equalized image
        histEq, bins = np.histogram(imEq, bins=256, range=(0, 255))
        # Normalize the image back to [0,1] range
        imOrig = imOrig.astype(np.float32) / 255.0
        imEq = imEq.astype(np.float32) / 255.0


    return imEq, histOrg, histEq




def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # Check if RGB image and convert to YIQ
    if imOrig.ndim == 3 and imOrig.shape[2] == 3:
        imYIQ = transformRGB2YIQ(imOrig)
        im = imYIQ[:, :, 0]
    else:
        im = imOrig.copy()

    # Normalize image values to [0,255]
    im = (im * 255).astype(np.uint8)

    # Set initial segment division
    segment_size = 256 // nQuant
    z = np.arange(0, 256, segment_size)
    z[-1] = 255

    # Quantization iterations
    error_list = []
    qImage_list = []
    for i in range(nIter):
        # Find optimal quantization values
        q = []
        for k in range(nQuant):
            indices = np.logical_and(im >= z[k], im < z[k + 1])
            if np.sum(indices) == 0:
                # Avoid division by zero
                q_k = 0
            else:
                q_k = np.mean(im[indices])
            q.append(q_k)

        # Quantize image using optimal values
        qImage = np.zeros_like(im)
        for k in range(nQuant):
            indices = np.logical_and(im >= z[k], im < z[k + 1])
            qImage[indices] = q[k]

        # Calculate error
        error = np.sum(np.power(im - qImage, 2))
        error_list.append(error)

        # Update segment division
        for k in range(1, nQuant):
            z[k] = (q[k - 1] + q[k]) / 2

        # Add to result list
        qImage_list.append(qImage.astype(np.float32) / 255)


    # Check if RGB image and convert back from YIQ
    if imOrig.ndim == 3 and imOrig.shape[2] == 3:
        qImage_list_rgb = []
        for qImage in qImage_list:
            qImageYIQ = np.zeros_like(imYIQ)
            qImageYIQ[:, :, 0] = qImage
            qImageYIQ[:, :, 1:] = imYIQ[:, :, 1:]
            qImage_list_rgb.append(transformYIQ2RGB(qImageYIQ))
        return qImage_list_rgb, error_list
    else:
        return qImage_list, error_list
