# ImageProcessing-EX1

The main purpose of this exercise is to get you acquainted with Python's basic syntax and some of its
image processing facilities. This exercise covers:
• Loading grayscale and RGB image representations.
• Displaying gures and images.
• Transforming RGB color images back and forth from the YIQ color space.
• Performing intensity transformations: histogram equalization.
• Performing optimal quantization


*ex1_utils:*
1. *imReadAndConvert*(filename: str, representation: int)
   Reading an image into a given representation
   
2. *imDisplay*(filename: str, representation: int)
    Reads an image as RGB or GRAY_SCALE and displays it
    
3. *transformRGB2YIQ*(imgRGB: np.ndarray)
    Transform an RGB image into the YIQ color space
    
4. *transformYIQ2RGB*(imgYIQ: np.ndarray)
   Transform an YIQ image into the RGB color space
   
5. *hsitogramEqualize*(imgOrig: np.ndarray)
    Performs histogram equalization of a given grayscale or RGB image.
    
    
6. *quantizeImage*(imOrig: np.ndarray, nQuant: int, nIter: int)
    Performs optimal quantization of a given grayscale or RGB image
        
    
*gamma:*    
function that performs gamma correction on an image with a given gamma

Python 3.9.10
