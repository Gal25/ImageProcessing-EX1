# ImageProcessing-EX1

The main purpose of this exercise is to get you acquainted with Python's basic syntax and some of its
image processing facilities. This exercise covers:

• Loading grayscale and RGB image representations.

• Displaying figures and images.

• Transforming RGB color images back and forth from the YIQ color space.

• Performing intensity transformations: histogram equalization.

• Performing optimal quantization


*ex1_utils:*
1. *imReadAndConvert*(filename: str, representation: int) - Reading an image into a given representation
  

   
2. *imDisplay*(filename: str, representation: int) - Reads an image as RGB or GRAY_SCALE and displays it

 ![צילום מסך 2023-04-09 161558](https://user-images.githubusercontent.com/92460450/230774884-14270e20-c685-4c2c-aadf-![צילום מסך 2023-04-09 161726](https://user-images.githubusercontent.com/92460450/230774925-cfabe556-db11-4e03-840c-afeb48df09df.jpg)
80e9811d35ef.jpg)

    
3. *transformRGB2YIQ*(imgRGB: np.ndarray)- Transform an RGB image into the YIQ color space
     ![צילום מסך 2023-04-09 161911](https://user-images.githubusercontent.com/92460450/230775025-6fc2eab1-a675-4349-a8e9-834ef327b069.jpg)  

4. *transformYIQ2RGB*(imgYIQ: np.ndarray)
   Transform an YIQ image into the RGB color space
   
5. *hsitogramEqualize*(imgOrig: np.ndarray)
    Performs histogram equalization of a given grayscale or RGB image.
    
    
6. *quantizeImage*(imOrig: np.ndarray, nQuant: int, nIter: int)
    Performs optimal quantization of a given grayscale or RGB image
        
    
*gamma:*    
function that performs gamma correction on an image with a given gamma

Python 3.9.10
