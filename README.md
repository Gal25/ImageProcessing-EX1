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

    ![צילום מסך 2023-04-09 161558](https://user-images.githubusercontent.com/92460450/230775132-65c0137a-27d8-4133-89b4-14df1a6e70c4.jpg)




![צילום מסך 2023-04-09 161726](https://user-images.githubusercontent.com/92460450/230775169-2942fe49-6f1f-4200-b2ba-02beeb89a668.jpg)


3. *transformRGB2YIQ*(imgRGB: np.ndarray)- Transform an RGB image into the YIQ color space
     ![צילום מסך 2023-04-09 161911](https://user-images.githubusercontent.com/92460450/230775025-6fc2eab1-a675-4349-a8e9-834ef327b069.jpg)  

4. *transformYIQ2RGB*(imgYIQ: np.ndarray) - Transform an YIQ image into the RGB color space


   
5. *hsitogramEqualize*(imgOrig: np.ndarray) - Performs histogram equalization of a given grayscale or RGB image.

![צילום מסך 2023-04-09 162242](https://user-images.githubusercontent.com/92460450/230775307-33a1b2f0-0084-4d83-8a02-90ee88e08769.jpg)

![צילום מסך 2023-04-09 162314](https://user-images.githubusercontent.com/92460450/230775314-a507a825-c32c-4787-97f1-e7c55372d8d8.jpg)

    
    
6. *quantizeImage*(imOrig: np.ndarray, nQuant: int, nIter: int)
    Performs optimal quantization of a given grayscale or RGB image
    ![צילום מסך 2023-04-09 162501](https://user-images.githubusercontent.com/92460450/230775393-ed9e99a9-a5c3-44cb-bdaf-b6a5bc1bda24.jpg)

![צילום מסך 2023-04-09 162511](https://user-images.githubusercontent.com/92460450/230775402-fe546dd4-d2e4-43b9-9c92-9e30579e5d2f.jpg)



![צילום מסך 2023-04-09 162523](https://user-images.githubusercontent.com/92460450/230775407-e3328dc5-b0c1-4173-b6a0-a21f344b06d3.jpg)

    
    
        
    
*gamma:*    
function that performs gamma correction on an image with a given gamma



Python 3.9.10
