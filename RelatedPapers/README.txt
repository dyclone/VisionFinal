How To Run The Shape Detecction Code
- Make sure all the files are in the same directory
- Press the run button or run command to run the file ContourFinal.py
- From the list of 10 images, in the console enter the number of the image you want to see shape detection on. Afterwards
you will be asked which method to use. One is the douglas peuker algorithm which was discussed in class. The second
algorithm is one that was mention from previous research
- Once the inputs are in wait for two window to pop up, if they don't look on your toolbar for two windows
- One window is a trackbar with three bars. One for threshold 1, threshold 2 and area. Both threshold bars can be moved
to make edges appear or disappear within the image. The edges play a heavy role in the process of being able to identify
shapes. Area affects which contours get displayed. At default the area is set to zero so all contours with an area
greater than zero are currently visible. Slide the bar to the left or right in order to increase the area of the
contour required to be shown.
- The second window shows the input image and the results of our program;
    Column 1 row 1: input image
    Column 1 row 2: input image after gaussian blur and grayscale
    Column 2 row 1: Canny edge detection on input image
    Column 2 row 2: Shape detection on edges from canny edge detection
    Column 3 row 1: Canny edge detection with dilated lines
    Column 3 row 2: Shape detection on edges with dilated lines
- The program will continue to run until you press the q key
- Rerun the program to run the program on another image or another algorithm
