# Control Your Mouse with Mediapipe Handtracking in Python

**Note: This implementation has only been tested on MacOS and may not work on Windows/Linux systems.**

## How to use

* Put your hand infront of your Webcam

* To perform a left click, tap your thumb and index finger while keeping your middle finger pointing up.
* For a right click, tap your thumb and middle finger while keeping your index finger pointing up.
  
## Requirements

* **Python 3.11** _(This version is required by Mediapipe)_

* **OpenCV for Python 3.11**

* **PyAutoGUI for Python 3.11**

## Settings

If the project fails to launch or launches with the wrong webcam, you may need to adjust the following value in `Main_HandTracking.py`:

```python
# Opens the webcam; input 1 is not always the webcam. If it is not, the project will not launch.
# You might need to play with the value to find your desired webcam.
cap = cv2.VideoCapture(1)
```

To change the sensitivity of the mouse, modify the multiplication factor for mouse speed:

```python
# Mouse Sensitivity
X_multi = 1.5
Y_multi = 1.5
```
Limit the number of hands that can be traked at one time:
```python
# set how manny hands it can detect at one time
# (More hands can somewhat effect performance)
Num_of_hands = 1
```


Adjust these settings as necessary to optimize the performance of the hand tracking and mouse control functionality.
