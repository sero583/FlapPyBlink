# FlapPyBlink

This project has been created for a class in our computer science study called AI in robotics.

This is an extension of the base game called [FlapPyBird](https://github.com/sourabhv/FlapPyBird). This extension has a blink detection implemented, which works as a control for the bird and basically simulates a space bar press when it notices a blink. The blink detection doesn't care about how many persons are in the image, so multiple persons can play this in theory.

# Requirements and customization

You will need on top of the base game requirements CV2 installed on your machine, so that the face recognition can work. The [Haarcascade files](https://github.com/opencv/opencv/tree/4.x/data/haarcascades) are from [OpenCV](https://github.com/opencv/opencv) and uses for face recognition. If you want to use your custom files, you can replace the files under haarcascades.


## Select the correct camera by its ID

My camera ID was 2, yours is most likely 0. If your camera doesn't work simply replace the ``CAMERA_ID`` variable at the very beginning of ``flappy.py in line 14`` to the ID you got. If you don't know it, just try by incrementing beginning from 0 (like 0, 1, 2, 3, ...) 

## Instructions on how to get this game running
   - Install [Python](https://www.python.org/downloads/) on your machine
   - Run in the terminal these commands in order to install pipenv and cv2:
      ```bash
      pip install pygame
      pip install pipenv
      pip install opencv-python
      ```
   - Now install the game with pipenv using in the root directory:
      ```bash
      pipenv install
      ```
   - After you installed everything with the previous two commands, run as a non-glasses user:
      ```bash
      pipenv run python flappy.py flappy.py
      ```
   - And as someone who wears glasses, run:
      ```bash
      pipenv run python flappy.py -glasses
      ```
   - Now the game should start running. If not, you might need to add to your PATH variable the freshly installed pipenv command.

## Change to your own haarcascades

   - For face recognition replace ``haarcascades/haarcascade_frontalface_default.xml``
   - For eye recognition replace ``haarcascades/haarcascade_eye.xml``
   - For eye recognitioin for people with glasses ``haarcascades/haarcascade_eye_tree_eyeglasses.xml``

## Tips:
   - Stand still during gameplay and don't move your head too much, otherwise the camera has to recognize your face first again and then your eyes
   - When game won't stop, force kill it by using a task manager or something similar
   - Use glasses mode, when you wear glasses (e.g. use start-for-glasses-users.cmd) or basically start flappy.py with the flag -glasses (e. g. py flappy.py -glasses)

## Known issues:
   - Game won't exit properly and remain in a deadlock
     - Probably linked to this experience: https://stackoverflow.com/questions/65115092/occasional-deadlock-in-multiprocessing-pool
   - Occasionally users nose (when showing the holes by looking a bit up) or mouth are recognized as eyes
     - This happens especially when your head is moving

### Note that this is just an experimental project and not a real game. You can have fun using it, but don't bother when it has issues.

# Credits

Many credits go out to the original creator [sourabhv](https://github.com/sourabhv) of the base game used in this project.