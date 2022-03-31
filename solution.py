import time
import cv2
import numpy as np

comp_vis_type = ["Template Matching"]

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
def GetLocation(move_type, env, current_frame):
    # time.sleep(1) #artificial one second processing time
    visionTypeToUse = comp_vis_type[0]

    # keep previous frame - JE
    global prev_frame
    
    #Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative":
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample() 
    #Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        if visionTypeToUse == comp_vis_type[0]:
            greyScaleFrame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            
            birdEye = cv2.imread("template_eye.png", cv2.IMREAD_GRAYSCALE)

            # Should be in WxHxD tuple form. (Where W is width, h is height, and d is depth (num of channels)).
            birdEyeShape = birdEye.shape
            currentFrameShape = current_frame.shape

            # find and remove background - JE
            if 'prev_frame' not in globals():
                print("Set Prev frame")
                prev_frame = greyScaleFrame
            diff_frame = np.where(greyScaleFrame == prev_frame,0,greyScaleFrame)
            total = sum(sum(diff_frame))
            if (total > 50000): # needs fix? triggered if non static background like rain - JE
                print("Set Prev frame, total: ",total)
                prev_frame = greyScaleFrame
            prev_frame = greyScaleFrame
            greyScaleFrame = diff_frame 

            isBird = cv2.matchTemplate(image=greyScaleFrame, templ=birdEye, method=cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(isBird)

            top_left = max_loc
            bottom_right = (top_left[0] + birdEyeShape[0], top_left[1] + birdEyeShape[1])

            coordinate = (int((bottom_right[1] + top_left[1]) / 2), int((bottom_right[0] + top_left[0]) / 2))
        else:
            coordinate = env.action_space_abs.sample()

    return [{'coordinate' : coordinate, 'move_type' : move_type}]
