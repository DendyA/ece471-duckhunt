import cv2
import numpy as np
import rf
from pathlib import Path
import matplotlib.pyplot as plt

comp_vis_type = ["Template Matching", "SIFT", "ML"]

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""


def GetLocation(move_type, env, current_frame):
    # time.sleep(1) #artificial one second processing time
    visionTypeToUse = comp_vis_type[2]

    # keep previous frame - JE
    global prev_frame

    global spriteKeys
    global spriteDescr

    greyScaleFrame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    # Use relative coordinates to the current position of the "gun", defined as an integer below
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
    # Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        if visionTypeToUse == comp_vis_type[0]:

            birdEye = cv2.imread("imgs/template_eye.png", cv2.IMREAD_GRAYSCALE)

            # Should be in WxHxD tuple form. (Where W is width, h is height, and d is depth (num of channels)).
            birdEyeShape = birdEye.shape
            currentFrameShape = current_frame.shape

            # find and remove background - JE
            if 'prev_frame' not in globals():
                print("Set Prev frame")
                prev_frame = greyScaleFrame
            diff_frame = np.where(greyScaleFrame == prev_frame, 0, greyScaleFrame)
            total = sum(sum(diff_frame))
            if (total > 50000):  # needs fix? triggered if non static background like rain - JE
                print("Set Prev frame, total: ", total)
                prev_frame = greyScaleFrame
            prev_frame = greyScaleFrame
            greyScaleFrame = diff_frame

            isBird = cv2.matchTemplate(image=greyScaleFrame, templ=birdEye, method=cv2.TM_CCOEFF)
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(
                isBird)

            top_left = max_location
            bottom_right = (top_left[0] + birdEyeShape[0], top_left[1] + birdEyeShape[1])

            coordinate = (int((bottom_right[1] + top_left[1]) / 2),
                          int((bottom_right[0] + top_left[0]) / 2))
        elif visionTypeToUse == comp_vis_type[1]:
            if 'spriteKeys' not in globals() or 'spriteDescr' not in globals():
                keypoints = []
                descriptors = []
                sift = cv2.SIFT.create()
                pathlist = Path("./sprites/").rglob('*.png')
                for path in pathlist:
                    tempFile = cv2.imread("./" + str(path), cv2.IMREAD_GRAYSCALE)
                    key, descr = cv2.SIFT.detectAndCompute(sift, tempFile, None)
                    keypoints.append(key)
                    descriptors.append(descr)

                spriteKeys = keypoints
                spriteDescr = descriptors

            if 'prev_frame' not in globals():
                print("Set Prev frame")
                prev_frame = greyScaleFrame
            diff_frame = np.where(greyScaleFrame == prev_frame, 0, greyScaleFrame)
            total = sum(sum(diff_frame))
            if (total > 50000):  # needs fix? triggered if non static background like rain - JE
                print("Set Prev frame, total: ", total)
                prev_frame = greyScaleFrame
            prev_frame = greyScaleFrame
            greyScaleFrame = diff_frame

            sift = cv2.SIFT.create()

            currKey, currDescr = cv2.SIFT.detectAndCompute(sift, greyScaleFrame, None)
            if len(currKey) == 0:
                coordinate = (0, 0)
            else:
                bruteMatcher = cv2.BFMatcher.create(normType=cv2.NORM_L2SQR)
                smallestDistance = []
                for desc in spriteDescr:
                    foundMatches = cv2.BFMatcher.match(bruteMatcher, currDescr, desc)
                    smallestDistance += list(foundMatches)

                smallestDistance.sort(key=lambda m: m.distance)
                coordX = 0
                coordY = 0
                numOfPointsToAv = 1
                for i in range(numOfPointsToAv):
                    coordX += currKey[smallestDistance[i].queryIdx].pt[0]
                    coordY += currKey[smallestDistance[i].queryIdx].pt[1]
                coordX /= numOfPointsToAv
                coordY /= numOfPointsToAv

                coordinate = (int(coordY), int(coordX))

        elif visionTypeToUse == comp_vis_type[2]:
            # machine learning
            # current_frame : np.ndarray (width, height, 3), np.uint8, RGB

            # else:
            #     # Read image
            # self.count += 1
            # img0 = cv2.imread(path)  # BGR
            # assert img0 is not None, f'Image Not Found {path}'
            # s = f'image {self.count}/{self.nf} {path}: '

            # # Padded resize
            # img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

            # # Convert
            # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # img = np.ascontiguousarray(img)

            # plt.imshow(current_frame)
            # frame = np.transpose(current_frame)
            # frame = np.reshape(frame,(1,3,768,1024))
            # print("frame")
            # print(frame.shape)
            # plt.imshow(frame)

            # coordinate = rf.predict_yolov5(frame)[0] #takes first coordinate set
            result = rf.predict_yolov5(current_frame)
            return_vals = []
            if not result:
                print("no ducks")
                coordinate = (0,0)
            else:
                # coordinate = result[0]
                for res in result:
                    return_vals.append({'coordinate': res, 'move_type': move_type})

                print("ducks found", len(return_vals))
                print(return_vals)
                return return_vals

            # coordinate = rf.predict_yolov5(current_frame)[0]  # takes first coordinate set
            # coordinate = rf.predict_yolov5_w_screenshots()
            # coordinate = (0,0)

        else:
            coordinate = env.action_space_abs.sample()

    return [{'coordinate': coordinate, 'move_type': move_type}]
