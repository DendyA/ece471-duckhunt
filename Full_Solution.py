import cv2
import numpy as np
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
    visionTypeToUse = comp_vis_type[0]

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
            ReadTemp = 0    # flag to read template
            ImgProc = 0     # flag to process input image (little effect for processes tried)
            BinIn = 0       # flag to bin the input image (only used for moving backgrounds, often degrades performance)
            MultGun = 0     # flag to use one shot for multiple targets (often better but takes more time, so a trade off)
            ShotGun = 0     # flag to use multiple shots for one target (usually degrades performance)

            # convert current frame to greyscale
            GreyFrame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

            # if previous frame is not defined, set it to the current greyscale frame
            if 'prev_frame' not in globals():
                prev_frame = GreyFrame

            # read or set the matching template
            if (ReadTemp):
                Template = cv2.imread("duckeye.png", cv2.IMREAD_GRAYSCALE) 
            else:
                # small default template originally derived from an image of a birds eye but made symmetric in x and y
                Template = np.array(                                                    \
                    [[ 47,  47,  46,  45,  42,  38,  39,  38,  42,  45,  46,  47,  47], \
                    [ 46,  44,  45,  51,  65,  85,  87,  85,  65,  51,  45,  44,  46], \
                    [ 46,  44,  49,  69, 107, 157, 163, 157, 107,  69,  49,  44,  46], \
                    [ 51,  63,  81, 107, 241, 179, 183, 179, 141, 107,  81,  63,  51], \
                    [ 63, 103, 140, 160, 154, 127, 123, 127, 154, 160, 140, 103,  63], \
                    [ 76, 145, 199, 207, 150,  44,  31,  44, 150, 207, 199, 145,  76], \
                    [ 77, 149, 204, 208, 144,  26,  13,  26, 144, 208, 204, 149,  77], \
                    [ 76, 145, 199, 207, 150,  44,  31,  44, 150, 207, 199, 145,  76], \
                    [ 63, 103, 140, 160, 154, 127, 123, 127, 154, 160, 140, 103,  63], \
                    [ 51,  63,  81, 107, 141, 179, 183, 179, 141, 107,  81,  63,  51], \
                    [ 46,  44,  49,  69, 107, 157, 163, 157, 107,  69,  49,  44,  46], \
                    [ 46,  44,  45,  51,  65,  85,  87,  85,  65,  51,  45,  44,  46], \
                    [ 47,  47,  46,  45,  42,  38,  39,  38,  42,  45,  46,  47,  47]], dtype=np.uint8)
            TemplateShape = Template.shape

            # if image processing flag is set to increase contrast 
            if (ImgProc):
                GreyFrame = cv2.equalizeHist(GreyFrame)
                GreyFrame = cv2.adaptiveThreshold(GreyFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

            # if bin input flag is set and moving background   
            if (BinIn and sum(sum(GreyFrame - prev_frame)) > 50000):
                # bin to black white and grey to try to eliminate rain
                BinFrame = np.full(GreyFrame.shape, 0, dtype=int)
                BinFrame = np.where(GreyFrame >= 85, 128, GreyFrame)
                BinFrame = np.where(GreyFrame >= 170, 220, GreyFrame)
                GreyFrame = BinFrame
                # bin template the same way
                BinTemp = np.full(Template.shape, 0, dtype=int)
                BinTemp = np.where(Template >= 85,130,Template)
                BinTemp = np.where(Template >= 170,255,Template)
                Template = BinTemp

            # create a difference frame to eliminate static backgronds and low any low contrast differences
            DiffFrame = np.where(GreyFrame == prev_frame, 0, GreyFrame)
            # save a new previous frame 
            prev_frame = GreyFrame

            # use template matching to find most likely template location
            Match = cv2.matchTemplate(image=DiffFrame, templ=Template, method=cv2.TM_CCOEFF)
            # find top left corner of the template match
            MinVal, MaxVal, MinLoc, MaxLoc = cv2.minMaxLoc(Match)
            # find middle of template and add coordinate and move type for output
            coordinate = (int((MaxLoc[1] + TemplateShape[1]/2)), int((MaxLoc[0] + TemplateShape[0]/2)))

            # set coordinate for the centre of the template for the best target
            out = [{'coordinate' : coordinate, 'move_type' : move_type}]

            # find multiple targets by masking best Match and looking for new probability maximums
            if (MultGun):
                Spread = 40 # size of the masking box (from the centre)
                NumTarg = 3 # number of shots
                for x in range(NumTarg):
                    Mask = np.full(Match.shape[:2], 255, dtype=np.uint8)
                    cv2.rectangle(Mask, (coordinate[1]-Spread,coordinate[0]-Spread), (coordinate[1]+Spread,coordinate[0]+Spread), 0, -1)
                    Match = cv2.bitwise_and(Match, Match, mask=Mask)
                    MinVal, MaxVal, MinLoc, MaxLoc = cv2.minMaxLoc(Match)
                    coordinate = (int((MaxLoc[1] + TemplateShape[1]/2)), int((MaxLoc[0] + TemplateShape[0]/2)))
                    out.append({'coordinate' : coordinate, 'move_type' : move_type})

            # take multiple shots around the best target to ensure a hit (up to 8 additional shots spread around the target)
            if (ShotGun):
                Spread = 30 # distance from the centre in each direction for additional shots
                out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]), 'move_type' : move_type})
                out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]), 'move_type' : move_type})
                out.append({'coordinate' : (coordinate[0],coordinate[1]+Spread), 'move_type' : move_type})
                out.append({'coordinate' : (coordinate[0],coordinate[1]-Spread), 'move_type' : move_type})
                #out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]+Spread), 'move_type' : move_type})
                #out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]+Spread), 'move_type' : move_type})
                #out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]-Spread), 'move_type' : move_type})
                #out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]-Spread), 'move_type' : move_type})
            return out

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
                prev_frame = greyScaleFrame
            diff_frame = np.where(greyScaleFrame == prev_frame, 0, greyScaleFrame)
            total = sum(sum(diff_frame))
            if (total > 50000):  # needs fix? triggered if non static background like rain - JE
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
            import rf
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
