import cv2
import numpy as np

def GetLocation(move_type, env, current_frame):
    global prev_frame

    # User flags
    ReadTemp = 0    # flag to read template
    ImgProc = 0     # flag to process input image (little effect for processes tried)
    BinIn = 0       # flag to bin the input image (only used for moving backgrounds, often degrades performance)
    MultGun = 0     # flag to use one shot for multiple targets (often better but takes more time, so a trade off)
    ShotGun = 0     # flag to use multiple shots for one target (usually degrades performance)
    MovBack = 1     # try to detect moving background and turn on improvements

    # convert current frame to greyscale
    greyScaleFrame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    # if previous frame is not defined, set it to the current greyscale frame
    if 'prev_frame' not in globals():
        prev_frame = greyScaleFrame
        
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

    # check for moving background
    if (MovBack):
        if sum(sum(greyScaleFrame - prev_frame)) > 50000:
            ImgProc = 1
            BinIn = 1
            MultGun = 1
            ShotGun = 1
            
    # if image processing flag is set to increase contrast 
    if (ImgProc):
        greyScaleFrame = cv2.equalizeHist(greyScaleFrame)
        greyScaleFrame = cv2.adaptiveThreshold(greyScaleFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # if bin input flag is set and moving background   
    if (BinIn and sum(sum(greyScaleFrame - prev_frame)) > 50000):
        # bin to black white and grey to try to eliminate rain
        BinFrame = np.full(greyScaleFrame.shape, 0, dtype=int)
        BinFrame = np.where(greyScaleFrame >= 85, 128, greyScaleFrame)
        BinFrame = np.where(greyScaleFrame >= 170, 220, greyScaleFrame)
        greyScaleFrame = BinFrame
        # bin template the same way
        BinTemp = np.full(Template.shape, 0, dtype=int)
        BinTemp = np.where(Template >= 85,130,Template)
        BinTemp = np.where(Template >= 170,255,Template)
        Template = BinTemp
        
    # create a difference frame to eliminate static backgronds and low any low contrast differences
    DiffFrame = np.where(greyScaleFrame == prev_frame, 0, greyScaleFrame)
    # save a new previous frame 
    prev_frame = greyScaleFrame

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
        Spread = 40 # distance from the centre in each direction for additional shots
        out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0],coordinate[1]+Spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0],coordinate[1]-Spread), 'move_type' : move_type})
        # out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]+Spread), 'move_type' : move_type})
        # out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]+Spread), 'move_type' : move_type})
        # out.append({'coordinate' : (coordinate[0]+Spread,coordinate[1]-Spread), 'move_type' : move_type})
        # out.append({'coordinate' : (coordinate[0]-Spread,coordinate[1]-Spread), 'move_type' : move_type})

    return out
