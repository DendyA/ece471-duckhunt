#import time
import cv2
import numpy as np

def GetLocation(move_type, env, CurrFrame):
    global PrevFrame

    ReadTemp = 0
    ImgProc = 0
    ShotGun = 0
    MultiShot = 1

# this should be removed. Only for loading previous images, trying different processing and writing images
    global shotcount
    if 'shotcount' not in globals():
        shotcount = 0
    shotcount = shotcount + 1
    print(shotcount)
    #if shotcount >= 10 and shotcount < 50:
    #    CurrFrame = cv2.imread('figures_rain/CurrFrame_'+str(shotcount)+'.jpg')
    #    CurrFrame = cv2.cvtColor(CurrFrame, cv2.COLOR_BGR2RGB)
        
    GreyFrame = cv2.cvtColor(CurrFrame, cv2.COLOR_RGB2GRAY)
        
    if (ImgProc): # not tested
        GreyFrame = cv2.equalizeHist(GreyFrame)
        clahe=cv2.createCLAHE(clipLimit=40)
        GreyFrame=clahe.apply(GreyFrame)
        GreyFrame = cv2.adaptiveThreshold(GreyFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    if 'PrevFrame' not in globals():
        PrevFrame = GreyFrame
        
    if (ReadTemp):
        Template = cv2.imread("duckeye.png", cv2.IMREAD_GRAYSCALE) 
    else:
        Template = np.array(                                                     \
        [[160, 152, 136, 119, 130, 163, 192, 188, 148, 119, 104, 103, 111, 126], \
         [146, 133, 113, 102, 128, 179, 219, 213, 158, 113,  85,  74,  76,  90], \
         [117, 110, 113, 130, 165, 209, 239, 235, 193, 151, 114,  84,  63,  61], \
         [ 85,  93, 131, 185, 221, 241, 250, 249, 237, 211, 170, 117,  68,  46], \
         [ 71,  89, 147, 217, 246, 255, 255, 255, 255, 239, 197, 136,  72,  40], \
         [ 71,  87, 143, 212, 240, 240, 233, 234, 244, 233, 193, 133,  72,  41], \
         [ 72,  90, 147, 213, 222, 190, 156, 162, 208, 222, 194, 137,  73,  42], \
         [ 73,  94, 154, 218, 207, 134,  69,  80, 168, 213, 199, 144,  75,  42], \
         [ 74,  97, 160, 222, 197,  98,  13,  26, 144, 208, 204, 149,  77,  42], \
         [ 75,  96, 156, 218, 198, 109,  31,  44, 150, 207, 199, 145,  76,  42], \
         [ 83,  81, 109, 151, 161, 144, 123, 127, 154, 160, 140, 103,  63,  44], \
         [ 88,  66,  65,  86, 116, 155, 183, 179, 141, 107,  81,  63,  51,  46], \
         [ 86,  58,  46,  51,  78, 126, 163, 157, 107,  69,  49,  44,  46,  46], \
         [ 83,  62,  52,  47,  55,  73,  87,  85,  65,  51,  45,  44,  46,  47], \
         [ 94,  81,  68,  54,  48,  42,  39,  38,  42,  45,  46,  47,  47,  46], \
         [112, 115,  96,  68,  58,  51,  46,  45,  45,  46,  47,  47,  46,  46]], dtype=np.uint8)

    TemplateShape = Template.shape
    DiffFrame = np.where(GreyFrame == PrevFrame,0,GreyFrame)
    PrevFrame = GreyFrame
    Match = cv2.matchTemplate(image=DiffFrame, templ=Template, method=cv2.TM_CCOEFF)
    MinVal, MaxVal, MinLoc, MaxLoc = cv2.minMaxLoc(Match)
    coordinate = (int((MaxLoc[1] + TemplateShape[1]/2)), int((MaxLoc[0] + TemplateShape[0]/2)))

    out = [{'coordinate' : coordinate, 'move_type' : move_type}]

    if (MultiShot): #Works but not with notable improvement.
        for x in range(3):
            masktopleft = (coordinate[1]-30,coordinate[0]-30)
            maskbotright = (coordinate[1]+30,coordinate[0]+30)
            mask = np.zeros(Match.shape[:2], dtype="uint8")
            cv2.rectangle(mask, masktopleft, maskbotright, 255, -1)
            mask = ~mask
            Match = cv2.bitwise_and(Match, Match, mask=mask)
            if shotcount == 10:
                cv2.imwrite('figures/mask_'+str(x)+'.jpg',mask)
                cv2.imwrite('figures/Match_'+str(x)+'.jpg',Match)
            MinVal, MaxVal, MinLoc, MaxLoc = cv2.minMaxLoc(Match)
            coordinate = (int((MaxLoc[1] + TemplateShape[1]/2)), int((MaxLoc[0] + TemplateShape[0]/2)))
            out.append({'coordinate' : coordinate, 'move_type' : move_type})
    if (ShotGun): # doesnt seem to work well
        spread = 30
        out.append({'coordinate' : (coordinate[0]+spread,coordinate[1]), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]-spread,coordinate[1]), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0],coordinate[1]+spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0],coordinate[1]-spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]+spread,coordinate[1]+spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]-spread,coordinate[1]+spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]+spread,coordinate[1]-spread), 'move_type' : move_type})
        out.append({'coordinate' : (coordinate[0]-spread,coordinate[1]-spread), 'move_type' : move_type})

# this should be removed. Only for saving images
    if shotcount >= 10 and shotcount < 100:
        Match = cv2.normalize(Match, None, 0, 255, cv2.NORM_MINMAX)
        for res in out:
            coordinate  = res['coordinate']
            Match = cv2.circle(Match,(coordinate[1],coordinate[0]),radius = 10,color = (255,0,0))
            DiffFrame = cv2.circle(DiffFrame,(coordinate[1],coordinate[0]),radius = 10,color = (255,0,0))
            CurrFrame = cv2.circle(CurrFrame,(coordinate[1],coordinate[0]),radius = 10,color = (255,0,0))
        cv2.imwrite('figures/CurrFrame_'+str(shotcount)+'.jpg',cv2.cvtColor(CurrFrame, cv2.COLOR_BGR2RGB))
        cv2.imwrite('figures/DiffFrame_'+str(shotcount)+'.jpg',DiffFrame)
        #cv2.imwrite('figures/Match_'+str(shotcount)+'.jpg',Match)
    #print(out)
    return out
