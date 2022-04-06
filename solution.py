import cv2
import numpy as np

def GetLocation(move_type, env, CurrFrame):
    global PrevFrame
    GreyFrame = cv2.cvtColor(CurrFrame, cv2.COLOR_RGB2GRAY)
    if 'PrevFrame' not in globals():
        PrevFrame = GreyFrame
    Template = np.array(                                                         \
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
    return [{'coordinate' : coordinate, 'move_type' : move_type}]
