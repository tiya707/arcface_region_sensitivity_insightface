import numpy as np

def mask_region(face_img, region="eyes"):
    h, w, _ = face_img.shape
    masked = face_img.copy()

    if region == "eyes":
        masked[int(0.2*h):int(0.45*h), :] = 0

    elif region == "nose":
        masked[int(0.35*h):int(0.6*h), int(0.3*w):int(0.7*w)] = 0

    elif region == "mouth":
        masked[int(0.6*h):h, :] = 0

    elif region == "upper":
        masked[0:int(0.5*h), :] = 0

    elif region == "lower":
        masked[int(0.5*h):h, :] = 0

    return masked