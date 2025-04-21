import cv2
import numpy as np
from PIL import Image

def detect_blur(pil_image):
    img = np.array(pil_image.convert("L"))
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var

def check_brightness(pil_image):
    img = np.array(pil_image.convert("L"))
    return np.mean(img)

def check_contrast(pil_image):
    img = np.array(pil_image.convert("L"))
    return img.std()

def check_resolution(pil_image):
    width, height = pil_image.size
    return width, height
