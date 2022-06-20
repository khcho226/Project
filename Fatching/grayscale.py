import os
import cv2
from PIL import Image

path = "C:/Users/khcho/Desktop/category_data/top"
os.chdir(path)
files = os.listdir(path)

for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("C:/Users/khcho/Desktop/top/" + file, img)