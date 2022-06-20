import csv
import os
import cv2

path = "C:/Users/khcho/Desktop/category_data_grayscale/bottom"
os.chdir(path)
files = os.listdir(path)

f = open("C:/Users/khcho/Desktop/label_bottom_modified.csv", "r", encoding = "utf-8")
lines = f.readlines()

count = 0

for file in files:
    img = cv2.imread(file)
    if lines[count].split(",")[4] == "tight\n":
        cv2.imwrite("C:/Users/khcho/Desktop/fit_data/bottom/tight/" + file, img)
    if lines[count].split(",")[4] == "normal\n":
        cv2.imwrite("C:/Users/khcho/Desktop/fit_data/bottom/normal/" + file, img)
    if lines[count].split(",")[4] == "loose\n":
        cv2.imwrite("C:/Users/khcho/Desktop/fit_data/bottom/loose/" + file, img)
    count = count + 1

f.close()