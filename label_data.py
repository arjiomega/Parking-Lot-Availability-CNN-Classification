from pathlib import Path
import os
import csv

import matplotlib.pyplot as plt
import cv2

DATA_DIR = Path(Path().absolute(),'data')

label_list = []

name_list = sorted(os.listdir(str(Path(DATA_DIR,'training'))))

for i,img in enumerate(name_list):
    print(f"{i} out of {len(name_list)}")
    print(img)
    load = cv2.imread(str(Path(DATA_DIR,'training',img)))

    cv2.imshow('img',load)
    key = cv2.waitKey(0)

    if key == ord('1'):
        label_list.append('car')
    if key == ord('2'):
        label_list.append('empty')

    if key == ord('q'):
        break


with open('test.csv',"w") as f:
    writer = csv.writer(f)

    writer.writerow(['image_name','label'])

    for i in range(len(name_list)):
        writer.writerow([name_list[i],label_list[i]])

