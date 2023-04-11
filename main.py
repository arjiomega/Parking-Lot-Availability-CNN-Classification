from pathlib import Path
import pickle
import os

import cv2

DATA_DIR = Path(Path().absolute(),'data')

cap = cv2.VideoCapture(str(Path(DATA_DIR,'parking_vid.mp4')))


with open('CarParkPos',"rb") as f:
    posList = pickle.load(f)

width, height = (60,30)
data_count = len(os.listdir(str(Path(DATA_DIR,'training'))))

label_list = []

while True:
    curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # first one gives current position, second gives total number of frames present in video
    if curr_frame == total_frames:
        # reset frame if we reach total number of frames
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, frame = cap.read()


    c = cv2.waitKey(1)

    if c & 0xFF == ord('q'):
        break

    if c & 0xFF == ord('c'):

        for index,pos in enumerate(posList):
            print(f'saving {index+1}/{len(posList)}')
            x,y = pos
            imgCrop = frame[y:y+height,x:x+width]

            status = cv2.imwrite(str(Path(DATA_DIR,'training','frame_{f}_slotpos_{a}'.format(f=int(curr_frame),a=(data_count+index))))+'.png',imgCrop)
            if not status:
                print('frame_{f}_slotpos_{a}'.format(f=int(curr_frame),a=(data_count+index)))
            #cv2.imshow('slotpos_{a}'.format(a=(data_count+index)),imgCrop)

        data_count = len(os.listdir(str(Path(DATA_DIR,'training'))))
        print('total training images: ',data_count)

    for index,pos in enumerate(posList):
        x,y = pos
        imgCrop = frame[y:y+height,x:x+width]

        color = (0,255,0)

        cv2.rectangle(frame,(x,y),(x+width,y+height),color,2)
        text_color = (255,255,255)
        cv2.putText(frame,str(index),(x,y+height-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,text_color,1)

    cv2.imshow('test',frame)


