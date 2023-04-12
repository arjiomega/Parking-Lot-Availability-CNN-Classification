from pathlib import Path
import pickle

import tensorflow as tf
import numpy as np
import cv2



DATA_DIR = Path(Path().absolute(),'data')

cap = cv2.VideoCapture(str(Path(DATA_DIR,'parking_vid.mp4')))

with open('CarParkPos',"rb") as f:
    posList = pickle.load(f)

model = tf.keras.models.load_model('parkingML.h5')

width, height = (60,30)


cv2.namedWindow('image')

def checkParkingSpace(imgProcessed):
    slots_avail = len(posList)
    image_list = []
    for i,pos in enumerate(posList):

        x,y = pos
        imgCrop = imgProcessed[y:y+height,x:x+width]

        imgCrop = cv2.resize(imgCrop, (96, 96))
        imgCrop = np.round(imgCrop)
        #imgCrop = np.expand_dims(imgCrop,axis=0)

        image_list.append(imgCrop)

    out = np.array(image_list)
    predictions = model.predict_on_batch(out).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 1, 0)
    predictions = predictions.numpy()

    for i,pos in enumerate(posList):
        x,y = pos
        imgCrop = imgProcessed[y:y+height,x:x+width]

        # empty
        if predictions[i] == 0:
            color = (0,255,0) # green
            #text_color = (0,0,0)
            #text = "free"
        else:
            color = (0,0,255)
            #text_color = (255,255,255)
            #text = "used"
            slots_avail -= 1

        cv2.rectangle(img,(x,y),(x+width,y+height),color,2)

        sub_img = img[y:y+height,x:x+width]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        white_rect[:] = color
        img[y:y+height,x:x+width] = cv2.addWeighted(sub_img,0.8,white_rect,0.2,1.0)

    # car count
    cv2.rectangle(img,(0,0),(0+200,0+30),(0,0,0),-1)
    cv2.putText(img,f"Slots available: {slots_avail}/{len(posList)}",(0,0+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)


# WRITE
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920,1080))

while True:
    # first one gives current position, second gives total number of frames present in video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # reset frame if we reach total number of frames
        #cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        print("================================DONE=================================")
    print(cap.get(cv2.CAP_PROP_POS_FRAMES), 'out of ',cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, img = cap.read()

    checkParkingSpace(img)


    out.write(img)
    cv2.imshow("image",img)

    #cv2.imshow("ImageBlur",imgDilate)
    c = cv2.waitKey(1)

    if c & 0xFF == ord('q'):
        break