from pathlib import Path
import pickle
import sys
import os

import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from config import config

from model import mobile_net_v2

model = mobile_net_v2()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']  )

train_ds = tf.keras.utils.image_dataset_from_directory(
  Path(config.DATA_DIR,'training'),
  validation_split=0.1,
  subset='training',
  seed=55,
  image_size=(96, 96),
  batch_size=32,
  shuffle=True)

valid_ds = tf.keras.utils.image_dataset_from_directory(
  Path(config.DATA_DIR,'training'),
  validation_split=0.1,
  subset='validation',
  seed=55,
  image_size=(96, 96),
  batch_size=32,
  shuffle=True)

classes = train_ds.class_names

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('parkingML.h5', save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]

history = model.fit(train_ds,epochs=50,validation_data=valid_ds,callbacks=callbacks)


with open('parkingML.pkl','wb') as f:
    pickle.dump(model,f)

cars = os.listdir(Path(config.DATA_DIR,'training','empty'))

for car in cars[:10]:
    img = tf.keras.utils.load_img(Path(config.DATA_DIR,'training','empty',car),target_size=(96,96))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    prediction = model.predict(img_array)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5,0,1)
    print(car,": ",classes[int(prediction)])


