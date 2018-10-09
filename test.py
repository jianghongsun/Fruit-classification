#加载图像对物体进行预测
# #
import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import skimage
from skimage import io, transform
from IPython.display import Image, display
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LSTM, Input
from keras.models import Model
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import os
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)
KTF.set_session(session)
img_size = 100

#Load Model from disk
json_file=open('cnn_model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("cnn_model.h5")
# print("CNN model loaded from disk")
#load test image 
img_filename="E:\\code\\fruits\\ra.jpg"
printouts=[]
current_img=io.imread(img_filename)
current_img=transform.resize(current_img,(img_size,img_size))
current_img=np.asarray(current_img)
current_img=np.asarray([current_img])
 y_pred = loaded_model.predict(current_img, None, 0).argmax(axis=-1)
# print(y_pred)

list1=['Apple Golden 3', 'Apple Granny Smith', 'Apple Red 1', 'Apple Red Yellow', 'Avocado', 'Banana Red', 'Cantaloupe 2',
        'Cherry 2', 'Cherry Rainier', 'Cherry Wax Yellow', 'Cocos', 'Granadilla', 'Grapefruit White', 'Huckleberry',  'Kaki',
        'Lemon', 'Lemon Meyer', 'Lychee', 'Mandarine', 'Mango', 'Nectarine','Passion Fruit', 'Pear Abate',
        'Pear Monster', 'Physalis with Husk', 'Plum', 'Pomegranate',  'Quince', 'Raspberry', 'Strawberry', 'Tamarillo',
        'Tangelo', 'Tomato Cherry Red', 'Tomato Maroon']
for index in  range(len(list1)):
     if y_pred==[index]:
         print("recognitioned fruit was ",list1[index])