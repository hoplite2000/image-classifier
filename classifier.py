import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

content = sys.argv[1]

model = keras.applications.vgg16.VGG16()
image = keras.preprocessing.image.load_img(content, target_size=(224,224))
image = keras.preprocessing.image.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = keras.applications.vgg16.preprocess_input(image)
yhat = model.predict(image)
label = keras.applications.vgg16.decode_predictions(yhat)
label = label[0]
print('\n \n')
print("=================================================")
for id,name,confidence in label:
  print('%s (%.2f%%)' % (name, confidence*100))
print("=================================================")

img=matplotlib.image.imread(content)
imgplot = plt.imshow(img)
plt.show()