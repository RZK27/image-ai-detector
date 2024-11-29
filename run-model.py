import numpy as np
from tabulate import tabulate
import cv2
import os
from tensorflow.keras import models

DIRECTORY = os.path.dirname(__file__)
CATEGORIES = ["real", "fake"]
IMG_SIZE = 256

images = []
img_names = []

for img in os.listdir(os.path.join(DIRECTORY, 'images')):
    img_path = os.path.join(DIRECTORY, 'images', img)
    img_arr = cv2.imread(img_path)
    img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    images.append(img_arr)
    img_names.append(img)

images = np.array(images) / 255
img_names = np.array(img_names)

filename = 'model.sav'
model = models.load_model('model.keras')

predictions = model.predict(images)

table_data = []
for num in range(len(img_names)):
    percent = predictions[num] * 100
    if percent < 50:
        label = "real"
    else:
        label = "fake"
    table_data.append([img_names[num], label, f"{percent[0]:.0f}%"])

headers = ["Image File Name", "Real or Fake?", "Confidence"]

print(tabulate(table_data, headers=headers, tablefmt="grid"))