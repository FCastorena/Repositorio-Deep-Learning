'''
        Momento de Retroalimentación Individual: Implementación de un modelo de Deep Learning.
                                Francisco Castorena Salazar, A00827756

Este modelo se realizó con ayuda de distintos recursos.

Referencias:

https://keras.io/api/applications/

https://drive.google.com/file/d/1yOuyWaHK5Trosk3OJAxlab54Sz8FoLTz/view?usp=drive_link

https://www.tensorflow.org/tutorials/images/classification?hl=es-419

https://github.com/nachi-hebbar/Transfer-Learning-ResNet-Keras/blob/main/ResNet_50.ipynb

https://www.youtube.com/watch?v=JcU72smpLJk

https://youtu.be/wruyZWre2sM?si=h7A3D1OEp3VAmocq

'''
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import filedialog
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

roses = list(data_dir.glob('roses/*'))
img_height,img_width=200,200
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=23,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=23,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(200,200,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Crear una instancia de la clase Tk
root = tk.Tk()
root.title("Predictor de Imágenes")

class_names = train_ds.class_names
labels =  train_ds.class_names

# Función para cargar y mostrar la imagen seleccionada
def load_img():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    # Redimensiona la imagen usando LANCZOS
    img = img.resize((200, 200), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img
    img_pred(file_path)

def img_pred(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = resnet_model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)
    resultado.config(text=f'Prediction: {predicted_class[0]} - {labels[predicted_class[0]]}')

# Botón para cargar la imagen
boton_cargar = tk.Button(root, text="Load Image", command=load_img)
boton_cargar.pack(pady=20)

# Etiqueta para mostrar la imagen
label = tk.Label(root)
label.pack()

# Etiqueta para mostrar la clase predicha
resultado = tk.Label(root, font=("Helvetica", 14))
resultado.pack(pady=20)

# Iniciar el bucle principal de Tkinter
root.mainloop()