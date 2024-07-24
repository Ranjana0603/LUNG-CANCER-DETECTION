import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from keras.models import load_model
from tkinter import *
import tkinter.messagebox
import PIL.Image
import PIL.ImageTk
from tkinter import filedialog

CATEGORIES = ["Adenocarcinoma","Normal","Squamous Carcinoma"]

root = Tk()
root.title("Lung Caner Analysis")
root.state('zoomed')
root.configure(bg='#D3D3D3')
root.resizable(width = True, height = True) 
value = StringVar()
panel = Label(root)
model = tf.keras.models.load_model("CNN.model")

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def prepare(file):
    IMG_SIZE = 150
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
    #img_array = clahe.apply(img_array)
    #median  = cv2.medianBlur(img_array.astype('uint8'), 5)
    #median = 255-median
    #ret,thresh = cv2.threshold(median.astype('uint8'),165,255,cv2.THRESH_BINARY_INV)\
    img_array=cv2.equalizeHist(img_array)
    #img_array = cv2.Canny(img_array, threshold1=40, threshold2=70)
    img_array = cv2.medianBlur(img_array,1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
def detect(filename):
    prediction = model.predict(prepare(filename))
    prediction = list(prediction[0])
    print(prediction)
    l=CATEGORIES[prediction.index(max(prediction))]
    print(CATEGORIES[prediction.index(max(prediction))])
    value.set(CATEGORIES[prediction.index(max(prediction))])
    i=int(prediction.index(max(prediction)))
    
def ClickAction(event=None):
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((250,250))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image = img)
    panel.image = img
    panel = panel.place(relx=0.435,rely=0.3)
    detect(filename)
    

button = Button(root, text='CHOOSE FILE', font=(None, 18), activeforeground='red', bd=20, bg='cyan', relief=RAISED, height=3, width=20, command=ClickAction)
button = button.place(relx=0.40, rely=0.05)
result = Label(root, textvariable=value, font=(None, 20))
result = result.place(relx=0.465,rely=0.7)
root.mainloop()



