from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog

import seaborn as sns
import os
import cv2
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import xgboost as xgb
from PIL import Image

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import model_from_json

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


main = tkinter.Tk()
main.title('AI-ENHANCED MICROSCOPIC IMAGE CLASSIFICATION FOR RBC INFECTIONS')
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")


global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"



def uploadDataset():
    global filename,categories,categories,path
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
    
def imageProcessing():
    text.delete('1.0', END)
    global X, Y, model_folder, filename,X_file,Y_file
    path = r"Dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    


    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")
    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        print("X and Y arrays loaded successfully.")
    else:
        X = [] # input array
        Y = [] # output array
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    img_resized = resize(img_array, (64, 64, 3))
                    # Append the input image array to X
                    X.append(img_resized.flatten())
                    # Append the index of the category in categories list to Y
                    Y.append(categories.index(name))
        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)

    text.insert(END, "Dataset Preprocessing completed\n")

def Train_Test_split():
    global X,Y,x_train,x_test,y_train,y_test
    
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    text.insert(END,"Total samples found in training dataset: "+str(x_train.shape)+"\n")
    text.insert(END,"Total samples found in testing dataset: "+str(x_test.shape)+"\n")


def calculateMetrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="Pastel1" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def Existing_ETC():
    global x_train,x_test,y_train,y_test,model_folder
    text.delete('1.0', END)
    
    model_filename = os.path.join(model_folder, "ETC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = ExtraTreesClassifier(n_estimators=10,max_depth=3)
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Model saved to {model_filename}')

    y_pred = mlmodel.predict(x_test)
    calculateMetrics("Existing Extra Trees Classifier", y_pred, y_test)


import os
import joblib
from sklearn.tree import DecisionTreeClassifier

def Existing_DTC():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "DTC_model.pkl")

    if os.path.exists(model_filename):
        dtc_model = joblib.load(model_filename)
    else:
        dtc_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dtc_model.fit(x_train, y_train)
        joblib.dump(dtc_model, model_filename)
        print(f'Model saved to {model_filename}')

    y_pred = dtc_model.predict(x_test)
    calculateMetrics("Existing Decision Tree Classifier", y_pred, y_test)




def cnnModel():
    global X,Y, x_train,x_test,y_train,y_test,model_folder,categories,model,history
    text.delete('1.0', END)
    
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
    
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    x_train = x_train.reshape((-1, 64, 64, 3))  # Assuming 64x64 RGB images
    x_test  = x_test.reshape((-1, 64, 64, 3))
    y_train = to_categorical(y_train, num_classes=len(categories))  
    y_test  = to_categorical(y_test, num_classes=len(categories))  
    
    Model_file = os.path.join(model_folder,    "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights(Model_weights)
        model._make_predict_function()   
        print(model.summary())
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            acc = history['accuracy']
    else:
        model = Sequential() #resnet transfer learning code here
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(units = 256, activation = 'relu'))
        model.add(Dense(units = num_classes, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(model.summary())
        hist = model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test), shuffle=True, verbose=2)
        model.save_weights(Model_weights)            
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)
        with open(Model_history, 'rb') as f:
            accuracy = pickle.load(f)
            acc = accuracy['accuracy']
            acc = acc[4] * 100
            print("CNN Model Prediction Accuracy = "+str(acc))

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1) 
    calculateMetrics("Proposed CNN", Y_pred_classes, y_test)


def predict():
    global model
    categories=['parasitized','uninfected']
    
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(-1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    
    X_test_features = model.predict(test)
    predict = np.argmax(X_test_features)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Classified as : '+categories[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Classified as : '+categories[predict], img)
    cv2.waitKey(0)
    
def graph():
    global history

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def close():
    main.destroy()
    
    
font = ('times', 18, 'bold')
title = Label(main, text='Ai-Enhanced Microscopic Image Classification for RBC Infections')
title.config(bg='pale green', fg='tomato')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

mlpButton = Button(main, text="Dataset Splitting", command=Train_Test_split)
mlpButton.place(x=20,y=200)
mlpButton.config(font=ff)

mlpButton = Button(main, text="Train Extra Trees Classifier", command=Existing_ETC)
mlpButton.place(x=20,y=250)
mlpButton.config(font=ff)

mlpButton = Button(main, text="Train Decision Classifier", command=Existing_DTC)
mlpButton.place(x=20,y=300)
mlpButton.config(font=ff)


modelButton = Button(main, text="Train CNN Model", command=cnnModel)
modelButton.place(x=20,y=350)
modelButton.config(font=ff)


predictButton = Button(main, text="Prediction from Test Image", command=predict)
predictButton.place(x=20,y=400)
predictButton.config(font=ff)
'''
graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=450)
graphButton.config(font=ff)'''

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=500)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg = 'azure')
main.mainloop()
