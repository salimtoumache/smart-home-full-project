from  tkinter import *
from  tkinter import  ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import glob
import serial
import time
import serial.tools.list_ports
import warnings
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
def win2():
    def train():
        data_epoch_get = data_epoch.get()
        if data_epoch_get == "epochs":
            messagebox.showinfo(title='info', message=f'\n  | select epochs | ')
        else:
            messagebox.showinfo(title='info', message=f'\n  | selected successfully | ')
            data_path = 'Dataset'
            categories = os.listdir(data_path)
            labels = [i for i in range(len(categories))]
            label_dict = dict(zip(categories, labels))
            img_size = 32
            data = []
            target = []
            facedata = "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(facedata)
            for category in categories:
                folder_path = os.path.join(data_path, category)
                img_names = os.listdir(folder_path)
                for img_name in img_names:
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path)
                    faces = cascade.detectMultiScale(img)
                    try:
                        for f in faces:
                            x, y, w, h = [v for v in f]
                            sub_face = img[y:y + h, x:x + w]
                            gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (img_size, img_size))
                            data.append(resized)
                            target.append(label_dict[category])
                    except Exception as e:
                        print('Exception:', e)
            warnings.filterwarnings('ignore')
            data = np.array(data) / 255.0
            data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
            target = np.array(target)
            new_target = np_utils.to_categorical(target)
            np.save('training/data', data)
            np.save('training/target', new_target)
            data = np.load('training/data.npy')
            target = np.load('training/target.npy')
            noOfFilters = 64
            sizeOfFilter1 = (3, 3)
            sizeOfFilter2 = (3, 3)
            sizeOfPool = (2, 2)
            noOfNode = 64
            model = Sequential()
            model.add((Conv2D(32, sizeOfFilter1, input_shape=data.shape[1:], activation='relu')))
            model.add((Conv2D(32, sizeOfFilter1, activation='relu')))
            model.add(MaxPooling2D(pool_size=sizeOfPool))
            model.add((Conv2D(64, sizeOfFilter2, activation='relu')))
            model.add((Conv2D(64, sizeOfFilter2, activation='relu')))
            model.add(MaxPooling2D(pool_size=sizeOfPool))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(noOfNode, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)
            checkpoint = ModelCheckpoint('training/model-{epoch:03d}.model', monitor='val_loss', verbose=0,
                                         save_best_only=True,
                                         mode='auto')
            history = model.fit(train_data, train_target, epochs=int(data_epoch_get), callbacks=[checkpoint],
                                validation_split=0.2)
            label = Label(text='| Model Saved |', bg="SpringGreen3", font=('times', 35, 'bold')).place(x=560, y=350)
    def dataknow():
        data_image_get = data_image.get()
        if data_image_get =='images':
            messagebox.showinfo(title='info', message=f'\n  | select number of photos | ')
        else:
            messagebox.showinfo(title='info', message=f'\n  | selected successfully | ')
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                faces = detector.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum = sampleNum + 1
                    cv2.rectangle(img=img, pt1=(10, 50), color=(10, 200, 0), pt2=(175, 130), thickness=5)
                    cv2.putText(img, str(sampleNum), (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
                    cv2.imwrite("Dataset/know/ " + str(sampleNum) + ".jpg",
                                img[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif sampleNum > int(data_image_get):
                    break
            cam.release()
            cv2.destroyAllWindows()
            label = Label(text='| Images Saved |', bg="SpringGreen3", font=('times', 30, 'bold')).place(x=50, y=350)
    app_width = 932
    app_height = 548
    window2 = Tk()
    width = window2.winfo_screenwidth()
    height = window2.winfo_screenheight()
    x = (width / 2) - (app_width / 2)
    y = (height / 2) - (app_height / 2)
    window2.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
    window2.title("train model")
    window2.resizable(False, False)
    filename = PhotoImage(file="app/2.png")
    background_label = Label(window2, image=filename)
    dataknow_image = PhotoImage(file="app/dataknow.png")
    train_image = PhotoImage(file="app/train.png")
    epoch = ["epochs",
             "50",
             "100",
             "200",
             "300",
             "400",
             "500",
             "600",
             "700",
             "800",
             "900",
             "1000", ]

    data_epoch = ttk.Combobox(values=epoch, height=50, state="readonly", width=10, font=('Consolas', 12))
    data_epoch.set(epoch[0])
    data_epoch.place(x=800, y=250)
    image = ["images",
             "100",
             "200",
             "300",
             "400",
             "500",
             "1000",
             "2000",
             "3000",
             "5000",
             "7000",
             "10000", ]
    data_image = ttk.Combobox(values=image,height=50, state="readonly", width=10, font=('Consolas', 12))
    data_image.set(image[0])
    data_image.place(x=330, y=250)
    dataknowmodel = Button(image=dataknow_image, command=dataknow).place(x=90, y=215)
    train_model = Button(image=train_image, command=train).place(x=560, y=220)
    b_back = Button(text='back', font=('times', 30, 'bold'),command=lambda: [window2.destroy(), win1()])
    b_back.place(x=400, y=450)
    background_label.pack()
    window2.mainloop()
def win3():
    def open_test():
        path_model = filedialog.askdirectory()
        model = load_model(path_model)
        face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        labels_dict = {0: 'know', 1: 'unknow'}
        color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
        ports = serial.tools.list_ports.comports()
        for i in ports:
            port = (i[0])
        try:
            ser = serial.Serial(port, 115200)
        except:
            pass
        finally:
            time.sleep(0.5)
        while (True):
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 3)
            for (x, y, w, h) in faces:
                face_img = gray[y:y + w, x:x + w]
                resized = cv2.resize(face_img, (32, 32))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 32, 32, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                if(len(faces))>1:
                    cv2.rectangle(img=img, pt1=(15, 70), color=(10, 200, 0), pt2=(543, 120), thickness=5)
                    cv2.putText(img, 'Please one person', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
                    try:
                        ser.write("OFF\r".encode())
                    except:
                        pass
                elif (len(faces))<1:
                    cv2.rectangle(img=img, pt1=(15, 70), color=(10, 200, 0), pt2=(543, 120), thickness=5)
                    cv2.putText(img, 'there is no one', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
                    try:
                        ser.write("OFF\r".encode())
                    except:
                        pass
                else:
                    if label == 0:
                        print(labels_dict[label])
                        try:
                            ser.write("ON\r".encode())
                        except:
                            pass
                    else:
                        try:
                            ser.write("OFF\r".encode())
                        except:
                            pass
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                    cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                    cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('Result', img)
            k = cv2.waitKey(1)
            if k == ord("q"):
                try:
                    ser.write("OFF\r".encode())
                except:
                    pass
                break
        cv2.destroyAllWindows()
        cap.release()
    app_width = 932
    app_height = 548
    window3 = Tk()
    width = window3.winfo_screenwidth()
    height = window3.winfo_screenheight()
    x = (width / 2) - (app_width / 2)
    y = (height / 2) - (app_height / 2)
    window3.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
    window3.title("test model")
    window3.resizable(False, False)
    filename = PhotoImage(file="app/4.png")
    background_label = Label(window3, image=filename)
    opencam_image = PhotoImage(file="app/opencam.png")
    button_opencam = Button(image=opencam_image, command=open_test).place(x=360, y=335)
    b_back = Button(text='back', font=('times', 30, 'bold'),command=lambda: [window3.destroy(), win1()])
    b_back.place(x=400, y=450)
    ports = serial.tools.list_ports.comports()
    for i in ports:
        port = (i[0])
        label = Label(text=port, font=('times', 15, 'bold')).place(x=700, y=200)
    background_label.pack()
    window3.mainloop()
def win1():
    app_width = 932
    app_height = 551
    win1 = Tk()
    width = win1.winfo_screenwidth()
    height = win1.winfo_screenheight()
    x = (width / 2) - (app_width / 2)
    y = (height / 2) - (app_height / 2)
    win1.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
    win1.title("deep learning")
    win1.resizable(False, False)
    filename = PhotoImage(file="app/1.png")
    background_label = Label(win1, image=filename)
    train_model_image = PhotoImage(file="app/train_model.png")
    test_model_image = PhotoImage(file="app/test_model.png")
    button_train_model = Button(image=train_model_image, command=lambda: [win1.destroy(), win2()]).place(x=140, y=230)
    button_test_model = Button(image=test_model_image, command=lambda: [win1.destroy(), win3()]).place(x=550, y=230)
    background_label.pack()
    win1.mainloop()
win1()
