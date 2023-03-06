from  tkinter import *
from  tkinter import  ttk
from tkinter import messagebox
import os
import glob
import time
import cv2
import face_recognition
import serial
import os
import glob
import serial
import time
ser = serial.Serial("/dev/ttyUSB1", 9600)


def win2():
    def train():
        data_epoch_get = data_epoch.get()
        if data_epoch_get == "epochs":
            messagebox.showinfo(title='info', message=f'\n  | select epochs | ')
        else:
            prog = ttk.Progressbar(orient=HORIZONTAL, length=932)
            prog.place(x=0, y=530)
            messagebox.showinfo(title='info', message=f'\n  | selected successfully | ')
            n=0
            while n<int(data_epoch_get):
                n = n + 1
                prog.config(value=int(n), maximum=int(data_epoch_get))
            label = Label(text='| Model Saved |', bg="SpringGreen3", font=('times', 35, 'bold')).place(x=560, y=350)
    def dataknow():
        data_image_get = data_image.get()
        if data_image_get == 'images':
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

                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/salim/ " + str(sampleNum) + ".jpg",
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
    data_image = ttk.Combobox(values=image, height=50, state="readonly", width=10, font=('Consolas', 12))
    data_image.set(image[0])
    data_image.place(x=330, y=250)
    dataknowmodel = Button(image=dataknow_image, command=dataknow).place(x=90, y=215)
    train_model = Button(image=train_image, command=train).place(x=560, y=220)
    b_back = Button(text='back', font=('times', 30, 'bold'), command=lambda: [window2.destroy(), win1()])
    b_back.place(x=400, y=450)

    background_label.pack()
    window2.mainloop()
def win3():
    def open_test():
        known_faces = []
        known_names = []
        known_faces_paths = []
        registered_faces_path = 'dataset/'
        for name in os.listdir(registered_faces_path):
            images_mask = '%s%s/*.jpg' % (registered_faces_path, name)
            images_paths = glob.glob(images_mask)
            known_faces_paths += images_paths
            known_names += [name for x in images_paths]

        def get_encodings(img_path):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            return encoding[0]

        known_faces = [get_encodings(img_path) for img_path in known_faces_paths]
        # Camera selection ________________
        vc = cv2.VideoCapture(0)
        prev_frame_time = 0
        new_frame_time = 0
        # data transfer protocol

        # real time tracking face ________________
        # data transfer protocol

        # real time tracking face ________________
        print("while")
        while True:
            ret, frame = vc.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame_rgb)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if len(faces) == 1:
                for face in faces:
                    top, right, bottom, left = face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    face_code = face_recognition.face_encodings(frame_rgb, [face])[0]
                    results = face_recognition.compare_faces(known_faces, face_code, tolerance=0.6)
                    if any(results):
                        datasend ="ON"
                        datasend = datasend + "\r"
                        ser.write(datasend.encode())
                        name = known_names[results.index(True)]
                        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    else:
                        datasend = "OFF"
                        datasend = datasend + "\r"
                        ser.write(datasend.encode())

                        name = 'unknown'
                        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            elif len(faces) > 0:

                datasend = "OFF"
                datasend = datasend + "\r"
                ser.write(datasend.encode())
                cv2.rectangle(img=frame, pt1=(15, 70), color=(10, 200, 0), pt2=(543, 120), thickness=5)
                cv2.putText(frame, 'Please one person', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
            else:

                datasend = "OFF"
                datasend = datasend + "\r"
                ser.write(datasend.encode())
                cv2.rectangle(img=frame, pt1=(85, 175), color=(10, 200, 0), pt2=(550, 300), thickness=3)
                cv2.putText(frame, 'there is no one', (124, 245), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            cv2.imshow('face recognition', frame)
            k = cv2.waitKey(1)
            # The program closes if you press the letter q on the keyboard
            if ord('q') == k:
                break
        cv2.destroyAllWindows()
        vc.release()

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
    b_back = Button(text='back', font=('times', 30, 'bold'), command=lambda: [window3.destroy(), win1()])
    b_back.place(x=400, y=450)
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
