# import libraries ________________
import cv2
import face_recognition
import os
import glob
import time
# Define people's photos in databases ________________
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
while True:
    ret, frame = vc.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(frame_rgb)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = f'FPS:{str(fps)}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (10, 60), font, 2, (100, 200, 200), 5, cv2.LINE_AA)
    if len(faces)==1:
        for face in faces:
            top, right, bottom, left = face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            face_code = face_recognition.face_encodings(frame_rgb, [face])[0]
            results = face_recognition.compare_faces(known_faces, face_code, tolerance=0.6)
            if any(results):
                name = known_names[results.index(True)]
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                name = 'unknown'
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif len(faces)>0:
        cv2.rectangle(img=frame, pt1=(15, 70), color=(10, 200, 0), pt2=(543, 120), thickness=5)
        cv2.putText(frame, 'Please one person', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
    else:
        cv2.rectangle(img=frame, pt1=(85, 175), color=(10, 200, 0), pt2=(550, 300), thickness=3)
        cv2.putText(frame, 'there is no one', (124, 245), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    cv2.imshow('face recognition', frame)
    k = cv2.waitKey(1)
    # The program closes if you press the letter q on the keyboard
    if ord('q') == k:
        break
cv2.destroyAllWindows()
vc.release()
