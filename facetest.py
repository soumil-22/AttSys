from PIL import Image
import numpy as np
import cv2
import pickle
from keras.models import load_model
from datetime import datetime
# for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 224
image_height = 224

#defining function to mark attenance
# def markAttendence(name):
#     with open('AttSys\Attendance.csv','r+') as record:
#         myDataList= record.readlines()
#         namelist=[]
#         for line in myDataList:
#             entry = line.split(',')
#             namelist.append(entry[0])
#             print(namelist)
#             if name not in namelist:q
#                 now = datetime.now()
#                 timestamp=now.strftime('%D %H:%M:%S')
#                 print(timestamp)
#                 record.writelines(f'\n{name},{timestamp}')
#             else:
#                 break;

def markAttendence(list):
    with open('AttSys\Attendance.csv','r+') as record:
        record.readlines()
        record.writelines(f'\n')
        now= datetime.now()
        timestamp = now.strftime('%D %H:%M:%S')
        for i in range(len(list)):
            record.writelines(f'{list[i]}')
            if i== len(list)-1:
                record.writelines(f',{timestamp}')
            else:
                record.writelines(f',')
        

#function to compile attendance

# load the trained model
model = load_model('transfer_learning_trained_face_cnn_model.h5')

# the labels for the trained model
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
    print(labels)
AttList = np.zeros(len(labels), dtype=int)
# default webcam
stream = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    success, frame = stream.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=6)
    
    # for each faces found
    for (x, y, w, h) in faces: 
        roi_rgb = rgb[y:y+h, x:x+w]

        # Draw a rectangle around the face
        color = (255, 0, 0) # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        # resize the image
        size = (image_width, image_height)
        resized_image = cv2.resize(roi_rgb, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1,image_width,image_height,3) 
        img = img.astype('float32')
        img /= 255

        # predict the image
        predicted_prob = model.predict(img)
        print(predicted_prob)

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        if max(predicted_prob[0])>0.7:
            name = labels[predicted_prob[0].argmax()]
            #markAttendence(name)
            AttList[predicted_prob[0].argmax()]=1
        else:
            name ="unverified"
        
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x,y-8),
            font, 1, color, stroke, cv2.LINE_AA)
    # Show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out of the loop
        break      

    # Cleanup

markAttendence(AttList)
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

