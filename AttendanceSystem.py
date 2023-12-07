import pickle
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_vggface import utils
from keras.models import load_model
#defining function to mark attendence
# def markAttendence(name):
#     with open('AttSys\Attendance.csv','r+') as record:
#         myDataList= record.readlines()
#         namelist=[]
#         for line in myDataList:
#             entry = line.split(',')
#             namelist.append(entry[0])
#             if name not in namelist:
#                 now = datetime.now()
#                 timestamp=now.strftime('%D %H:%M:%S')
#                 record.writelines(f'\n{name},{timestamp}')
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
           
# dimension of images
image_width = 224
image_height = 224

# load the training labels
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, "rb") as f: 
    class_dictionary = pickle.load(f)

    class_list = [value for _, value in class_dictionary.items()]
print(class_list)

#attendance list
AttList = np.zeros(len(class_list), dtype=int)

# for detecting faces
facecascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#for i in range(1,30): 
test_image_filename = 'AttSys\SampleHeadshots\MohitPrikshit.jpg'

# load the image
imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
image_array = np.array(imgtest, "uint8")

# get the faces detected in the image
faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)
#print(faces)
#print(i)
#print(len(faces))
# if not exactly 1 face is detected, skip this photo
# if len(faces) != 1: 
#     print('---We need exactly 1 face; ')
#     print('photo skipped---')
#continue
#for face in faces:
for i in range(0, len(faces)):
    (x_, y_, w, h) = faces[i]
    print(faces[i])
    # draw the face detected
    face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
    plt.imshow(face_detect)
    plt.show()

    # resize the detected face to 224x224
    size = (image_width, image_height)
    roi = image_array[y_: y_ + h, x_: x_ + w]
    resized_image = cv2.resize(roi, size)

    # prepare the image for prediction
    # x = np.array(resized_image, "uint8")
    # #x = image.img_to_array(resized_image)
    # x = np.expand_dims(x, axis=0)
    # x = utils.preprocess_input(x, version=1)

    
    face_array = np.array(resized_image, "uint8")
    img = face_array.reshape(1,image_width,image_height,3) 
    img = img.astype('float32')
    img /= 255

    # making prediction
    model = load_model('transfer_learning_trained_face_cnn_model.h5')
    predicted_prob = model.predict(img)
    #print(predicted_prob)
    #print(predicted_prob[0].argmax())
    if max(predicted_prob[0])>0.5:
        print("Predicted face: " + class_list[predicted_prob[0].argmax()])
        print("============================\n")
        AttList[predicted_prob[0].argmax()]=1
markAttendence(AttList)


'''possible additions:
    1. GUI
    2. recog from video feed
    3. record of all unverified persons clicked
    4. GUI for different classes
    5. manual entry of missed people if unverified by comparing with remaining students
    '''