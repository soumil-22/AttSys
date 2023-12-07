import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

headshots_folder_name = 'C:\\Users\\soumil\\Desktop\\Python Projects\\AttSys\\Headshots'
# dimension of images
image_width = 224
image_height = 224

# for detecting faces
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images_dir = os.path.join(".", headshots_folder_name)

for root, _, files in os.walk(headshots_folder_name):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            # path of the image
            path = os.path.join(root, file)
            # load the image
            #path = "AttSys\Headshots\Obama\Obama1.jpg"
            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            image_array = np.array(imgtest, "uint8")
            print(path)
            # get the faces detected in the image
            faces =  facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

            for (x_, y_, w, h) in faces:

                # draw the face detected
                face_detect = cv2.rectangle(imgtest,
                        (x_, y_),
                        (x_+w, y_+h),
                        (255, 0, 255), 2)
                #plt.imshow(face_detect)
                #plt.show()

                # resize the detected face to 224x224
                size = (image_width, image_height)

                # detected face region
                roi = image_array[y_: y_ + h, x_: x_ + w]

                # resize the detected head to target size
                resized_image = cv2.resize(roi, size)
                image_array = np.array(resized_image, "uint8")

                # remove the original image
                os.remove(path)

                # replace the image with only the face
                im = Image.fromarray(image_array)
                im.save(path)
