import cv2
import numpy as np


import os


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=3):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

class_id = 0  # Labels for the given file
names = {}  # mapping id with name
dataset_path = 'data/'

face_data = []
labels = []

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping btw class_id and name
        names[class_id] = fx[:-4]

        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # create labels for the class
        target = class_id * np.ones((data_item.shape[0],))

        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

train_set = np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

# testing

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    # faces = sorted(faces,key=lambda f:f[2]*f[3])

    # pick the last face(because it has the largest area)
    for face in faces:
        # draw bounding box or the rectangle
        x, y, w, h = face

        # extract (crop out the required face) : region of interest
        offset = 10
        face_section = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # predict
        out = knn(train_set, face_section.flatten())

        # Display the output on the screen
        pred_name = names[int(out)]
        cv2.putText(gray_frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # face_data.append(face_section)
        # print(len(face_section))

        # cv2.imshow("Frame", frame)
    cv2.imshow("gray_frame", gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()