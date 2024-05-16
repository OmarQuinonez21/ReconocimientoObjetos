import cv2 #opencv
import urllib.request #para abrir y leer URL
import numpy as np

# URL of the ESP32-CAM
url = 'http://192.168.1.155/cam-hi.jpg'
winName = 'ESP32 CAMERA'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

def determine_slot_number(box):
    # Logic to determine slot number based on box coordinates
    # Placeholder function, implement as needed
    # For example:
    x_center = box[0] + box[2] / 2
    if x_center < 160:
        return 0
    elif x_center < 320:
        return 1
    elif x_center < 480:
        return 2
    else:
        return 3

# Load class names from coco.names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize parking slots array
cajones = [0] * 4  # Assuming 4 parking slots

while True:
    imgResponse = urllib.request.urlopen(url)  # Open URL
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)  # Decode image

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate if needed

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    # Reset parking slots status
    cajones = [0] * 2

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classNames[classId - 1]
            if label == 'car' or label == 'truck' or label == 'bus':  # Update based on class
                slot_number = determine_slot_number(box)  # Implement this function based on your logic
                cajones[slot_number] = 1
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    print(f"Cajones: {cajones}")

    cv2.imshow(winName, img)  # Show image

    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:  # Exit on ESC
        break

cv2.destroyAllWindows()


