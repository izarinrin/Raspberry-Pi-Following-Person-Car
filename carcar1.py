import cv2
import RPi.GPIO as GPIO
from time import sleep

# Steering Motor Pins
in1 = 17
in2 = 27
steering_enable = 22

# Throttle Motors Pins
in3 = 23
in4 = 24
throttle_enable = 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(steering_enable, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(throttle_enable, GPIO.OUT)

p_steering = GPIO.PWM(steering_enable, 1000)
p_throttle = GPIO.PWM(throttle_enable, 1000)

p_steering.start(75)
p_throttle.start(75)

GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

name = "person"

classNames = []
classFile = "/home/pi/tutorial/objecttracking/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/tutorial/objecttracking/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/tutorial/objecttracking/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(160, 160)  # Reduced input size for faster processing
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        img,
                        classNames[classId - 1].upper(),
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        img,
                        str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 160)  # Reduced frame width for faster processing
    cap.set(4, 120)  # Reduced frame height for faster processing

    GPIO.setwarnings(False)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2, draw=False, objects=[name])

        if objectInfo:
            # If a person is detected, move the car forward
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
            GPIO.output(in4, GPIO.HIGH)
            GPIO.output(in3, GPIO.LOW)

            # Adjust steering based on the center of the bounding box
            x, _, w, _ = objectInfo[0][0]
            center_x = x + w // 2
            if center_x < 80:
                # Turn left
                GPIO.output(in1, GPIO.LOW)
                GPIO.output(in2, GPIO.HIGH)
            elif center_x > 80:
                # Turn right
                GPIO.output(in1, GPIO.HIGH)
                GPIO.output(in2, GPIO.LOW)

            print("Following person")
        else:
            # Stop the car if no person is detected
            GPIO.output([in1, in2, in4, in3], GPIO.LOW)
            print('Stop')

        # Display the video stream
        #cv2.imshow("Output", img)
        #cv2.waitKey(1)
