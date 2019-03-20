import cv2
import os
import openvino
from models import FaceDetector
from utils import SyFrame
from utils import draw_bounding_box
from utils import load_emojis 
from utils import emoji_overlay
from models import EmotionClassifier

emotion_classifier = EmotionClassifier(model_xml="./assets/emotion_recognition/FP32/em.xml",\
                                       model_bin="./assets/emotion_recognition/FP32/em.bin",\
                                       device="CPU", cpu_extension="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so",\
                                       emotion_label_list=["neutral", "happy", "sad", "surprise", "anger"])

emojis = load_emojis("./assets/emojis/")

face_xml="./assets/face_detection/FP32/fd.xml"
face_bin="./assets/face_detection/FP32/fd.bin"

face_detector = FaceDetector(model_xml=face_xml,\
                             model_bin=face_bin,\
                             device="CPU",\
                             confidence_threshold=0.5,\
                             cpu_extension="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()
    detected_faces=face_detector.detect(SyFrame(img))

    for face in detected_faces:
        emotion = emotion_classifier.predict(face.get_square_frame_region().frame)
        emoji_overlay(emojis[emotion], img, face.location)        
        draw_bounding_box(\
        face,\
        img,\
        )


    cv2.imshow("Demo",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
