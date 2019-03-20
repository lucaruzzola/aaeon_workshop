import cv2
import os
import openvino
from models import FaceDetector
from utils import SyFrame
from utils import draw_bounding_box

face_xml="./assets/face_detection/FP32/fd.xml"
face_bin="./assets/face_detection/FP32/fd.bin"

face_detector = FaceDetector(model_xml=face_xml,\
                             model_bin=face_bin,\
                             device="CPU",\
                             confidence_threshold=0.4,\
                             cpu_extension="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()
    detected_faces=face_detector.detect(SyFrame(img))

    for face in detected_faces:
        sq_loc = face.get_square_location()
        blur_face = face.get_square_frame_region().frame
        blur_face = cv2.blur(blur_face,(45,45))

        img[sq_loc.y:sq_loc.y+blur_face.shape[0], sq_loc.x:sq_loc.x+blur_face.shape[1]] = blur_face

    cv2.imshow("Demo",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()