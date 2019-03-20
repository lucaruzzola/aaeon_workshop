from utils import *

class FaceDetector(ObjectDetector):

    def __init__(self, model_xml, model_bin, device, confidence_threshold, cpu_extension=None):
        super().__init__(model_xml, model_bin, device, cpu_extension=cpu_extension)
        self.confidence_threshold = confidence_threshold

    def preprocess(self, sy_frame, *args):
        frame = copy.deepcopy(sy_frame.frame)
        resized_frame = cv2.resize(frame, (self.net_input_width, self.net_input_height))
        transposed_frame = resized_frame.transpose((2,0,1))
        return SyFrame(transposed_frame, sy_frame.id_)

    def postprocess(self, detection_result, sy_frame, *args):
        results = []

        for obj in detection_result[0][0]:
            textual_label = str(int(obj[1]))  # edited
            confidence = obj[2]

            if int((obj[1])) != -1 and confidence > self.confidence_threshold:
                x_min = max(0, int(obj[3] * sy_frame.width))
                y_min = max(0, int(obj[4] * sy_frame.height))
                x_max = int(obj[5] * sy_frame.width)
                y_max = int(obj[6] * sy_frame.height)

                result = SyRegion(label=textual_label, confidence=confidence, location=Location(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min), sy_frame=sy_frame)

                results.append(result)

        return results
    
    

class EmotionClassifier(ImageClassifier):

    def __init__(self, model_xml, model_bin, device, cpu_extension, emotion_label_list=None, num_requests = 1):
        super().__init__(model_xml, model_bin, device, cpu_extension,num_requests=num_requests)
        if emotion_label_list is not None:
            self.emotion_map = emotion_label_list
        else:
            self.emotion_map = []

    def preprocess(self, frame, *args):
        emotion_img = cv2.resize(frame, (64, 64))
        emotion_img = np.transpose(emotion_img, (2, 0, 1))
        return emotion_img

    def postprocess(self, result, *args):
        key_list = list(result.keys())
        if len(key_list) == 1:
            rez = result[key_list[0]]
            emotion = np.reshape(rez, (5))
            return self.emotion_map[int(np.argmax(emotion))]
        else:
            logger.error("They key list does not match expectations.")    
    
