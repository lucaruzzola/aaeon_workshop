from utils import *

#TODO implement pre and post processing functions for this network
class FaceDetector(ObjectDetector):

    def __init__(self, model_xml, model_bin, device, confidence_threshold, cpu_extension=None):
        super().__init__(model_xml, model_bin, device, cpu_extension=cpu_extension)
        self.confidence_threshold = confidence_threshold

    def preprocess(self, sy_frame, *args):
        return sy_frame

    def postprocess(self, detection_result, sy_frame, *args):
        results = []
        return [SyRegion(label="default_class", confidence=1, location=Location(x=0,y=0,w=0,h=0), sy_frame+sy_frame)]
    
    
#TODO implement pre and post processing functions for this network
class EmotionClassifier(ImageClassifier):

    def __init__(self, model_xml, model_bin, device, cpu_extension, emotion_label_list=None, num_requests = 1):
        super().__init__(model_xml, model_bin, device, cpu_extension,num_requests=num_requests)
        if emotion_label_list is not None:
            self.emotion_map = emotion_label_list
        else:
            self.emotion_map = []

    def preprocess(self, frame, *args):
        return frame

    def postprocess(self, result, *args):
        return "happy" 
    
