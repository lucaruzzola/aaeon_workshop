import os
import logging
from openvino.inference_engine import IENetwork, IEPlugin
import copy
import uuid
import cv2
import numpy as np

logger = logging.getLogger(__name__)

#TODO implement this function to return square values
def make_square(x, y, w, h, max_w, max_h):
    return x, y, x+w, y+h

class Location:
    """
    Representation of a location of a frame

    Attributes
    ----------
    x: double
        Top left corner X.
    y: double
        Top left corner Y.
    w: double
        Location width.
    h: double
        Location height.
    """

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return 'Location(x={}, y={}, w={}, h={})'.format(self.x, self.y, self.w, self.h)

    def __repr__(self):
        return self.__str__()

    def get_centroid(self):
        """
        Calculates centroid
        :return: Coordinate tuple
        """
        return Point((self.x + self.w)/2, (self.y + self.h)/2)

class SyRequest:
    def __init__(self, request, frame):
        self.request = request
        self.frame = frame
        
class SyFrame:

    def __init__(self, frame, id_=None):
        self.frame = frame
        self.id_ = uuid.uuid4() if id_ is None else id_
        self.width = frame.shape[1]
        self.height = frame.shape[0]


class SyRegion:

    def __init__(self, sy_frame, location=None, label=None, confidence=None, region_id=None, metadata=dict):
        self.location = location
        self.region_id = uuid.uuid4() if region_id is None else region_id
        self.label = label
        self.confidence = confidence
        self.sy_frame = sy_frame
        self.metadata = metadata

    def __repr__(self):
        return f"SyRegion(label={self.label}, location={self.location}, confidence={self.confidence}, " \
            f"region_id={self.region_id})"

    def get_frame_region(self):
        """
        Crop area near region in frame
        :return: numpy.ndarray containing the cropped frame
        """

        return self.sy_frame.frame[self.location.y:self.location.y+self.location.h, self.location.x:self.location.x+self.location.w]

    def get_square_frame_region(self):
        """
        Crop square area near region in frame
        :return: numpy.ndarray containing the cropped frame
        """
        square_location = self.get_square_location()

        return SyFrame(frame=self.sy_frame.frame[square_location.y:square_location.y + square_location.h, square_location.x:square_location.x+square_location.w], id_=self.sy_frame.id_)

    def get_square_location(self):
        """
        Crop square area near region in frame
        :return: numpy.ndarray containing the cropped frame
        """
        square_x_min, square_y_min, square_x_max, square_y_max = \
            make_square(
                self.location.x,
                self.location.y,
                self.location.w,
                self.location.h,
                max_w=self.sy_frame.width,
                max_h=self.sy_frame.height
            )

        return Location(x=square_x_min, y=square_y_min, w=square_x_max-square_x_min, h=square_y_max-square_y_min)

class ObjectDetector:
    """
    Detect objects on frames using a neural network model.
    Model must be in Intel OpenVINO IE Format

    Notes
    -----
    Model Optimizer: https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer

    Attributes
    ----------
    threshold: double
        Confidence threshold that a detections has to have to be considered valid.
    n, c, net_input_height, net_input_width: int
        Optimized model's shape.
    plugin: openvino.inference_engine.ie_api.IEPlugin
        OpenVINO inference engine plugin.
    input_layer: str
        Network's input layer's name.
    output_layer: str
        Network's output layer's name.
    exec_net: inference_engine.ie_api.ExecutableNetwork
        OpenVINO network model.
    labels: :obj:`list` of :obj:`str`:
        List of classes to which might belong detected objects
    """

    def __init__(self, model_xml, model_bin, device, cpu_extension=None, pre=None, post=None, plugin_dir=None, num_requests=1, labels=None):
        """
        Parameters
        ----------
        model_xml: str
            Xml file path (obtained with the Model Converter).
        model_bin: str
            bin file path
        device:
            Select the device you want to use to run the model (CPU, GPU, MYRIAD).
        cpu_extension: str
            CPU extension file path. Used only if device==CPU.
        plugin_dir: str
            Plugin directory path (never used, always None).
        num_requests: int
            Number of OpenVINO requests alive at the same time.
        pre: func
            Frame pre-processing (before detection) function.
        post: func
            Frame post-processing (after detection) function.
        """

        self.n = -1
        self.c = -1
        self.net_input_width = -1
        self.net_input_height = -1
        self.plugin = self._load_plugin(plugin_dir, device, cpu_extension)
        self.input_layer, self.output_layer, self.exec_net = self._read_ir(model_xml=model_xml,
                                                                           model_bin=model_bin,
                                                                           num_requests=num_requests)
        if labels:
            self.labels = labels
        self._pre = pre
        self._post = post
        self.request_id = None

    @staticmethod
    def _load_plugin(plugin_dir, device, cpu_extension=None):
        """
        Loads OpenVINO plugin

        Returns
        -------
        openvino.inference_engine.ie_api.IEPlugin:
            OpenVINO inference engine plugin.
        """

        logger.debug("Initializing plugin for {} device from {}...".format(device, plugin_dir))
        plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        if cpu_extension and 'CPU' in device:
            logger.debug("Loading extension: {}".format(cpu_extension))
            plugin.add_cpu_extension(cpu_extension)
        return plugin

    def _read_ir(self, model_xml, model_bin, num_requests):
        """
        Loads OpenVINO optimized model

        Parameters
        ----------
        num_requests: int
            Number of OpenVINO requests alive at the same time.

        Returns
        -------
        str:
            Network's input layer's name.
        str:
            Network's output layer's name.
        inference_engine.ie_api.ExecutableNetwork:
            OpenVINO network model.
        """


        logger.info("Reading and loading IR {}".format(model_xml))
        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1, "Object Detector supports only single input topologies"
        assert len(net.outputs) == 1, "Object Detector supports only single output topologies"
        input_layer = next(iter(net.inputs))
        output_layer = next(iter(net.outputs))
        exec_net = self.plugin.load(network=net, num_requests=num_requests)
        self.n, self.c, self.net_input_height, self.net_input_width = net.inputs[input_layer].shape
        del net
        return input_layer, output_layer, exec_net

    def preprocess(self, sy_frame, *args):
        """
        Here you can chain lots of transformation of your frame.
        You can use the default behaviour of the preprocess function, that consists in applying an identity function
        to the frame_wrapper, or you can create your behaviour extending this class.
        """
        return sy_frame

    def postprocess(self, detection_result, sy_frame, *args):
        """
        It applies some transformation on network's outputs (if you want).
        If you don't override this function, we're gonna apply the identity function.

        Parameters
        ----------
        detection_result: whatever the output of the output layer is

        Returns
        -------
        network_output: whataver the output of the output layer is
        """

        return (detection_result, sy_frame)

    def start_detection(self, sy_frame, req_id = 0):

        preprocessed_frame = self.preprocess(sy_frame)

        return SyRequest(request = self.exec_net.start_async(request_id=req_id, inputs={self.input_layer: preprocessed_frame.frame}), frame=sy_frame)

    def get_detection(self, sy_request, wait_time = -1):
        request = sy_request.request
        sy_frame = sy_request.frame

        network_output = request.outputs[self.output_layer] if request.wait(wait_time) == 0 else None

        return self.postprocess(network_output, sy_frame)

    def detect(self, sy_frame, req_id=0, wait_time=-1):
        sy_request = self.start_detection(sy_frame, req_id)
        return self.get_detection(sy_request, wait_time)

    def destroy(self):
        """
        Remove network related object from the scope.

        Returns
        -------
            None.
        """
        del self.exec_net
        del self.plugin

class ImageClassifier:

    def __init__(self, model_xml, model_bin, device, cpu_extension, plugin_dir = None, num_requests=1):
        """
        Parameters
        ----------
        model_xml: str
            Xml file path (obtained with the Model Converter).
        model_bin: str
            bin file path
        threshold: double
            Predictions with confidence under this value will be ignored.
        device:
            Select the device you want to use to run the model (CPU, GPU, MYRIAD).
        cpu_extension: str
            CPU extension file path. Used only if device==CPU.
        plugin_dir: str
            Plugin directory path (never used, always None).
        num_requests: int
            Number of OpenVINO requests alive at the same time.
        """

        self.n = -1
        self.c = -1
        self.net_input_width = -1
        self.net_input_height = -1
        self.plugin = self._load_plugin(plugin_dir, device, cpu_extension)
        self.input_layer, self.output_layer, self.exec_net = self._read_ir(model_xml = model_xml, model_bin = model_bin, num_requests = num_requests)
    @staticmethod
    def _load_plugin(plugin_dir, device, cpu_extension):
        """
        Loads OpenVINO plugin

        Returns
        -------
        openvino.inference_engine.ie_api.IEPlugin:
            OpenVINO inference engine plugin.
        """

        logger.debug("Initializing plugin for {} device...".format(device))
        plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)
        return plugin

    def _read_ir(self, model_xml, model_bin, num_requests):
        """
        Loads OpenVINO optimized model

        Parameters
        ----------
        num_requests: int
            Number of OpenVINO requests alive at the same time.

        Returns
        -------
        str:
            Network's input layer's name.
        str:
            Network's output layer's name.
        inference_engine.ie_api.ExecutableNetwork:
            OpenVINO network model.
        """

        logger.info("Reading and loading IR {}".format(model_xml))

        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1, "Image Classifier supports only single input topologies"

        input_layer = next(iter(net.inputs))
        output_layer = net.outputs
        exec_net = self.plugin.load(network=net, num_requests=num_requests)
        del net
        return input_layer, output_layer, exec_net

    def preprocess(self, frame, *args):
        return frame

    def postprocess(self, result, *args):
        return result

    def start_prediction(self, frame_wrap, req_id = 0):

        frame = self.preprocess(frame_wrap)
        self.current_frame = frame
        
        request = self.exec_net.start_async(request_id=req_id, inputs={self.input_layer: frame})

        return request

    def get_prediction(self, request, wait_time = -1):
        network_output = None
        if request.wait(wait_time) == 0:
            network_output = request


        output = dict()
        for out in self.output_layer:
            output[out] = network_output.outputs[out]

        return self.postprocess(output)

    def predict(self, frame, wait_time = -1):
        req = self.start_prediction(frame, 0)
        return self.get_prediction(req, wait_time)

    def destroy(self):
        """
        Remove network related object from the scope.

        Returns
        -------
            None.
        """
        del self.exec_net
        del self.plugin

def draw_bounding_box(detection_output, frame, color=(50, 50, 50), thickness=2):
    if detection_output is not None:
        min_x = detection_output.location.x
        min_y = detection_output.location.y

        cv2.rectangle(
            frame,
            (min_x, min_y),
            (min_x + detection_output.location.w, min_y + detection_output.location.h),
            color,
            thickness
        )
        
def load_emojis(emojis_path):
    anger = cv2.imread(emojis_path + "anger.png", cv2.IMREAD_UNCHANGED)
    surprise = cv2.imread(emojis_path +"surprise.png", cv2.IMREAD_UNCHANGED)
    happy = cv2.imread(emojis_path + "happy.png", cv2.IMREAD_UNCHANGED)
    neutral = cv2.imread(emojis_path + "neutral.png", cv2.IMREAD_UNCHANGED)
    sadness = cv2.imread(emojis_path + "sadness.png", cv2.IMREAD_UNCHANGED)
    emojis = {"anger": anger, "happy": happy, "surprise": surprise, "neutral": neutral, "sad": sadness}
    return emojis


def emoji_overlay(emoji, frame, location):
    
    max_dim = location.w if location.w > location.h else location.h
    min_dim = location.w if location.w < location.h else location.h
    semidiff = (max_dim - min_dim)/2
    
    if max_dim == location.h:
        y_offset = int(location.y)
        x_offset = int(location.x-semidiff)
    else:
        y_offset = int(location.y+semidiff)
        x_offset = int(location.x)
        
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
        
    emoji = cv2.resize(emoji, (max_dim, max_dim))
    
    y1, y2 = y_offset, y_offset + emoji.shape[0]
    x1, x2 = x_offset, x_offset + emoji.shape[1]


    
    if x1 >= frame_w or x2 >= frame_w or y1 >= frame_h or y2 >= frame_h or x1<0 or x2<0 or y1<0 or y2< 0:
        return frame

    else:

        alpha_s = emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            asd = alpha_s * emoji[:, :, c]
            qwe = alpha_l * frame[y1:y2, x1:x2, c]
            frame[y1:y2, x1:x2, c] = (asd + qwe)

        return frame