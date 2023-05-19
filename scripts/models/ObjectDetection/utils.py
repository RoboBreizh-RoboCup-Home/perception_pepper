from perception_utils.utils import get_pkg_path
import os
import numpy as np
import json

class Model:
    def __init__(self, args):

        # default constructor
        self._model_name = "" # required
        self._base_path = get_pkg_path()
        self._type = ""
        self._model_path = ""
        self._classes = []
        self._graph_path = ""
        self._input_size = None
        self._object_mapping = {}
        self._engine = ""
        self._iou_threshold = 0.5
        self._conf_threshold = 0.3

        required_args = ["model_name", "model_path", "classes"]

        if "type" in args:
            if args["type"] == "tensorflow":
                required_args.append("graph_path")
        
        # Check if all required arguments are present in the input dictionary
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Required argument '{arg}' is missing")
        
        # Set only the keys present in the dictionary
        for key in args:
            setattr(self, key, args[key])

    def inference(self, image:np.array):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError
    
    def get_input_size(self):
        raise NotImplementedError
    
    def pre_process(self, image:np.array):
        raise NotImplementedError

    def post_process(self, outputs,  image:np.array):
        raise NotImplementedError
    
    # property, setters and getters
    @property
    def model_name(self):
        return self._model_name
    
    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def base_path(self):
        return self._base_path
    
    @base_path.setter
    def base_path(self, base_path):
        self._base_path = base_path

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, type):
        self._type = type
    
    @property
    def model_path(self):
        return self._model_path
    
    @model_path.setter
    def model_path(self, model_path):
        self._model_path = os.path.join(self.base_path, model_path)

    @property
    def classes(self):
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        classes_path = os.path.join(self.base_path, classes)
        with open(classes_path, 'r') as f:
            lines = f.read().splitlines()
        classes = [line.strip().lower() for line in lines]
        self._classes = classes

    @property
    def graph_path(self):
        return self._graph_path
    
    @graph_path.setter
    def graph_path(self, graph_path):
        self._graph_path = os.path.join(self.base_path, graph_path)

    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    @property
    def object_mapping(self):
        return self._object_mapping
    
    @object_mapping.setter
    def object_mapping(self, object_mapping):
        with open(object_mapping, 'r') as f:
            self._object_mapping  = json.load(f)

    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def iou_threshold(self):
        return self._iou_threshold
    
    @iou_threshold.setter
    def iou_threshold(self, _iou_threshold):
        self._iou_threshold = _iou_threshold

    @property
    def conf_threshold(self):
        return self._conf_threshold
    
    @conf_threshold.setter
    def conf_threshold(self, _conf_threshold):
        self._conf_threshold = _conf_threshold

    def __str__(self):
        # return list of arguments in string format
        return f"model_name: {self.model_name}, base_path: {self.base_path}, type: {self.type}, model_path: {self.model_path}, classes: {self.classes}, res: {self.res}, graph_path: {self.graph_path}, input_size: {self.input_size}, object_mapping: {self.object_mapping}, engine: {self.engine}, iou_threshold: {self.iou_threshold}, conf_threshold: {self.conf_threshold}"