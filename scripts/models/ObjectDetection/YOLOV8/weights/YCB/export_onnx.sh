pip install ultralytics
pip install onnx
pip install onnxsim
pip install onnx2tf onnx_graphsurgeon tflite_support sng4onnx

# Example export
yolo export model=yolov8n_320.pt format=onnx imgsz=320,240 opset=11 int8=True dynamic=True simplify=True

# Args:

# Key		Value				Description
# format	'torchscript'		format to export to
# imgsz		640					image size as scalar or (h, w) list, i.e. (640, 480)
# keras		False				use Keras for TF SavedModel export
# optimize	False				TorchScript: optimize for mobile
# half		False				FP16 quantization
# int8		False				INT8 quantization
# dynamic	False				ONNX/TF/TensorRT: dynamic axes
# simplify	False				ONNX: simplify model
# opset		None				ONNX: opset version (optional, defaults to latest)
# workspace	4					TensorRT: workspace size (GB)
# nms		False				CoreML: add NMS