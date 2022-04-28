from pathlib import Path
from transformers.convert_graph_to_onnx import quantize

quantize(Path("test/test.onnx"))
