###
#export onnx model from pytorch or tensorflow train framework
###
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
 
convert(
    framework="pt",      #train framewrok
    model="./",          #model path
    output=Path("./test/test.onnx"),  #output model path
    opset=11,             
    pipeline_name="sentiment-analysis")  #pipline ["feature-extraction","ner","sentiment-analysis"...]

 ##more info in https://github.com/huggingface/transformers/blob/main/src/transformers/convert_graph_to_onnx.py

