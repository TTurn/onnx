###
#optimize the onnx model
###
from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model("test/test.onnx", model_type='bert', num_heads=12, hidden_size=768)
optimized_model.save_model_to_file("test/bert_opt.onnx")
