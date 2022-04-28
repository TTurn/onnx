import onnxruntime
import numpy as np
from torch import nn
import torch

sess = onnxruntime.InferenceSession("onnx/test/test.onnx", providers=["CUDAExecutionProvider"])

for input_meta in sess.get_inputs():
    print(input_meta.name)

for output_name in sess.get_outputs():
    print(output_name.name)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

sentence = ["111","111"]
feature = tokenizer.batch_encode_plus([sentence], max_length=64)
print(feature)

import time
#pred_onnx = sess.run(["output_0","output_1"], {"input_ids":feature["input_ids"], "attention_mask":feature["attention_mask"],"token_type_ids":feature["token_type_ids"]})
pred_onnx = sess.run(None, {"input_ids":feature["input_ids"], "attention_mask":feature["attention_mask"],"token_type_ids":feature["token_type_ids"]})
print(pred_onnx[0])
activation_fct = nn.Sigmoid()
print(activation_fct(torch.from_numpy(pred_onnx[0])))
print(sentence)
print("耗时：")
start_time = time.time()
for i in range(20):
    sess.run(None, {"input_ids":feature["input_ids"], "attention_mask":feature["attention_mask"], "token_type_ids":feature["token_type_ids"]})
    end_time = time.time()
    print(end_time-start_time)
    start_time = time.time()

#print(len(pred_onnx))
#print(np.shape(np.array(pred_onnx[0])), np.shape(np.array(pred_onnx[1])))
#print(pred_onnx)
activation_fct = nn.Sigmoid()

print("相似结果：",activation_fct(torch.from_numpy(pred_onnx[0])))
