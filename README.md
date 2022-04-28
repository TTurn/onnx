# onnx

### requirments

#### cpu:
    transformers==4.2.1
    torch==1.8.1+cu101
    onnxruntime==1.10.0
    sentence-transformers==2.0.0
#### gpu:
    sentence-transformers==2.0.0
    transformers==4.2.1
    onnxruntime-gpu==1.4.0
    torch==1.8.1+cu101
    CUDA Version: 10.1

more cuda dependencies information in :https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html


### usage:
#### 1.tran a pytorch model 
    example:
        train a cross-encoder model in: https://github.com/TTurn/cross-encoder
#### 2.export model from pytorch
    python3.8 convert_onnx.py
    https://blog.csdn.net/choose_c/article/details/124481697
    
#### 2.quantiz model
    python3.8 quantize.py
    https://blog.csdn.net/choose_c/article/details/124482056
  
#### 3.optimize the onnx model
    python3.8 optimize.py

#### 4.test 
    python3.8 test_onnx.py
