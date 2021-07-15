# Introduction

[NOTA](https://nota.ai)와 함께하는 프로젝트 페이지입니다.

아래와 같은 내용을 다루고 있습니다.
- PyTorch model to ONNX Export
- ORT Quantization


# 사용법

## ONNX Export

- Pretrained FP32 모델 (torchvision.models)
```cmd
python pth2onnx.py --pretrained --arch='mobilenet_v2' --opset=12 \
        --output='/home/workspace/model_optimization/pth/mobilenet_v2.onnx'
```

- Custom 모델
```cmd
python pth2onnx.py --arch='mobilenet_v2' --opset=12 \
        --model_file='/home/workspace/model_optimization/pth/mobilenet_v2.pth \
        --output='/home/workspace/model_optimization/pth/mobilenet_v2.onnx'
```
torchvision.models 에 정의된 모델이 아닌경우, models/ 내에 모델 class를 정의한뒤 main 함수에서 불러오면 됩니다.


## Quantization & Evaluation

- FP32 모델을 이용해 Static/Dynamic quantization 진행
```cmd
python ort.py --quant --data_dir='/home/workspace/datasets/imagenet_1k' \
        --model_file='/home/workspace/model_optimization/pth/test_fp32.onnx' \
        --output-dir='/home/workspace/model_optimization/pth' \
        --output_prefix='mobilenetv2' \
```

- ONNX 모델을 evalution
```cmd
python ort.py --eval --data_dir='/home/workspace/datasets/imagenet_1k' \
        --model_file='/home/workspace/model_optimization/pth/mobilenetv2_static.onnx'
```