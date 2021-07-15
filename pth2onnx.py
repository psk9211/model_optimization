import argparse
import logging

import torch
import torch.onnx
import torchvision

from models import *

"""
dir(torchvision.models)

['AlexNet', 'DenseNet', 'GoogLeNet', 'GoogLeNetOutputs', 'Inception3', 'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'MobileNetV3', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', '_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_utils', 'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 'quantization', 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'utils', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']
"""

def args_parsing():

    parser = argparse.ArgumentParser(description="PyTorch to ONNX Export")

    # Basic
    parser.add_argument('--arch', type=str,
                        help='Neural net architecture',
                        default='mobilenet')
    parser.add_argument('--model_file', type=str,
                        help='PyTorch model file',
                        default='/home/workspace/model_optimization/pth/mobilenet_v2.pth')
    parser.add_argument('--output', type=str,
                        help='Path to output onnx model',
                        default='/home/workspace/model_optimization/pth/mobilenet_float.onnx')

    parser.add_argument('--opset', type=int, default=12)
    parser.add_argument('--pretrained', action='store_true')

    # Logger
    parser.add_argument('--logging-level', type=str,
                        help='Define logging level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
                        default='INFO')

    return parser


def main(args):
    
    available_model_zoo = dir(torchvision.models)
    if args.arch not in available_model_zoo:
        raise ValueError(f'{args.arch} is not supported in torchvision model zoo: {available_model_zoo}')

    if args.pretrained:
        # Use torchvision model zoo
        model = torchvision.models.__dict__[args.arch](pretrained=True)
    else:
        model = torchvision.models.__dict__[args.arch](pretrained=False)
        model.load_state_dict(torch.load(args.model_file))


    # Export to ONNX
    mainlogger.info('\n\nExport to ONNX')
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    args.output,               # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=args.opset,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}},
                    verbose=True,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    
if __name__ == "__main__":
    
    # Get args
    parser = args_parsing()
    args = parser.parse_args()

    # Set Logger & Handlers
    mainlogger = logging.getLogger()
    mainlogger.setLevel(args.logging_level)

    log_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt="%m-%d %H:%M")
    
    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    console.setFormatter(log_formatter)  
    mainlogger.addHandler(console)

    # Libraries version check
    mainlogger.info("=====Version Check=====")
    mainlogger.info("Pytorch: " + torch.__version__)
    mainlogger.info("Torchvision: " + torchvision.__version__)
    mainlogger.info(f"Torch CUDA: {torch.cuda.is_available()}")
    mainlogger.info(f"Torch quantization engines: {torch.backends.quantized.supported_engines}")

    mainlogger.debug(args)
    
    # Call main
    main(args)
