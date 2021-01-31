import numpy as np
import os
import time
import sys
import argparse
import logging
import datetime

import onnx
import onnxruntime
import onnxoptimizer
import onnxsim

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.quantization
import torch.onnx

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from models.get_models import *
from models.mobilenetV2 import MobileNetV2
from utils.utils import *

"""


onnx_model_name = ['mobilenetv2_torch170_pretrain_float.onnx',
                    'mobilenetv2_script_quan.onnx',
                    'mobilenetv2_qat.onnx']
"""

def args_parsing():

    parser = argparse.ArgumentParser(description="Model optimization test tool")

    # Basic
    parser.add_argument('--data-dir', type=str, 
                        help='dataset directory',
                        default='/home/workspace/datasets/imagenet2012/imagenet_1k')
    parser.add_argument('--root-dir', type=str,
                        help='Project root directory',
                        default='/home/workspace/model_optimization/')
    parser.add_argument('--model-dir', type=str,
                        help='Model file directory',
                        default='/home/workspace/model_optimization/pth/')
    parser.add_argument('--arch', type=str,
                        help='Neural net architecture',
                        default='mobilenet')
    parser.add_argument('--pretrained-model', type=str,
                        help='Name of pretrained model file',
                        default='mobilenetv2_torch170_pretrain_float.pth')
    parser.add_argument('--static-model', type=str,
                        help='Name of static model file',
                        default='mobilenet_quantization_scripted_quantized_qnnpack.pth')
    parser.add_argument('--qat-model', type=str,
                        help='Name of pretrained model file',
                        default='mobilenetv2_script_qat.pth')    
    parser.add_argument('--output-model', type=str,
                        help='Name of optimized model file. pth and onnx files will be generate',
                        default='mobilenetv2_script_quan')
    parser.add_argument('--device', type=str,
                        help='',
                        default='cpu')

    # DataLoader
    parser.add_argument('--train-batch-size', type=int,
                        help='Dataloader train batch size',
                        default=30)
    parser.add_argument('--eval-batch-size', type=int,
                        help='Dataloader evaluation batch size',
                        default=30)                    

    # Quantization
    parser.add_argument('--quan-engine', type=str,
                        help='Quantization backend engine: ["qnnpack", "fbgemm"]',
                        default='qnnpack')                  
    parser.add_argument('--cal-batch-size', type=int,
                        help='Calibration batch size for per channel quan',
                        default=10)
    parser.add_argument('--qat-epoch', type=int,
                        help='The number of QAT epoch',
                        default=30)

    # Tool config                        
    parser.add_argument('--run-eval', type=bool,
                        help='Do evaluation benchmark or not',
                        default=False)
    parser.add_argument('--run-static', type=bool,
                        help='Perform per channel quantization',
                        default=False)
    parser.add_argument('--run-qat', type=bool,
                        help='Perform QAT',
                        default=False)
    parser.add_argument('--onnx-export', type=bool,
                        help='Export pytorch model to onnx',
                        default=False)
    parser.add_argument('--onnx-optim', type=bool,
                        help='Perform ONNX optimization',
                        default=True)
    parser.add_argument('--onnxruntime-optim', type=bool,
                        help='Perform onnxruntime optimization',
                        default=True)
    parser.add_argument('--onnxsim-optim', type=bool,
                        help='Perform ONNX simplifier optimization',
                        default=True)

    # Logger
    parser.add_argument('--logging-level', type=str,
                        help='Define logging level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
                        default='WARNING')
    parser.add_argument('--log-dir', type=str,
                        help='Log file directory',
                        default='/home/workspace/model_optimization/log/')
    parser.add_argument('--log-clear', type=bool,
                        help='Delete all existing log files and make new one',
                        default=False)

    return parser


def get_model_info(model, file_name):
    mainlogger.info(f"Target model: {file_name}")
    mainlogger.debug('Model state dict:')
    for param_tensor in model.state_dict():
        mainlogger.debug(f"{param_tensor} {model.state_dict()[param_tensor].size()}")




def main(args):
    print('==================================================')
    print('Model optimization Test')
    print('==================================================\n\n')

    # Model Setting
    model_name = MobileNetV2()

    data_loader, data_loader_test = prepare_data_loaders(args.data_dir, 
                train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    #####################
    # Get Models & INFO
    #####################
    float_model = load_model(args.model_dir + args.pretrained_model, model_name).to(args.device)
    float_model.eval()
    float_model.fuse_model()

    torch.backends.quantized.engine = args.quan_engine 
    per_channel_quantized_model = torch.jit.load(args.model_dir + args.static_model)
    qat_model = torch.jit.load(args.model_dir + args.qat_model)

    get_model_info(float_model, args.pretrained_model)
    get_model_info(per_channel_quantized_model, args.static_model)
    get_model_info(qat_model, args.qat_model)

    #####################
    # Eval & Time benchmark
    #####################
    num_eval_batches = 10
    if args.run_eval:
        num_images = num_eval_batches * args.eval_batch_size

        mainlogger.info(f'=======Benchmarks=======')

        # Fp32 model
        top1, top5, time = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        mainlogger.info(f'Model: {args.pretrained_model}')
        mainlogger.info(f'Evaluation accuracy on {num_eval_batches * args.eval_batch_size} images, top1: {top1.avg:.2f} / top5: {top5.avg:.2f}')
        mainlogger.info(f'Elapsed time: {time/num_images*1000:.2f}ms\n')



        # Per-channel quantized model
        top1, top5, time = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        mainlogger.info(f'Model: {scripted_quantized_model_file}')
        mainlogger.info(f'Evaluation accuracy on {num_eval_batches * args.eval_batch_size} images, top1: {top1.avg:.2f} / top5: {top5.avg:.2f}')
        mainlogger.info(f'Elapsed time: {time/num_images*1000:.2f}ms\n')

        time_static = run_benchmark(args.model_dir + scripted_quantized_model_file, data_loader_test)
        mainlogger.info(f'Elapsed time using run_benchmark(): {time_static/num_images*1000:.2f}ms\n')

        # QAT model
        top1, top5, time = evaluate(qat_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        mainlogger.info(f'Model: {qat_model_file}')
        mainlogger.info(f'Evaluation accuracy on {num_eval_batches * args.eval_batch_size} images, top1: {top1.avg:.2f} / top5: {top5.avg:.2f}')
        mainlogger.info(f'Elapsed time: {time/num_images*1000:.2f}ms\n')


    if args.onnx_export:
        #####################
        # Export to ONNX
        #####################
        print('\n\nExport to ONNX')
        batch_size = num_eval_batches
        x = torch.randn(1, 3, 224, 224, requires_grad=True)

        torch.onnx.export(float_model,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        '/home/workspace/model_optimization/pth/mobilenet_float.onnx',   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        verbose=True,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    
 
    
if __name__ == "__main__":
    
    # Get args
    parser = args_parsing()
    args = parser.parse_args()

    # Set Logger & Handlers
    mainlogger = logging.getLogger()
    mainlogger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt="%m-%d %H:%M")
    
    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    console.setFormatter(log_formatter)
    
    if args.log_clear:
        os.system(f'rm {args.log_dir}*.log')

    now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
    file_hadler = logging.FileHandler(f'{args.log_dir}{now}.log', mode='w')
    file_hadler.setLevel(logging.DEBUG)
    file_hadler.setFormatter(log_formatter)    

    mainlogger.addHandler(console)
    mainlogger.addHandler(file_hadler)

    # Libraries version check
    mainlogger.info("=====Version Check=====")
    mainlogger.info("Pytorch: " + torch.__version__)
    mainlogger.info("Torchvision: " + torchvision.__version__)
    mainlogger.info(f"Torch CUDA: {torch.cuda.is_available()}")
    mainlogger.info(f"Torch quantization engines: {torch.backends.quantized.supported_engines}")
    mainlogger.info("ONNX: " + onnx.__version__)
    mainlogger.info("onnxruntime: " + onnxruntime.__version__)
    mainlogger.info("onnxsim: " + onnxsim.__version__ + '\n')

    mainlogger.debug(args)
    
    # Call main
    main(args)
