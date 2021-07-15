import os
import time
import sys
import argparse
import logging
import datetime

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, CalibrationMethod, QuantFormat
import onnxoptimizer
#import onnxsim

from PIL import Image

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from models.get_models import *
from models.mobilenetV2 import MobileNetV2
from utils.utils import *


def args_parsing():
    parser = argparse.ArgumentParser(description="ONNX optimization test tool")

    # Basic
    parser.add_argument('--data-dir', type=str, 
                        help='dataset directory',
                        default='/home/workspace/datasets/imagenet_1k')
    parser.add_argument('--model-file', type=str,
                        help='Path to model file',
                        default='/home/workspace/model_optimization/pth/test.onnx')
    parser.add_argument('--output-dir', type=str,
                        help='Path to output quantized model file',
                        default='/home/workspace/model_optimization/pth/')
    parser.add_argument('--output_prefix', type=str,
                        help='Output file naming rule. Quantized output model name will be \{prefix\}_\{quantization_method\}.onnx',
                        default='/home/workspace/model_optimization/pth/test.onnx')
    parser.add_argument('--eval', action='store_true')                    

    # DataLoader
    parser.add_argument('--train-batch-size', type=int,
                        help='Dataloader train batch size',
                        default=30)
    parser.add_argument('--eval-batch-size', type=int,
                        help='Dataloader evaluation batch size',
                        default=30)                    

    # Quantization
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--cal-batch-size', type=int,
                        help='Calibration batch size for per channel quan',
                        default=10)

    # Logger
    parser.add_argument('--logging-level', type=str,
                        help='Define logging level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
                        default='INFO')
    parser.add_argument('--log-dir', type=str,
                        help='Log file directory',
                        default='/home/workspace/model_optimization/log/ort/')
    parser.add_argument('--log-clear', action='store_true',
                        help='Delete all existing log files and make new one')
    
    return parser


def get_model_info(ort_session, file_name):
    mainlogger.info(f"Target model: {file_name}")

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type

    label_name = ort_session.get_outputs()[0].name
    label_shape = ort_session.get_inputs()[0].shape
    label_type = ort_session.get_inputs()[0].type
    mainlogger.info(f'Input Info: {input_name}, {input_shape}, {input_type}')
    mainlogger.info(f'Label Info: {label_name}, {label_shape}, {label_type}')


def main(args):
    # Set session options
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_profiling = False
    sess_options.log_severity_level = 3
    sess_options.intra_op_num_threads=1
    execution_provider = ["CPUExecutionProvider"]
    
    # Set opt filepath for offline mode
    #sess_options.optimized_model_filepath = f'{args.model_dir}{model_name}_ort_opt.onnx'
    
    # General options
    train_batch_size = 30
    eval_batch_size = 30
    num_eval_batches = 10
    _, data_loader_test = prepare_data_loaders(args.data_dir, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size)
    criterion = nn.CrossEntropyLoss()


    mainlogger.info(f"Model : {args.model_file}")
    
    try:
        model = onnx.load(args.model_file)
        onnx.checker.check_model(model)
    except:
        mainlogger.error(f"{args.model_file} is not an onnx file")
    
    
    mainlogger.debug("Model Info")
    mainlogger.debug(onnx.helper.printable_graph(model.graph))
    
    ort_session = onnxruntime.InferenceSession(args.model_file, sess_options)

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type

    label_name = ort_session.get_outputs()[0].name
    label_shape = ort_session.get_inputs()[0].shape
    label_type = ort_session.get_inputs()[0].type
    mainlogger.debug('Input Info: ', input_name, input_shape, input_type)
    mainlogger.debug('Label Info: ', label_name, label_shape, label_type)


    if args.quant:
        # Do quantization
        output_path = os.path.join(args.output_dir, args.output_prefix)

        # Apply onnxoptimizer
        model = onnxoptimizer.optimize(model)

        # Static U8U8
        dr = MobileNetV2DataReader(args.data_dir)
        onnxruntime.quantization.quantize_static(model, f'{output_path}_static.onnx', dr, activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)
        
        # Dynamic U8U8
        onnxruntime.quantization.quantize_dynamic(model, f'{output_path}_dynamic.onnx', activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)

        mainlogger.info("\nDone")
        

    if args.eval:
        # Do eval

        ort_session = onnxruntime.InferenceSession(args.model_file, sess_options)
        ort_session.set_providers(['CPUExecutionProvider'])

        run_iterative_benchmark(model=ort_session, criterion=criterion, data_loader=data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=args.eval_batch_size, is_onnx=True, iter=4, name=f'{args.model_file}')

    
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
    file_hadler = logging.FileHandler(f'{args.log_dir}ort_{now}.log', mode='w')
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
    #mainlogger.info("onnxsim: " + onnxsim.__version__ + '\n')

    mainlogger.debug(args)
    
    # Call main
    main(args)
