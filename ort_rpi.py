import os
import time
import sys
import argparse
import logging
import datetime

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType
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
                        default='/home/pi/projects/datasets/imagenet2012/imagenet_1k')
    parser.add_argument('--root-dir', type=str,
                        help='Project root directory',
                        default='/home/pi/projects/model_optimization/')
    parser.add_argument('--model-dir', type=str,
                        help='Model file directory',
                        default='/home/pi/projects/model_optimization/pth/')

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

    # Logger
    parser.add_argument('--logging-level', type=str,
                        help='Define logging level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
                        default='INFO')
    parser.add_argument('--log-dir', type=str,
                        help='Log file directory',
                        default='/home/pi/projects/model_optimization/log/ort/')
    parser.add_argument('--log-clear', type=bool,
                        help='Delete all existing log files and make new one',
                        default=False)
    
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


def onnxruntime_quantize(model, model_dir, data_dir, return_session=True, do_quant=True):
    if do_quant:
        
        # Static
        dr = MobileNetV2DataReader(data_dir)
        onnxruntime.quantization.quantize_static(model, model_dir+'onnxruntime_static.onnx', dr)
        #onnxruntime.quantization.quantize_static(model, model_dir+'onnxruntime_static_aui8_wi8.onnx', dr, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        
        # Dynamic
        onnxruntime.quantization.quantize_dynamic(model, model_dir+'onnxruntime_dynamic.onnx')
        onnxruntime.quantization.quantize_dynamic(model, model_dir+'onnxruntime_dynamic_aui8_wi8.onnx', activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        onnxruntime.quantization.quantize_dynamic(model, model_dir+'onnxruntime_dynamic_ai8_wui8.onnx', activation_type=QuantType.QInt8, weight_type=QuantType.QUInt8)
        onnxruntime.quantization.quantize_dynamic(model, model_dir+'onnxruntime_dynamic_ai8_wi8.onnx', activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

        # QAT        
        onnxruntime.quantization.quantize_qat(model, model_dir+'onnxruntime_qat.onnx')
        onnxruntime.quantization.quantize_qat(model, model_dir+'onnxruntime_qat_aui8_wi8.onnx', activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        onnxruntime.quantization.quantize_qat(model, model_dir+'onnxruntime_qat_ai8_wui8.onnx', activation_type=QuantType.QInt8, weight_type=QuantType.QUInt8)
        onnxruntime.quantization.quantize_qat(model, model_dir+'onnxruntime_qat_ai8_wi8.onnx', activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    if return_session:
        '''
        2021.02.01
        ConvInteger doesn't supporst Int8, it only supports uint8 activation, uint8 weight.
        '''

        static = []
        static_aui8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_static.onnx')
        static.append(static_aui8_wui8_sess)

        dynamic = []
        dynamic_aui8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_dynamic.onnx')
        dynamic.append(dynamic_aui8_wui8_sess)

        qat = []
        qat_aui8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_qat.onnx')
        qat.append(qat_aui8_wui8_sess)

        return static, dynamic, qat




def main(args):

    onnx_model = args.model_dir + "mobilenet_float.onnx"
    model = onnx.load(onnx_model)
    onnx.checker.check_model(model)
    
    mainlogger.debug("Model Info")
    mainlogger.debug(onnx.helper.printable_graph(model.graph))

    # Set session options
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads=1

    ort_session = onnxruntime.InferenceSession(onnx_model)
    
    train_batch_size = 30
    eval_batch_size = 30
    num_eval_batches = 10

    _, data_loader_test = prepare_data_loaders(args.data_dir, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size)
    criterion = nn.CrossEntropyLoss()
    
    top1, top5, elapsed = evaluate(ort_session, criterion, data_loader_test, neval_batches=num_eval_batches, is_onnx=True)
    num_images = num_eval_batches * eval_batch_size
    mainlogger.info(f'Evaluation accuracy on {num_images} images, top1: {top1.avg:.2f} / top5: {top5.avg:.2f}')
    mainlogger.info(f'Elapsed time: {elapsed/num_images*1000:.2f}ms\n')
    

    ##################################
    # Quantization
    ##################################
    static_sess, dynamic_sess, qat_sess = onnxruntime_quantize(onnx_model, args.model_dir, args.data_dir, return_session=True, do_quant=False)

    for sess in static_sess:
        run_iterative_benchmark(model=sess, criterion=criterion, data_loader=data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=args.eval_batch_size, is_onnx=True, iter=10, name=f'Static Quant: {sess}')

    for sess in dynamic_sess:   
        run_iterative_benchmark(model=sess, criterion=criterion, data_loader=data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=args.eval_batch_size, is_onnx=True, iter=10, name=f'Dynamic Quant: {sess}')

    for sess in dynamic_sess:
        run_iterative_benchmark(model=sess, criterion=criterion, data_loader=data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=args.eval_batch_size, is_onnx=True, iter=10, name=f'QAT Quant: {sess}')


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

    mainlogger.debug(args)
    
    # Call main
    main(args)
