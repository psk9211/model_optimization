import os
import time
import sys
import argparse
import logging
import datetime

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType
import onnxoptimizer
import onnxsim

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
                        default='/home/workspace/datasets/imagenet2012/imagenet_1k')
    parser.add_argument('--root-dir', type=str,
                        help='Project root directory',
                        default='/home/workspace/model_optimization/')
    parser.add_argument('--model-dir', type=str,
                        help='Model file directory',
                        default='/home/workspace/model_optimization/pth/')

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
                        default='/home/workspace/model_optimization/log/ort/')
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



"""
def evaluate(ort_session, criterion, data_loader, neval_batches):
    elapsed = 0
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            start = time.time()

            output = ort_session.run([ort_session.get_outputs()[0].name], {ort_session.get_inputs()[0].name: to_numpy(image)})[0]

            end = time.time()
            elapsed = elapsed + (end-start)

            output = torch.from_numpy(output)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5, elapsed
    
    return top1, top5, elapsed
"""


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
        #static_aui8_wi8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_static_aui8_wi8.onnx')
        static.append(static_aui8_wui8_sess)

        dynamic = []
        dynamic_aui8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_dynamic.onnx')
        dynamic.append(dynamic_aui8_wui8_sess)
        '''
        dynamic_aui8_wi8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_dynamic_aui8_wi8.onnx')
        dynamic.append(dynamic_aui8_wi8_sess)
        dynamic_ai8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_dynamic_ai8_wui8.onnx')
        dynamic.append(dynamic_ai8_wui8_sess)
        dynamic_ai8_wi8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_dynamic_ai8_wi8.onnx')
        dynamic.append(dynamic_ai8_wi8_sess)
        '''

        qat = []
        qat_aui8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_qat.onnx')
        qat.append(qat_aui8_wui8_sess)
        '''
        qat_aui8_wi8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_qat_aui8_wi8.onnx')
        qat.append(qat_aui8_wi8_sess)
        qat_ai8_wui8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_qat_ai8_wui8.onnx')
        qat.append(qat_ai8_wui8_sess)
        qat_ai8_wi8_sess = onnxruntime.InferenceSession(model_dir+'onnxruntime_qat_ai8_wi8.onnx')
        qat.append(qat_ai8_wi8_sess)
        '''

        return static, dynamic, qat




def main(args):

    onnx_model = args.model_dir + "mobilenet_float.onnx"
    #onnx_model = args.model_dir + 'mobilenetv2_torch170_pretrain_float.onnx'
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

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type

    label_name = ort_session.get_outputs()[0].name
    label_shape = ort_session.get_inputs()[0].shape
    label_type = ort_session.get_inputs()[0].type
    #mainlogger.debug('Input Info: ', input_name, input_shape, input_type)
    #mainlogger.debug('Label Info: ', label_name, label_shape, label_type)

    '''
    top1, top5, elapsed = evaluate(ort_session, criterion, data_loader_test, neval_batches=num_eval_batches, is_onnx=True)
    num_images = num_eval_batches * eval_batch_size
    mainlogger.info(f'Evaluation accuracy on {num_images} images, top1: {top1.avg:.2f} / top5: {top5.avg:.2f}')
    mainlogger.info(f'Elapsed time: {elapsed/num_images*1000:.2f}ms\n')
    '''

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


    """
    all_passes = onnxoptimizer.get_available_passes()
    print("\n\nAvailable optimization passes:")
    for p in all_passes:
        print(p)
    print()
    
    
    inferred_model = onnx.shape_inference.infer_shapes(model)
    #print(inferred_model.graph.value_info)
    
    passes = ['fuse_consecutive_transposes']
    all_passes = ['eliminate_deadend',
                    'eliminate_duplicate_initializer',
                    'eliminate_identity',
                    'eliminate_nop_cast',
                    'eliminate_nop_dropout',
                    'eliminate_nop_flatten',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_log_softmax',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm',
                    'lift_lexical_references',
                    'nop',
                    'split_init',
                    'split_predict']
    optimized_model = onnxoptimizer.optimize(inferred_model, all_passes)
    mainlogger.debug("Optimized Model Info")
    mainlogger.debug(onnx.helper.printable_graph(model.graph))

    

    
    ort_session = onnxruntime.InferenceSession(optimized_model)
    top1, top5, elapsed = evaluate(ort_session, criterion, data_loader_test, neval_batches=num_eval_batches)
    num_images = num_eval_batches * eval_batch_size
    print('Evaluation accuracy on %d images, %2.2f'%(num_images, top1.avg))
    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    #print("\nOptimized Model Info")
    #print(onnx.helper.printable_graph(optimized_model.graph))
    
    IndexError: Input features.0.0.0.weight is undefined!
        Related:
        https://github.com/onnx/onnx/issues/2902
        https://github.com/onnx/onnx/issues/2903

    polished_model = onnx.utils.polish_model(model)
    print("Polished Model Info")
    print(onnx.helper.printable_graph(model.graph))
    IndexError: Input features.0.0.0.weight is undefined!
    

    fixed = onnx.load(onnx_model_fixed)
    model_simp, check = onnxsim.simplify(fixed)
    assert check, "Simplified ONNX model could not be validated"
    mainlogger.info("Simpilfied Model Info")
    mainlogger.info(onnx.helper.printable_graph(model_simp.graph))
    onnx.save(model_simp, args.model_dir+'simplified.onnx')
    """
    



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
    mainlogger.info("onnxsim: " + onnxsim.__version__ + '\n')

    mainlogger.debug(args)
    
    # Call main
    main(args)
