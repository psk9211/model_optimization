import os
import time
import sys

import onnx
import onnxruntime
import onnxoptimizer
import onnxsim

import netron

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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


def main():

    data_path = '/home/workspace/datasets/imagenet2012/imagenet_1k'
    model_dir = "/home/workspace/model_optimization/pth/"
    onnx_model_fixed = model_dir + "mobilenet_float.onnx"
    onnx_model = model_dir + 'mobilenetv2_torch170_pretrain_float.onnx'
    model = onnx.load(onnx_model)
    onnx.checker.check_model(model)
    
    print("Model Info")
    print(onnx.helper.printable_graph(model.graph))

    ort_session = onnxruntime.InferenceSession(onnx_model)
    
    train_batch_size = 30
    eval_batch_size = 30
    num_eval_batches = 10

    _, data_loader_test = prepare_data_loaders(data_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type

    label_name = ort_session.get_outputs()[0].name
    label_shape = ort_session.get_inputs()[0].shape
    label_type = ort_session.get_inputs()[0].type
    print('Input Info: ', input_name, input_shape, input_type)
    print('Label Info: ', label_name, label_shape, label_type)

    
    top1, top5, elapsed = evaluate(ort_session, criterion, data_loader_test, neval_batches=num_eval_batches)
    num_images = num_eval_batches * eval_batch_size
    print('Evaluation accuracy on %d images, %2.2f'%(num_images, top1.avg))
    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))


    ##################################
    # Optimize
    ##################################
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
    print("Optimized Model Info")
    print(onnx.helper.printable_graph(model.graph))

    

    """
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

    """


    """
    polished_model = onnx.utils.polish_model(model)
    print("Polished Model Info")
    print(onnx.helper.printable_graph(model.graph))
    IndexError: Input features.0.0.0.weight is undefined!
    """

    fixed = onnx.load(onnx_model_fixed)
    model_simp, check = onnxsim.simplify(fixed)
    assert check, "Simplified ONNX model could not be validated"
    print("Simpilfied Model Info")
    print(onnx.helper.printable_graph(model_simp.graph))
    onnx.save(model_simp, model_dir+'simplified.onnx')






if __name__ == "__main__":
    
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.backends.quantized.supported_engines)
    print(onnx.__version__)
    print(onnxruntime.__version__)

    main()