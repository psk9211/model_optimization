import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import torch.quantization

import os
import time
import sys

from models.mobilenetV2 import MobileNetV2
from utils.utils import *


def per_channel_quantize(model_file, model_name, )

    num_calibration_batches = 10

    per_channel_quantized_model = load_model(model_file, model_name)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(per_channel_quantized_model.qconfig)

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print(top1, top5)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)