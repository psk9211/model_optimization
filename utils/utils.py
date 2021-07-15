import os
import time
import sys

import numpy as np
import scipy.io
from PIL import Image

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import onnx

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches, is_onnx=False):
    elapsed = 0
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        if is_onnx:
            for image, target in data_loader:
                start = time.time()

                # model parameter => onnxruntime.InferenceSession()
                output = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: to_numpy(image)})[0]

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
                
        else:
            model.eval()
            for image, target in data_loader:
                start = time.time()

                output = model(image)
                loss = criterion(output, target)

                end = time.time()
                elapsed = elapsed + (end-start)

                cnt += 1
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                print('.', end = '')
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
                if cnt >= neval_batches:
                    return top1, top5, elapsed

    return top1, top5, elapsed


def load_model(model_file, model_name):
    #model = MobileNetV2()
    model = model_name
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Model: ' + model_file)
    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed


def run_iterative_benchmark(model, criterion, data_loader, num_eval_batches, eval_batch_size, is_onnx, name, iter=10):
    top1_toal = 0
    top5_toal = 0 
    time_total = 0
    for i in range(iter):
        top1, top5, time = evaluate(model, criterion, data_loader, neval_batches=num_eval_batches, is_onnx=is_onnx)  
        top1_toal += top1.avg
        top5_toal += top5.avg
        time_total += time

    num_images = num_eval_batches * eval_batch_size
    print(f'Model: {name}')
    print(f'Average on {iter} times')
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, top1: {top1_toal/iter:.2f} / top5: {top5_toal/iter:.2f}')
    print(f'Elapsed time: {time_total/num_images*1000/iter:.2f}ms\n')




def onnx_export(torch_model, onnx_model_name, random_input, opset_version=10, operator_export_type=torch.onnx.OperatorExportTypes.ONNX):
    batch_size = random_input.size()[0]
    torch.onnx.export(torch_model,               # model being run
                    random_input,                         # model input (or a tuple for multiple inputs)
                    onnx_model_name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=opset_version,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}},
                    verbose=True,
                    operator_export_type=operator_export_type)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

"""
def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    valdir_names = os.listdir(images_folder+'/val')
    image_names = []
    for dir in valdir_names:
        image_names.extend(os.listdir(images_folder+'/val/'+dir))
    
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    #print(batch_filenames)

    for image_name, dir in zip(batch_filenames, valdir_names):
        image_filepath = images_folder + '/val/' + dir + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - \
        np.array([123.68, 116.78, 103.94], dtype=np.float32)
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data
"""


def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_data = to_numpy(transform(pillow_img))
        
        nchw_data = np.expand_dims(input_data, axis=0)        
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class MobileNetV2DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            nhwc_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)



class ImageNetDataReader(CalibrationDataReader):
    def __init__(self, image_folder,
                       width=224,
                       height=224,
                       start_index=0,
                       end_index=0,
                       stride=1,
                       batch_size=1,
                       model_path='augmented_model.onnx',
                       input_name='data'):
        '''
        :param image_folder: image dataset folder
        :param width: image width
        :param height: image height 
        :param start_index: start index of images
        :param end_index: end index of images
        :param stride: image size of each data get 
        :param batch_size: batch size of inference
        :param model_path: model name and path
        :param input_name: model input name
        '''

        self.image_folder = image_folder + "/val"
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.datasize = 0
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(self.image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1
        self.batch_size = batch_size
        self.input_name = input_name

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:      
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width 
        height = self.width 
        nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(self.image_folder, height, width, self.start_index, self.stride)
        input_name = self.input_name

        data = []
        for i in range(len(nchw_data_list)):
            nhwc_data = nchw_data_list[i]
            file_name = filename_list[i]
            data.append({input_name: nhwc_data})
        return data

    def load_batches(self):
        width = self.width 
        height = self.height 
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index 
            nchw_data_list, filename_list, image_size_list = self.preprocess_imagenet(self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            for i in range(len(nchw_data_list)):
                nhwc_data = np.squeeze(nchw_data_list[i], 0)
                nchw_data_batch.append(nhwc_data)
            batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
            data = {input_name: batch_data}

            batches.append(data)

        return batches

    def preprocess_imagenet(self, images_folder, height, width, start_index=0, size_limit=0):
        '''
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter start_index: image index to start with   
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        '''
    
        def preprocess_images(input, channels=3, height=224, width=224):
            image = input.resize((width, height), Image.ANTIALIAS)
            input_data = np.asarray(image).astype(np.float32)
            if len(input_data.shape) != 2:
                input_data = input_data.transpose([2, 0, 1])       
            else:
                input_data = np.stack([input_data] * 3)
            mean = np.array([0.079, 0.05, 0]) + 0.406
            std = np.array([0.005, 0, 0.001]) + 0.224
            for channel in range(input_data.shape[0]):
                input_data[channel, :, :] = (input_data[channel, :, :] / 255 - mean[channel]) / std[channel]
            return input_data
    
        image_names = os.listdir(images_folder)
        image_names.sort() 
        if size_limit > 0 and len(image_names) >= size_limit:
            end_index = start_index + size_limit
            if end_index > len(image_names):
                end_index = len(image_names)       
            batch_filenames = [image_names[i] for i in range(start_index, end_index)]
        else:
            batch_filenames = image_names
    
        unconcatenated_batch_data = []
        image_size_list = []
    
        for image_name in batch_filenames:  
            image_filepath = images_folder + '/' + image_name
            img = Image.open(image_filepath)

            # ILSVRC2012_val_00019877.JPEG is CMYK
            if img.mode == 'CMYK':
                print(f"This image is CMYK 4 channel format: {image_filepath}")
                img = img.convert('RGB') 
                
            image_data = preprocess_images(img)
            image_data = np.expand_dims(image_data, 0)
            unconcatenated_batch_data.append(image_data)
            image_size_list.append(np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2))
    
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data, batch_filenames, image_size_list

    def get_synset_id(self, image_folder, offset, dataset_size):
        ilsvrc2012_meta = scipy.io.loadmat(image_folder + "/devkit/data/meta.mat")
        id_to_synset = {}    
        for i in range(1000):
            id = int(ilsvrc2012_meta["synsets"][i,0][0][0][0])
            id_to_synset[id] = ilsvrc2012_meta["synsets"][i,0][1][0]
        
        synset_to_id = {}
        file = open(image_folder + "/synset_words.txt","r")
        index = 0
        for line in file:
            parts = line.split(" ")
            synset_to_id[parts[0]] = index
            index = index + 1
        file.close()
      
        file = open(image_folder + "/devkit/data/ILSVRC2012_validation_ground_truth.txt","r")
        id = file.read().strip().split("\n")
        id = list(map(int, id))
        file.close()
        
        image_names = os.listdir(image_folder + "/val")
        image_names.sort()
        image_names = image_names[offset : offset + dataset_size]
        seq_num = []
        for file in image_names:
            seq_num.append(int(file.split("_")[-1].split(".")[0]))
        id = np.array([id[index - 1] for index in seq_num])       
        synset_id = np.array([synset_to_id[id_to_synset[index]] for index in id])
    
        # one-hot encoding
        synset_id_onehot = np.zeros((len(synset_id), 1000), dtype=np.float32)
        for i, id in enumerate(synset_id):
            synset_id_onehot[i, id] = 1.0
        return synset_id_onehot


def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    input = model.graph.input
    input_name = input[0].name
    shape = input[0].type.tensor_type.shape
    dim = shape.dim
    if not dim[0].dim_param:
        dim[0].dim_param = 'N'
        model = onnx.shape_inference.infer_shapes(model)        
        model_name = model_path.split(".")
        model_path = model_name[0] + "_dynamic.onnx"
        onnx.save(model, model_path)
    return [model_path, input_name]

def get_dataset_size(dataset_path, calibration_dataset_size, batch_size):
    total_dataset_size = len(os.listdir(dataset_path + "/val"))        
    if calibration_dataset_size > total_dataset_size:
        print("Warning: calibration data size is bigger than available dataset. Will assign half of the dataset for calibration")
        calibration_dataset_size = total_dataset_size // 2
    calibration_dataset_size =  (calibration_dataset_size // batch_size) * batch_size
    if calibration_dataset_size == 0:
        print("Warning: No dataset is assigned for calibration. Please use bigger dataset")

    prediction_dataset_size = ((total_dataset_size - calibration_dataset_size) // batch_size) * batch_size
    if prediction_dataset_size <= 0:
        print("Warning: No dataset is assigned for evaluation. Please use bigger dataset")
    return [calibration_dataset_size, prediction_dataset_size] 

