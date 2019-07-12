# -*- coding: utf-8 -*-

"""
author: chao.yang
"""

import numpy as np
from scipy import stats
import math, copy
import sys,os
import argparse
import matplotlib.pyplot as plt

import cv2
import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

RuntimeRoot = "D:/yang/workPro/tensorStack/TensorStack/lib/x64/Release"
sys.path.append(RuntimeRoot)

from tensorstack.module import Module as mo
from tensorstack.node import Node
from tensorstack.backend.api import *

import quantize
from translator import Translator
from quantize_translator_option import QuantizeTranslatorOption

np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--num_bins', help='the num of bins of activation data', type=int, default=2048)
    parser.add_argument('--num_quantized_bins', help='the num of bins of quantized data', type=int, default=127)
    parser.add_argument('--dataset_path', help='Calibration dataset path', type=str, default=None)
    parser.add_argument('--model_path', help='Tensorstack model path', type=str, default=None)
    parser.add_argument('--device', help='compute device:cpu or gpu', type=str, default='cpu')
    parser.add_argument('--output_table', help='save path for calibration table file', type=str, default=None) 
    parser.add_argument('--output_module', help='save path for int8 module file', type=str, default=None) 

    args = parser.parse_args()
    return args,parser

def get_files(path,files):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            get_files(file_path,files)
        else:
            files.append(file_path)

def sava_calibration_table(path, quantize_node_list):
    file = open(path,'w')
    save_str_list = []
    for quantize_node in quantize_node_list.get().values():
        weight_str = quantize_node.name + "_weight"
        bottom_str = quantize_node.name
        for i in range(quantize_node.group_num):
            weight_str = weight_str + " " + str(quantize_node.weight_scale[i])
        bottom_str = bottom_str + " " + str(quantize_node.bottom_scale)
        save_str_list.append(weight_str)
        save_str_list.append(bottom_str)

    for write_str in save_str_list:
        file.write(write_str + "\n")

    file.close()

def build_dataset(dataset_files):
    tensor_dataset = []
    for data in dataset_files:
        cvimage = cv2.imread(data)
        tensor = Tensor(cvimage, UINT8, (1, cvimage.shape[0], cvimage.shape[1], cvimage.shape[2]))
        tensor_dataset.append(tensor)  
    return tensor_dataset      

def convert():

    args,parser = parse_args()
    print(args)

    num_bins = args.num_bins
    num_quantized_bins = args.num_quantized_bins
    model_path = args.model_path
    dataset_path = args.dataset_path
    compute_device = args.device
    output_path = args.output_table
    output_module = args.output_module

    # num_bins = 2048
    # num_quantized_bins = 127
    # model_path = "D:/yang/workPro/tensorStack/TensorStack/model/RN30.tsm"
    # dataset_path = "D:/yang/workPro/tensorStack/TensorStack/Quantize/2kx2k/quantization_248x248"
    # compute_device = "cpu"
    # output_path = "test.table"
    # output_module = "RN30_INT8_py.tsm"

    node_list = quantize.QuantizNodeList()  

    #load module 
    with open(model_path, "rb") as fi:
        module = mo.Load(fi)
    #quantize kernel
    quantize.quantize_weight(module, num_bins, num_quantized_bins, node_list)

    #build ts workbench
    _module = Module.Load(model_path)
    device = Device(compute_device,0)
    bench = Workbench(device=device)
    bench.setup_context()
    bench.setup_device()
    bench.set_computing_thread_number(1)
    bench.setup(bench.compile(_module))

    #preprosessor
    filter = ImageFilter(device=device)
    filter.center_crop(248,248)
    filter.to_float()
    filter.channel_swap([2, 1, 0])
    filter.scale(1 / 255.0)
    filter.to_chw()
    bench.bind_filter(0, filter)

    #get all calibration dataset
    dataset_files = []
    get_files(dataset_path,dataset_files)

    #build ts tensor dataset by dataset
    tensor_dataset = build_dataset(dataset_files)

    #quantize activation value
    quantize.quantize_bottom(bench, tensor_dataset, num_bins, num_quantized_bins, node_list)

    #translate and save int8 module
    quantize_option = QuantizeTranslatorOption(node_list)
    options = []
    options.append(quantize_option)
    translator = Translator(device, options)
    new_module = translator.translate(module)
    with open(output_module, "wb") as fo:
        mo.Save(fo, module=new_module)

    #save calibration data table
    sava_calibration_table(output_path, node_list)

if __name__ == "__main__":
    convert()