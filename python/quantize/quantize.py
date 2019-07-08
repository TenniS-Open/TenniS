# -*- coding: utf-8 -*-

"""
author: chao.yang
"""

import numpy as np
from scipy import stats
import copy
from tensorstack.node import Node


class QuantizNodeList:
    def __init__(self):
        self.quantizer_node_list = {}

    def push(self,quantize_node):
        self.quantizer_node_list[quantize_node.name] = quantize_node

    def get(self):
        return self.quantizer_node_list

class QuantizNode:
    def __init__(self, name, op_name, input_name, group_num, num_bins, num_quantized_bins):
        self.name = name
        self.op_name = op_name
        self.input_name = input_name
        self.group_num = group_num
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.bottom_max = 0.0
        self.bottom_distubution = np.zeros(num_bins)
        self.optimize_threshold = 0
        self.bottom_scale = 1.0
        self.group_zero = np.zeros(group_num)

    def initial_range(self, bottom_data):
        #find max
        max_val = np.max(bottom_data)
        min_val = np.min(bottom_data)
        self.bottom_max = max(self.bottom_max, max(abs(max_val), abs(min_val)))
        #initial reference distrubution interval
        # self.bottom_distubution_interval = self.bottom_max / self.num_bins
        # print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, self.bottom_max, self.bottom_distubution_interval))  

    def initial_histogram(self, bottom_data):
        # collect histogram of reference data
        threshhold = self.bottom_max
        # Note:current only consider positive axis,mayebe we can support asymmetric quantization
        hist, hist_edge = np.histogram(bottom_data, bins=self.num_bins, range=(0, threshhold))
        self.bottom_distubution += hist

    def quantize_bottom(self):
        candidate_distribution = np.array(self.bottom_distubution)  
        # pick threshold which minimizes KL divergence
        optimal_threshold,min_distribution_p,min_distribution_q = get_optimal_threshold(candidate_distribution,self.num_bins, self.num_quantized_bins+1) 
        self.optimize_threshold = optimal_threshold
        bottom_distubution_interval = self.bottom_max / self.num_bins
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, self.bottom_max, bottom_distubution_interval))
        threshold = (optimal_threshold + 0.5) * bottom_distubution_interval

        # get the calibration value
        self.bottom_scale = self.num_quantized_bins / threshold
        print("%-20s thresh_bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, optimal_threshold, threshold, bottom_distubution_interval, self.bottom_scale))

class QuantizConvNode(QuantizNode):
    def __init__(self, name, op_name, input_name, group_num, num_bins, num_quantized_bins):
        QuantizNode.__init__(self, name, op_name, input_name, group_num, num_bins, num_quantized_bins)
        self.weight_scale = np.zeros(group_num)

    def quantize_weight(self, weight_data):
        #just use maximum quantization directly because weight data is even-distributed always
        weight_data = np.array(weight_data.flatten())
        weight_group_data = np.array_split(weight_data,self.group_num)
        for i,data in enumerate(weight_group_data):
            threshold = max(abs(np.max(data)),abs(np.min(data)))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
                print("%-20s weight_number : %d threshhold : %-10.8f weight_scale : %-10.8f" % (self.name, i, threshold, self.weight_scale[i]))
            else:
                self.weight_scale[i] = self.num_quantized_bins / threshold
                print("%-20s weight_number : %d threshhold : %-10.8f weight_scale : %-10.8f" % (self.name, i, threshold, self.weight_scale[i]))

def smooth_distribution(p, eps=0.0001):
    """
    Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
         https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    params:
        p: distribution need to smooth
    returns:
        hist: distribution smoothed
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        #Note: There are maybe some bugs, qaq
        # hist = p.astype(np.float32)
        # hist[hist == 0] = 0.001
        # return hist
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist

def get_optimal_threshold(distribution, num_bins=2048, num_quantized_bins=128):
    """
    Given a candidate distribution, find the optimal threshold for quantizing it.
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
         
    params:
        distribution: a candidate distribution by histogram and normalize
        num_quantized_bins: num of mapping intervals
    returns:
        opt_thresh: num of bin to minium kl
    """
    #Note:some trick,delete value 0
    #ref:https://github.com/BUG1989/caffe-int8-convert-tools/issues/26
    distribution = distribution[1:]
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    kl_divergence = np.zeros(distribution.size - num_quantized_bins)

    distribution_p_list = []
    distribution_q_list = []

    for i in range(num_quantized_bins,distribution.size):
        sliced_nd_hist = copy.deepcopy(distribution[:i])
        # generate reference distribution p
        reference_distribution_p = sliced_nd_hist.copy()
        outliers_count = sum(distribution[i:])
        reference_distribution_p[i-1] += outliers_count
        # is_nonzeros[k] indicates whether distribution[k] is nonzero
        is_nonzeros = (reference_distribution_p != 0).astype(np.int32)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # merge distribution into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        quantized_distribution_q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                quantized_distribution_q[start:stop] = float(quantized_bins[j]) / float(norm)
        quantized_distribution_q[reference_distribution_p == 0] = 0

        reference_distribution_p[reference_distribution_p == 0] = 0.0001
        quantized_distribution_q[quantized_distribution_q == 0] = 0.0001

        #reference_distribution_p = smooth_distribution(reference_distribution_p)
        # # There is a chance that q is an invalid probability distribution.
        # try:
        #     quantized_distribution_q = smooth_distribution(quantized_distribution_q)
        # except ValueError:
        #     kl_divergence[i - num_quantized_bins] = float("inf")
        kl_divergence[i - num_quantized_bins] = stats.entropy(reference_distribution_p, quantized_distribution_q)
        distribution_p_list.append(reference_distribution_p)
        distribution_q_list.append(quantized_distribution_q)

    min_kl_divergence_idx = np.argmin(kl_divergence)
    opt_thresh = min_kl_divergence_idx + num_quantized_bins
    min_distribution_p = distribution_p_list[min_kl_divergence_idx]
    min_distribution_q = distribution_q_list[min_kl_divergence_idx]

    # plt.figure("test")
    # plt.hist(x=min_distribution_p.flatten(), bins=min_distribution_p.size, color='#0504aa',alpha=0.75)
    # #plt.hist(x=min_distribution_q, bins=min_distribution_q.size, color='#green',alpha=0.7, rwidth=0.85)
    # #plt.grid(axis='y', alpha=0.75)
    # plt.show()

    return opt_thresh,min_distribution_p,min_distribution_q

def node_quantize(node, num_bins, num_quantized_bins, quantize_node_list, ready_set):
    """
    quantize node of conv2d on ts model 
    params:
        node:ts Node
        num_bins: num of bins
        num_quantized_bins: num of qunatized bins
        quantize_node_list: list for quantize node
        ready_set: quantized node set
    returns:
        none
    """
    flag = True
    if node in ready_set:
        flag = False
    name = node.name
    # if name == 'stage3_unit2_conv3' or name == 'stage1_unit1_conv3' or name == 'stage1_unit1_conv2_with_bn' or name == 'stage2_unit4_conv2_with_bn':
    #     flag = False
    inputs = node.inputs
    if flag:
        op = node.op  
        #Note: input_name is mean the name of node of input stream.
        input_name = ""
        weight_node = Node()
        need_quantize = False
        if op == "conv2d" or op == "depthwise_conv2d":
            input_name = inputs[0].name
            need_quantize = True
            weight_node = inputs[1]
        elif op == "conv2d_v2" or op == "depthwise_conv2d_v2":
            input_name = inputs[0].name
            need_quantize = True
            weight_node = inputs[2]
        if need_quantize:
            weight_data = weight_node.get("value")
            kernel_num = weight_data.shape[0]
            quantized_node = QuantizConvNode(name, op, input_name, kernel_num, num_bins, num_quantized_bins)
            quantized_node.quantize_weight(weight_data)
            quantize_node_list.push(quantized_node)
            ready_set.add(node)
    
    for input in inputs:
        node_quantize(input, num_bins, num_quantized_bins, quantize_node_list, ready_set)

def quantize_weight(module, num_bins, num_quantized_bins, quantize_node_list):
    """
    quantize weight tensor of conv2d on ts model 
    params:
        module:ts module
        num_bins: num of bins
        num_quantized_bins: num of qunatized bins
        quantize_node_list:list for quantize node
    returns:
        none
    """
    output_nodes = module.outputs
    ready_set = set()
    for node in output_nodes:
        node_quantize(node, num_bins, num_quantized_bins, quantize_node_list, ready_set)
    
def quantize_bottom(bench, dataset, num_bins, num_quantized_bins, quantize_node_list):
    """
    quantize activation input tensor.
    Ref:http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    params:
        bench:ts workbench
        quantize_node_list:list for quantize node
    returns:
        none
    """
    node_list = quantize_node_list.get()
    #get all bottom data
    name_list = []
    for quantize_node in node_list.values():
        name_list.append(quantize_node.input_name)

    for i,data in enumerate(dataset):
        bench.input(0, data)
        bench.run_hook(name_list)
        if i % 100 == 0:
            print("initial activation range: %d in loop %d" % (i, len(dataset)))
        # initial activation range  
        for quantize_node in node_list.values():
            bottom_data = bench.output(quantize_node.input_name)
            quantize_node.initial_range(bottom_data.numpy)  

    for i,data in enumerate(dataset): 
        bench.input(0, data)
        bench.run_hook(name_list)
        if i % 100 == 0:
            print("initial histogram: %d in loop %d" % (i, len(dataset)))
        # initial histogram by activation range
        for quantize_node in node_list.values():
            bottom_data = bench.output(quantize_node.input_name)
            quantize_node.initial_histogram(bottom_data.numpy)

    # quantize bottom data
    for quantize_node in node_list.values():
        quantize_node.quantize_bottom()
