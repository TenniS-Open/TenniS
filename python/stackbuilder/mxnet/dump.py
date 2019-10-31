#!python
# coding: UTF-8
"""
author: kier
"""

import mxnet
import numpy
import os
import tensorstack as ts


def find_epoch_params(model_prefix, epoch):
    import os
    for i in range(10):
        model_params = "{}-{}{}.params".format(model_prefix, '0' * i, epoch)
        if os.path.exists(model_params):
            return model_params
    raise FileNotFoundError("{}-{}.params".format(model_prefix, epoch))


def load_params(model_params, model_json):
    save_dict = mxnet.nd.load(model_params)
    symbol = mxnet.sym.load(model_json)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


def load_checkpoint(model_prefix, epoch):
    model_params = find_epoch_params(model_prefix, epoch)
    model_json = '%s-symbol.json' % (model_prefix, )

    return load_params(model_params, model_json)


def dump_image(model_prefix, epoch, data, output_root, inputs=None, outputs=None):
    """
    :param model_prefix:
    :param epoch:
    :param data: Union[numpy.ndarray, str]
    :param output_root: str, path to output root
    :param inputs: str, input label name
    :param outputs: list of str, output label name
    :return: None
    """
    model_params = find_epoch_params(model_prefix, epoch)
    model_json = '%s-symbol.json' % (model_prefix, )

    print("[INFO] Loading params: {}".format(model_params))
    print("[INFO] Loading json: {}".format(model_json))

    # set parameter
    if inputs is None:
        inputs = 'data'
    if outputs is not None and not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # get data, if image
    data = ts.tensor.compatible_string(data)
    if isinstance(data, str):
        import cv2
        image_path = data
        data = cv2.imread(image_path)
        if data is None:
            raise ValueError("Can not open or access image: {}".format(image_path))
        data = data[:, :, [2, 1, 0]]                        # to rgb
        data = numpy.asarray(data, dtype=numpy.float32)     # to float
        data /= 255.0                                       # div std
        data = numpy.transpose(data, [2, 0, 1])             # to chw
        data = numpy.expand_dims(data, 0)                   # to 4-D

    # load graph
    sym, arg_params, aux_params = load_params(model_params, model_json)

    print("[INFO] Graph loaded.")

    # load data
    batch_size = data.shape[0]
    label = numpy.zeros((batch_size, 1))
    data_iter = mxnet.io.NDArrayIter(data=data, label=label, batch_size=batch_size)
    data_shape = data.shape

    # list outputs
    sym_internals = sym.get_internals()
    all_output_set = sym_internals.list_outputs()

    # get output
    if outputs is None:
        outputs = all_output_set
    else:
        temp_outputs = []
        for name in outputs:
            if name not in all_output_set:
                name_output = name + '_output'
                if name_output not in all_output_set:
                    raise ValueError("{} not in graph".format(name))
                name = name_output
            temp_outputs.append(name)
        outputs = temp_outputs

    # filter outputs
    temp_outputs = []
    for name in outputs:
        if name != inputs and not name.endswith('_output'):
            continue
        temp_outputs.append(name)
    outputs = temp_outputs

    # got outputs group
    output_group = mxnet.symbol.Group([sym_internals[name] for name in outputs])

    context = mxnet.cpu()
    module = mxnet.mod.Module(symbol=output_group, context=context)
    module.bind(for_training=False, data_shapes=[(inputs, data_shape)])
    module.set_params(arg_params=arg_params, aux_params=aux_params)

    print("[INFO] Graph setup.")
    print("[INFO] Graph forwarding...")

    data_iter.reset()
    this_data = data_iter.next()
    module.forward(this_data, is_train=False)

    output_features = module.get_outputs()

    print("[INFO] ====== Saving outputs ======")
    for i in range(len(outputs)):
        name = outputs[i]
        output_feature = output_features[i].asnumpy()

        tag_name = name
        if tag_name.endswith('_output'):
            tag_name = name[:-7]

        print("[INFO] Saving {}: {}".format(tag_name, output_feature.shape))
        ts.tensor.write(os.path.join(output_root, "{}.t".format(tag_name)), output_feature)
