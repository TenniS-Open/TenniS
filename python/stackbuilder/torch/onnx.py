import torch


def convert(input_module, output_file, input, verbose=False):
    """
    :param input_module:
    :param output_file:
    :param input: tuple of input shape
    :param verbose:
    :return:
    """
    torch_model = None
    if isinstance(input_module, str):
        torch_model = torch.load(input_module)
    elif isinstance(input_module, torch.nn.Module):
        torch_model = input_module
    if torch_model is None:
        raise NotImplementedError("Not supported model: {}".format(type(input_module)))
    for param in torch_model.parameters():
        param.requires_grad = False
    torch_model.eval()

    for i in range(len(input)):
        node = input[i]
        if isinstance(node, (tuple, list)):
            for i in node:
                if not isinstance(i, (int, )):
                    raise RuntimeError("input must be a list of tuple[int]")
        else:
            raise RuntimeError("input must be a list of tuple[int]")

    dummy_input = [torch.autograd.Variable(torch.randn(o)) for o in input]
    assert len(dummy_input) == 1
    dummy_input = dummy_input[0]
    torch.onnx.export(torch_model, dummy_input, output_file, verbose=verbose,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      export_params=True, keep_initializers_as_inputs=True)

    print("============ Summary ============")
    if isinstance(input_module, str):
        print("Input file: {}".format(input_module))
    elif isinstance(input_module, torch.nn.Module):
        print("Input file is memory module")
    print("Output file: {}".format(output_file))
