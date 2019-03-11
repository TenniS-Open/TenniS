import torch


def convert(input_module, output_file, input, verbose=False):
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

    # summary(torch_model, (3, 224, 224))
    #
    # ts_node = convert_module.convert_module(torch_model)

    dummy_input = torch.autograd.Variable(torch.randn(input))
    torch.onnx.export(torch_model, dummy_input, output_file, verbose=verbose)

    print("============ Summary ============")
    # print("Input file: {}".format(input_module))
    print("Output file: {}".format(output_file))
