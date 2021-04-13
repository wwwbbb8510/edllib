import torch


def load_torch_pre_trained_model(model_name, sub_type=None):
    """
    @param model_name: the name of the model to load. e.g. densenet
    @param sub_type: the sub type of the model to load. e.g. 121
    @return: pre-trained model
    """
    model = None
    if model_name == 'inception_v3':
        print('inception_v3')
        model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    elif model_name == 'mobilenet_v2':
        print('mobilenet_v2')
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    elif model_name == 'shufflenet_v2':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
    elif model_name == 'squeezenet1':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_' + sub_type, pretrained=True)
    elif model_name == 'densenet':
        print('densenet' + sub_type)
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet' + sub_type, pretrained=True)
    elif model_name == 'resnet':
        print('resnet' + sub_type)
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet' + sub_type, pretrained=True)
    elif model_name == 'vgg':
        print('vgg' + sub_type)
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg' + sub_type, pretrained=True)

    model.eval()
    return model
