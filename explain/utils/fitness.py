import torch


def predict_batch_images(model, input_batch):
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    return torch.nn.functional.softmax(output, dim=1)
