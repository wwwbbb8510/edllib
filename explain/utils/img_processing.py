from imagenet_stubs.imagenet_2012_labels import label_to_name as imagenet_label_to_name
from PIL import Image
from torchvision import transforms
import torch
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np


def get_imagenet_preprocess():
    imagenet_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return imagenet_preprocess


def imagenet_class_to_text_label(image_classes):
    image_classes_labels = map(
        lambda label: {label: imagenet_label_to_name(label)},
        image_classes
    )

    return list(image_classes_labels)


def load_image_from_imagenet(filename):
    input_image = Image.open(filename)
    return input_image


def load_image_as_numpy_array(filename):
    orig_image = io.imread(filename)
    float_image = img_as_float(orig_image)
    return orig_image, float_image


def slic_segment(float_image, ns=100, sigma=5):
    segments = slic(float_image, n_segments=ns, sigma=sigma)
    return segments


def mask_image_by_blackout_superpixels(orig_image, segments, mask_array):
    mask_segment_indexes = [i for i, v in enumerate(mask_array) if v == 1]
    mask_segments = np.ones(segments.shape)
    for x in mask_segment_indexes:
        mask_segments = np.logical_and(mask_segments, segments != x)
    mask_segments.astype(int)
    mask_segments = mask_segments.reshape(mask_segments.shape + (1,))
    masked_image = Image.fromarray(np.multiply(orig_image, mask_segments))
    return masked_image


def mask_image_by_batch_blackout_superpixels(orig_image, segments, mask_batch_array):
    batch_images = [mask_image_by_blackout_superpixels(orig_image, segments, mask_array)
                    for mask_array in mask_batch_array]
    return batch_images


def transform_batch_images_to_tensor(batch_images):
    input_batch = [get_imagenet_preprocess()(img) for img in batch_images]
    return torch.stack(input_batch)
