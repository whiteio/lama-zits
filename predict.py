import io
from helper import ceil_modulo, load_img, numpy_to_bytes, resize_max_size


import multiprocessing
import os
import imghdr
import torch
from PIL import Image
import numpy as np
import cv2

from model_manager import ModelManager
from helper import norm_img
from schema import HDStrategy, LDMSampler
from lama import LaMa

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w

model: ModelManager = None
device = None
input_image_path: str = None

def current_model():
    return model.name

def model_downloaded(name):
    return str(model.is_downloaded(name))   

def set_input_photo():
    if input_image_path:
        with open(input_image_path, "rb") as f:
            image_in_bytes = f.read()
            return image_in_bytes
    else:
        return "No Input Image" 


def run():
    device = torch.device('cpu')

    model = ModelManager(name='lama', device=device)
    f = open('dog_photo.png', 'rb') 
    m = open('masker_image.png', 'rb')

    input_mask_bytes = m.read()
    input_image_bytes = f.read()

    mask, _ = load_img(input_mask_bytes, gray=True)
    image, alpha_channel = load_img(input_image_bytes, )
    print(image)

    interpolation = cv2.INTER_CUBIC
    original_shape = image.shape
    size_limit = max(image.shape)

    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
        
    res_np_img = model(image, mask)

    torch.cuda.empty_cache()

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(input_image_bytes)
    resulting_bytes = numpy_to_bytes(res_np_img, ext)

    final_image = Image.open(io.BytesIO(resulting_bytes))
    final_image.save('resultant_image.{}'.format(ext))
run()