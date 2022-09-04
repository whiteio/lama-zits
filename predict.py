from helper import ceil_modulo, resize_max_size


import multiprocessing
import os
import imghdr
import torch
from PIL import Image
import numpy as np
import cv2

from model_manager import ModelManager
from helper import norm_img
from schema import Config, HDStrategy, LDMSampler
from zits import ZITS

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

interpolation = cv2.INTER_CUBIC

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

    # model = ZITS(device=device)
    model = ModelManager(name='zits', device=device)
    # RGB
    # with open("unwant_object_clean.jpg") as image:
    #     origin_image_bytes = image.read()

    #    hdStrategy: HDStrategy.CROP,
    # hdStrategyResizeLimit: 1024,
    # hdStrategyCropTrigerSize: 1024,
    # hdStrategyCropMargin: 128,

    config = Config(
        ldm_steps=25,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=HDStrategy.CROP,
        zits_wireframe=True,
        hd_strategy_crop_margin=128,
        hd_strategy_crop_trigger_size=1024,
        hd_strategy_resize_limit=1024,
    )


    image = np.random.randint(0, 256, (512, 512, 3)).astype(np.uint8)
    image = resize_max_size(image, size_limit=1080, interpolation=interpolation)
    # image = norm_img(image)

    mask = np.random.randint(0, 255, (512, 512)).astype(np.uint8)
    mask = resize_max_size(mask, size_limit=1080, interpolation=interpolation)
    res_np_img = model(image, mask, config)
    print(res_np_img)
    print('Ran')


run()