import io
from helper import ceil_modulo, load_img, numpy_to_bytes, resize_max_size


import multiprocessing
import os
import imghdr
import torch
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, send_file, make_response

from model_manager import ModelManager
from helper import norm_img
from schema import Config, HDStrategy, LDMSampler
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

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Running model on: {device}")
model = ModelManager(name='lama', device=device)
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

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    origin_image_bytes = input["image"].read()
    mask_image_bytes = input["mask"].read()

    config = Config(
        ldm_steps=25,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=HDStrategy.RESIZE,
        zits_wireframe=False,
        hd_strategy_crop_margin=128,
        hd_strategy_crop_trigger_size=2048,
        hd_strategy_resize_limit=2048,
    )


    mask, _ = load_img(mask_image_bytes, gray=True)
    image, alpha_channel = load_img(origin_image_bytes, )
    print(image)

    interpolation = cv2.INTER_CUBIC
    original_shape = image.shape
    size_limit = max(image.shape)

    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    res_np_img = model(image, mask, config)

    torch.cuda.empty_cache()

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    response = make_response(
        send_file(
            io.BytesIO(numpy_to_bytes(res_np_img, ext)),
            mimetype=f"image/{ext}",
        )
    )
    return response
if __name__ == '__main__':
    app.run()
