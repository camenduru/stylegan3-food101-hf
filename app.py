#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pickle
import sys

sys.path.insert(0, 'stylegan3')

import gradio as gr
import numpy as np
import PIL.Image
import torch
from huggingface_hub import hf_hub_download

MODEL_REPO = 'hysts/stylegan3-food101-model'
MODEL_FILE_NAME = '010000.pkl'
TOKEN = os.environ['TOKEN']

DEFAULT_SEED = 1424059097

TITLE = 'StyleGAN3 Food Image Generation'


def make_transform(translate: tuple[float, float], angle: float) -> np.ndarray:
    mat = np.eye(3)
    sin = np.sin(angle / 360 * np.pi * 2)
    cos = np.cos(angle / 360 * np.pi * 2)
    mat[0][0] = cos
    mat[0][1] = sin
    mat[0][2] = translate[0]
    mat[1][0] = -sin
    mat[1][1] = cos
    mat[1][2] = translate[1]
    return mat


def generate_z(seed, device):
    return torch.from_numpy(np.random.RandomState(seed).randn(1,
                                                              512)).to(device)


@torch.inference_mode()
def generate_image(seed, truncation_psi, tx, ty, angle, model, device):
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
    z = generate_z(seed, device)
    c = torch.zeros(0).to(device)

    mat = make_transform((tx, ty), angle)
    mat = np.linalg.inv(mat)
    model.synthesis.input.transform.copy_(torch.from_numpy(mat))

    out = model(z, c, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(out[0].cpu().numpy(), 'RGB')


def load_model(device):
    path = hf_hub_download(MODEL_REPO, MODEL_FILE_NAME, use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, 512)).to(device)
        c = torch.zeros(0).to(device)
        model(z, c)
    return model


def main():
    device = torch.device('cpu')

    model = load_model(device)
    func = functools.partial(generate_image, model=model, device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=DEFAULT_SEED, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate X'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate Y'),
            gr.inputs.Slider(-180, 180, step=5, default=0, label='Angle'),
        ],
        gr.outputs.Image(type='pil', label='Output'),
        title=TITLE,
        enable_queue=True,
        allow_screenshot=False,
        allow_flagging=False,
    ).launch()


if __name__ == '__main__':
    main()
