#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'stylegan3')

TITLE = 'StyleGAN3 Food Image Generation'

MODEL_REPO = 'hysts/stylegan3-food101-model'
MODEL_FILE_NAME = '010000.pkl'

HF_TOKEN = os.getenv('HF_TOKEN')


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


def generate_z(seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(1,
                                                              512)).to(device)


@torch.inference_mode()
def generate_image(seed: int, truncation_psi: float, tx: float, ty: float,
                   angle: float, model: nn.Module,
                   device: torch.device) -> np.ndarray:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
    z = generate_z(seed, device)
    c = torch.zeros(0).to(device)

    mat = make_transform((tx, ty), angle)
    mat = np.linalg.inv(mat)
    model.synthesis.input.transform.copy_(torch.from_numpy(mat))

    out = model(z, c, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(device: torch.device) -> nn.Module:
    path = hf_hub_download(MODEL_REPO,
                           MODEL_FILE_NAME,
                           use_auth_token=HF_TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, 512)).to(device)
        c = torch.zeros(0).to(device)
        model(z, c)
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
func = functools.partial(generate_image, model=model, device=device)

gr.Interface(
    fn=func,
    inputs=[
        gr.Slider(label='Seed',
                  minimum=0,
                  maximum=10000000000,
                  step=1,
                  value=1424059097),
        gr.Slider(label='Truncation psi',
                  minimum=0,
                  maximum=2,
                  step=0.05,
                  value=0.7),
        gr.Slider(label='Translate X',
                  minimum=-1,
                  maximum=1,
                  step=0.05,
                  value=0),
        gr.Slider(label='Translate Y',
                  minimum=-1,
                  maximum=1,
                  step=0.05,
                  value=0),
        gr.Slider(label='Angle', minimum=-180, maximum=180, step=5, value=0),
    ],
    outputs=gr.Image(label='Output', type='numpy'),
    title=TITLE,
).queue().launch(show_api=False)
