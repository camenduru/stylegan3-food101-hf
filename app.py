#!/usr/bin/env python

from __future__ import annotations

import argparse
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
DESCRIPTION = 'Expected execution time on Hugging Face Spaces: 20s'
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.stylegan3-food101" alt="visitor badge"/></center>'

MODEL_REPO = 'hysts/stylegan3-food101-model'
MODEL_FILE_NAME = '010000.pkl'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


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
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)
    func = functools.partial(generate_image, model=model, device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=1424059097, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate X'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate Y'),
            gr.inputs.Slider(-180, 180, step=5, default=0, label='Angle'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging='never',
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
