#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uuid
import shutil
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import cv2

# Load MiDaS model and transforms once
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def predict_depth_from_file(image_path: str, colormap: str = 'plasma') -> str:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    input_batch = transform(img_np).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized = (depth_map - depth_min) / (depth_max - depth_min)

    cmap = plt.get_cmap(colormap)
    depth_vis = cmap(normalized)[:, :, :3]
    depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)

    out_path = f"depth_map_{uuid.uuid4().hex[:8]}.png"
    Image.fromarray(depth_vis_uint8).save(out_path)

    return out_path

# FastAPI setup
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"upload_{uuid.uuid4().hex[:8]}.{file_ext}"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_path = predict_depth_from_file(temp_filename)
    os.remove(temp_filename)
    return FileResponse(output_path)

# Serve the HTML front-end
from fastapi.responses import HTMLResponse
from fastapi import HTTPException

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")
    except Exception as e:
        print("Error reading index.html:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
