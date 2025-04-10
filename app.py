#!/usr/bin/env python
# coding: utf-8

import asyncio
import uuid
import shutil
import os
import time

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- Configuration ---
# Use the standard /tmp directory which is typically writable in containers
UPLOAD_DIR = "/tmp/temp_uploads"
OUTPUT_DIR = "/tmp/temp_outputs"
# --- END CHANGE ---

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose MiDaS model type:
# Options: "MiDaS_small", "DPT_Hybrid", "DPT_Large" (Highest quality)
MODEL_TYPE = "DPT_Hybrid"

# --- Model Loading ---
print("Loading MiDaS model...")
start_load_time = time.time()
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    midas_model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if MODEL_TYPE == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    midas_model.to(device)
    midas_model.eval()
    load_time = time.time() - start_load_time
    print(f"MiDaS model '{MODEL_TYPE}' loaded successfully in {load_time:.2f} seconds.")

except Exception as e:
    print(f"FATAL ERROR: Could not load MiDaS model '{MODEL_TYPE}' from torch.hub.")
    print(f"Error details: {e}")
    print("Please ensure internet connectivity and correct MODEL_TYPE.")
    import sys
    sys.exit(1)

# --- Helper Function: Depth Prediction ---
def predict_depth(image_path: str, output_dir: str, colormap: str = 'plasma') -> str:
    """
    Reads image, predicts depth, saves visualization with decorations, returns output path.
    """
    try:
        print(f"Processing image: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image file: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            start_pred_time = time.time()
            prediction = midas_model(input_batch)
            pred_time = time.time() - start_pred_time
            print(f"Inference time: {pred_time:.3f} seconds")

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            normalized_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized_map = np.zeros_like(depth_map)

        # --- Save Visualization ---
        output_filename = f"depth_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(output_dir, output_filename)

        plt.figure(figsize=(10, 6))
        plt.imshow(normalized_map, cmap=colormap)
        plt.title(f"Estimated Depth Map ({MODEL_TYPE})")
        cbar = plt.colorbar()
        cbar.set_label("Relative Inverse Depth (Higher Value = Closer)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Saved depth map visualization to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during depth prediction for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        raise

# --- FastAPI Application Setup ---
app = FastAPI(title="MiDaS Depth Estimation API")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main HTML frontend."""
    html_file_path = "index.html"
    if not os.path.exists(html_file_path):
        return HTMLResponse(content="<html><body><h1>Error</h1><p>Frontend file 'index.html' not found.</p></body></html>", status_code=404)
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        print(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error reading frontend file.")

@app.post("/predict/", response_class=FileResponse)
async def create_prediction(
    file: UploadFile = File(...),
    colormap: str = Query("plasma", enum=["viridis", "plasma", "magma", "inferno", "cividis", "gray"])
):
    """
    Accepts an uploaded image, predicts depth using the specified colormap,
    and returns the depth map image.
    """
    temp_filepath = None
    output_path = None

    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'. Allowed types: {allowed_extensions}")

    try:
        # --- CORRECT LOGIC BLOCK ---
        temp_filename = f"upload_{uuid.uuid4().hex[:8]}.{file_ext}"
        temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

        start_save_time = time.time()
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        save_time = time.time() - start_save_time
        print(f"Uploaded file saved to: {temp_filepath} in {save_time:.3f} seconds")

        print(f"Using colormap: {colormap}")
        output_path = predict_depth(temp_filepath, OUTPUT_DIR, colormap=colormap)

        return FileResponse(
            output_path,
            media_type='image/png',
            filename=f"depth_{colormap}_{file.filename}"
        )
        # --- END CORRECT LOGIC BLOCK ---

    # REMOVED THE DUPLICATE TRY BLOCK THAT WAS HERE

    except ValueError as ve:
         print(f"Value Error processing file: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred during prediction for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

    finally:
        # --- Cleanup ---
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print(f"Removed temporary upload file: {temp_filepath}")
            except Exception as e_rem_in:
                 print(f"Warning: Error removing temporary input file {temp_filepath}: {e_rem_in}")

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    import os # Import os to potentially read PORT

    print("Starting FastAPI server for LOCAL DEVELOPMENT...")

    # Use environment variable for port if available (good practice), otherwise default
    port = int(os.environ.get("PORT", 8000))
    # For local testing, 127.0.0.1 is usually fine.
    # For testing container networking, you might use "0.0.0.0" here too.
    host = "127.0.0.1"

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True # Keep reload=True for local development ease
        # log_level="info" # Optional: Set log level
    )

    # NOTE FOR DEPLOYMENT:
    # Production servers (like Gunicorn with Uvicorn workers) are typically used
    # in deployment and configured separately (e.g., via Procfile or platform settings).
    # They usually bind to host="0.0.0.0" and use a $PORT environment variable.
    # This __main__ block is NOT executed by those servers.
