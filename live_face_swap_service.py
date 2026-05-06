"""
Live Face Swap inference service for GenFlow.

Run this on the same GPU pod that hosts ComfyUI, or any machine with a CUDA GPU.

Setup:
    pip install fastapi uvicorn[standard] python-multipart insightface onnxruntime-gpu opencv-python-headless numpy pillow

Run:
    python live_face_swap_service.py --port 8189

Then in GenFlow Settings, set "Live Face Swap URL" to:
    https://<your-runpod-host>-8189.proxy.runpod.net

This serves two endpoints:
    POST /set_source       — multipart "image" → caches the source face embedding
    POST /swap             — multipart "image" → returns JPEG bytes with face swapped
    GET  /healthz
"""
import argparse
import io
import os
import time
import urllib.request
from typing import Optional

import cv2
import insightface
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

app = FastAPI(title="GenFlow Live Face Swap")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- model loading (once at startup) -----------------------------------------
print("[live-swap] loading InsightFace buffalo_l detector...")
analyzer = FaceAnalysis(name="buffalo_l")
# 320x320 detector cuts detection time roughly in half versus 640x640.
# Plenty of resolution for webcam-sized frames.
analyzer.prepare(ctx_id=0, det_size=(320, 320))

# inswapper_128 — the library's auto-downloader looks for a .zip that no longer
# exists. Fetch the raw .onnx directly into the expected cache location.
_INSWAP_DIR = os.path.expanduser("~/.insightface/models")
_INSWAP_PATH = os.path.join(_INSWAP_DIR, "inswapper_128.onnx")
_INSWAP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
os.makedirs(_INSWAP_DIR, exist_ok=True)
if not os.path.exists(_INSWAP_PATH) or os.path.getsize(_INSWAP_PATH) < 100_000_000:
    print(f"[live-swap] downloading inswapper_128.onnx (~554MB) → {_INSWAP_PATH}")
    urllib.request.urlretrieve(_INSWAP_URL, _INSWAP_PATH)
    print("[live-swap] download complete")

print("[live-swap] loading inswapper_128.onnx ...")
swapper = insightface.model_zoo.get_model(_INSWAP_PATH)

source_face = None  # cached after /set_source


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "JPEG encode failed")
    return buf.tobytes()


@app.get("/healthz")
def healthz():
    return {"ok": True, "source_loaded": source_face is not None}


@app.post("/set_source")
async def set_source(image: UploadFile = File(...)):
    global source_face
    img = _decode(await image.read())
    faces = analyzer.get(img)
    if not faces:
        raise HTTPException(400, "No face detected in source image")
    # use the largest face
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    source_face = faces[0]
    return {"ok": True, "bbox": [int(v) for v in source_face.bbox.tolist()]}


def _expand_swap_chin(orig: np.ndarray, swapped: np.ndarray, face) -> np.ndarray:
    """Aggressively cover beard/chin by painting the lower face with clean
    skin tone sampled from the swapped cheeks.

    Inswapper only regenerates the central face oval, leaving the beard intact
    on the chin/jaw. A plain inpaint averages nearby pixels — which still
    contain beard hair — so the result looks "softly blurred beard". Instead
    we grab the clean skin color from the upper-cheek area (which inswapper
    DID paint) and paint the beard region with that tone, then re-apply
    shading from the swapped image to keep natural lighting.
    """
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        kps = face.kps
        l_eye, r_eye, nose, l_mouth, r_mouth = kps

        H, W = orig.shape[:2]

        # 1. Lower-face mask: large ellipse from just above the mouth to past
        # the chin. Pad wider than the bbox so we catch sideburn area.
        mouth_y = int((l_mouth[1] + r_mouth[1]) / 2)
        cx = int((x1 + x2) / 2)
        # Start the ellipse at the mouth line, extend down past bottom of bbox
        cy = int((mouth_y + y2) / 2)
        w = int((x2 - x1) * 0.55)
        h = int((y2 - mouth_y) * 1.15)
        if w < 5 or h < 5:
            return swapped
        chin_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(chin_mask, (cx, cy), (w, h), 0, 0, 360, 255, -1)

        # 2. Find pixels untouched by the swapper (beard zone).
        diff = cv2.absdiff(swapped, orig).max(axis=2)
        untouched = (diff < 10).astype(np.uint8) * 255
        beard_mask = cv2.bitwise_and(chin_mask, untouched)

        # Also aggressively include any dark pixels inside the chin ellipse
        # (dark = likely beard/stubble), even if the swapper kinda-touched them.
        gray = cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY)
        dark = ((gray < 90).astype(np.uint8) * 255)
        dark_in_chin = cv2.bitwise_and(chin_mask, dark)
        beard_mask = cv2.bitwise_or(beard_mask, dark_in_chin)

        # Dilate generously to cover beard edges that fade into skin
        beard_mask = cv2.dilate(beard_mask, np.ones((9, 9), np.uint8))

        if cv2.countNonZero(beard_mask) < 100:
            return swapped

        # 3. Sample CLEAN skin tone from upper-cheek patches (areas inswapper
        # painted fresh). These sit just under the eyes, above the mouth.
        def sample_patch(pt, half=6):
            px, py = int(pt[0]), int(pt[1])
            x0 = max(0, px - half); x1_ = min(W, px + half)
            y0 = max(0, py - half); y1_ = min(H, py + half)
            patch = swapped[y0:y1_, x0:x1_]
            if patch.size == 0:
                return None
            # Use median for robustness against outliers
            return np.median(patch.reshape(-1, 3), axis=0)

        # Cheek sample points: below each eye, offset toward the nose
        left_cheek = sample_patch(((l_eye[0] + nose[0]) / 2, (l_eye[1] + nose[1]) / 2 + 6))
        right_cheek = sample_patch(((r_eye[0] + nose[0]) / 2, (r_eye[1] + nose[1]) / 2 + 6))
        samples = [s for s in (left_cheek, right_cheek) if s is not None]
        if not samples:
            return swapped
        skin_color = np.median(np.stack(samples, axis=0), axis=0)  # BGR

        # 4. Paint the beard zone with that skin color, then restore shading.
        # Build a solid-color layer of the skin tone.
        skin_layer = np.full_like(swapped, skin_color.astype(np.uint8))

        # Preserve the swapped image's luminance (shading) so it's not flat.
        # Blend: skin_layer provides chroma, swapped provides structure.
        swapped_blur = cv2.GaussianBlur(swapped, (15, 15), 0)
        swapped_gray = cv2.cvtColor(swapped_blur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Normalize luminance into a 0.85..1.15 multiplier for gentle shading
        lum = (swapped_gray / swapped_gray.mean()).clip(0.85, 1.15)[:, :, None]
        shaded = (skin_layer.astype(np.float32) * lum).clip(0, 255).astype(np.uint8)

        # 5. Feather the mask and composite shaded skin over swapped result.
        feather = cv2.GaussianBlur(beard_mask, (41, 41), 0).astype(np.float32) / 255.0
        feather = feather[:, :, None]
        result = shaded.astype(np.float32) * feather + swapped.astype(np.float32) * (1 - feather)
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[live-swap] chin-expand failed: {e}")
        return swapped


@app.post("/swap")
async def swap(image: UploadFile = File(...), cover_beard: int = 0):
    if source_face is None:
        raise HTTPException(400, "Source face not set. POST /set_source first.")
    t0 = time.time()
    img = _decode(await image.read())

    faces = analyzer.get(img)
    if not faces:
        # No face in this frame — return original so the stream stays smooth.
        return Response(content=_encode_jpeg(img), media_type="image/jpeg",
                        headers={"X-Latency-Ms": f"{int((time.time() - t0) * 1000)}", "X-Faces": "0"})

    res = img.copy()
    for face in faces:
        swapped = swapper.get(res, face, source_face, paste_back=True)
        if cover_beard:
            res = _expand_swap_chin(res, swapped, face)
        else:
            res = swapped

    out = _encode_jpeg(res)
    return Response(content=out, media_type="image/jpeg",
                    headers={"X-Latency-Ms": f"{int((time.time() - t0) * 1000)}", "X-Faces": str(len(faces))})


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8189)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
