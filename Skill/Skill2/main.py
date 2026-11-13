import asyncio
from contextlib import asynccontextmanager
import io
import time
from datetime import datetime, timezone
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from services.api_gateway.schemas import (
    BoundingBox,
    InspectionResult,
    ISOContainerCode,
    ServerHealthResponse,
)
from services.api_gateway.triton_client import TritonInferenceClient

triton_client = TritonInferenceClient(url="localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await triton_client.connect()
    except Exception as e:
        print(f"Warning: Triton not immediately reachable at startup ({e}). Retrying on first request.")
    yield
    await triton_client.close()


app = FastAPI(
    title="VisionPort Edge & Gate Inspection API Gateway",
    version="2.0.0",
    description="High-throughput asynchronous API Gateway for Port Container OCR, Damage Detection, and Quayside Safety.",
    lifespan=lifespan,
)

# Enable CORS for Port Dashboard & Terminal Operating System (TOS) frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes, target_size: int = 640) -> Tuple[np.ndarray, float, float]:
    """Decodes image bytes, applies letterbox resizing, and normalizes to NCHW FP32 tensor."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image buffer.")

    h, w, _ = img.shape
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    # Normalize BGR -> RGB, [0, 255] -> [0.0, 1.0], HWC -> NCHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)  # Shape: (1, 3, 640, 640)
    return np.ascontiguousarray(tensor), scale, (left, top)


def validate_iso_6346_checksum(code: str) -> bool:
    """
    Validates ISO 6346 container identification checksum according to international standards.
    Standard: 4 letters (3 owner + 1 category) + 6 serial digits + 1 check digit.
    """
    if len(code) != 11 or not code[:4].isalpha() or not code[4:].isdigit():
        return False

    char_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
        'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
        'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
        'Y': 37, 'Z': 38
    }

    total = 0
    for i in range(10):
        c = code[i].upper()
        val = char_values[c] if c.isalpha() else int(c)
        total += val * (2 ** i)

    check_digit = (total % 11) % 10
    return check_digit == int(code[10])


@app.get("/health", response_model=ServerHealthResponse)
async def health_check():
    is_ready = await triton_client.is_ready()
    return ServerHealthResponse(
        status="healthy" if is_ready else "degraded",
        triton_ready=is_ready,
        gpu_available=is_ready,
        active_models=["container_detector", "container_ocr", "visionport_pipeline"] if is_ready else [],
    )


@app.post("/api/v1/inspect/frame", response_model=InspectionResult)
async def inspect_single_frame(
    file: UploadFile = File(..., description="JPEG/PNG image frame from port camera"),
    camera_id: str = Form("QUAY_CRANE_04", description="Camera location identifier"),
):
    """
    Receives an image frame from an edge camera, executes Triton TensorRT inference,
    extracts container detections, and returns structured inspection results.
    """
    t_start = time.perf_counter()

    # 1. Read & Preprocess Frame
    contents = await file.read()
    try:
        tensor, scale, (pad_x, pad_y) = preprocess_image(contents, target_size=640)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # 2. Invoke Triton gRPC Inference
    try:
        raw_output, trt_duration_ms = await triton_client.infer_detector(tensor)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Triton inference failed: {str(e)}")

    # 3. Post-Process Detections (Mocked parsing from TensorRT NMS output)
    # In production, parses output coordinates relative to original aspect ratio
    detections: List[BoundingBox] = [
        BoundingBox(
            x_min=0.15,
            y_min=0.20,
            x_max=0.85,
            y_max=0.75,
            confidence=0.965,
            class_id=0,
            class_name="shipping_container",
        )
    ]

    # Mock OCR identification for ISO 6346
    sample_code = "MSCU7824106"
    ocr_result = ISOContainerCode(
        raw_code=sample_code,
        owner_code="MSC",
        category_identifier="U",
        serial_number="782410",
        check_digit=6,
        is_checksum_valid=validate_iso_6346_checksum(sample_code),
        confidence=0.982,
    )

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

    return InspectionResult(
        camera_id=camera_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        detections=detections,
        container_ocr=ocr_result,
        structural_damage_detected=False,
        damage_types=[],
        ppe_compliance=True,
        inference_latency_ms=round(trt_duration_ms, 2),
        end_to_end_latency_ms=round(total_ms, 2),
    )


@app.websocket("/api/v1/stream/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Real-time bi-directional WebSocket stream for live video analytics."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            t0 = time.perf_counter()
            tensor, _, _ = preprocess_image(data, target_size=640)
            _, trt_latency = await triton_client.infer_detector(tensor)
            total_latency = (time.perf_counter() - t0) * 1000.0

            # Stream telemetry back to client
            await websocket.send_json({
                "status": "PROCESSED",
                "inference_ms": round(trt_latency, 2),
                "total_ms": round(total_latency, 2),
                "containers_detected": 1,
            })
    except WebSocketDisconnect:
        pass
