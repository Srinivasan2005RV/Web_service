"""
PCB Defect Detection API
FastAPI-based cloud-ready detection service
Accepts image input and returns detection results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os

# Load the best performing model
MODEL_PATH = "runs/pcb_detect/pcb_custom_v1_continued/weights/best.pt"
model = None

def load_model():
    """Load the YOLO model"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"Model not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    try:
        load_model()
        print("🚀 PCB Detection API is ready!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
    yield
    # Cleanup code here if needed

# Initialize FastAPI app
app = FastAPI(
    title="PCB Defect Detection API",
    description="AI-powered PCB defect detection service",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "PCB Defect Detection API",
        "version": "1.0.0",
        "model": "YOLO11n",
        "endpoints": {
            "detect": "/detect",
            "detect_json": "/detect/json",
            "detect_image": "/detect/image",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Main detection endpoint - returns both JSON results and annotated image
    
    Args:
        file: Image file (JPG, PNG)
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
    
    Returns:
        JSON with detection results and base64 encoded annotated image
    """
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Extract detection data
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i].item()),
                    "class_name": model.names[int(boxes.cls[i].item())],
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": {
                        "x1": float(boxes.xyxy[i][0].item()),
                        "y1": float(boxes.xyxy[i][1].item()),
                        "x2": float(boxes.xyxy[i][2].item()),
                        "y2": float(boxes.xyxy[i][3].item())
                    }
                }
                detections.append(detection)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Convert to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": img.shape[1],
                "height": img.shape[0],
                "filename": file.filename
            },
            "detection_summary": {
                "total_defects": len(detections),
                "defect_types": list(set([d["class_name"] for d in detections]))
            },
            "detections": detections,
            "annotated_image_base64": img_base64,
            "parameters": {
                "confidence_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/detect/json")
async def detect_json_only(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detection endpoint - returns only JSON results (no image)
    Faster response, suitable for applications that don't need the annotated image
    """
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Extract detection data
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i].item()),
                    "class_name": model.names[int(boxes.cls[i].item())],
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": {
                        "x1": float(boxes.xyxy[i][0].item()),
                        "y1": float(boxes.xyxy[i][1].item()),
                        "x2": float(boxes.xyxy[i][2].item()),
                        "y2": float(boxes.xyxy[i][3].item())
                    }
                }
                detections.append(detection)
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": img.shape[1],
                "height": img.shape[0],
                "filename": file.filename
            },
            "detection_summary": {
                "total_defects": len(detections),
                "defect_types": list(set([d["class_name"] for d in detections]))
            },
            "detections": detections
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/detect/image")
async def detect_image_only(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detection endpoint - returns only annotated image
    Suitable for direct display in web applications
    """
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Convert to bytes for image response
        _, buffer = cv2.imencode('.jpg', annotated_img)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=detected_{file.filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PCB Defect Detection API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run("pcb_detection_api:app", host="0.0.0.0", port=8000, reload=False)
