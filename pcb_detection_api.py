"""
PCB Defect Detection API
FastAPI-based cloud-ready detection service
Accepts image input and returns detection results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
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
import json
from pathlib import Path

# Load the best performing model
MODEL_PATH = "runs/pcb_detect/pcb_custom_v1_continued/weights/best.pt"
model = None

# Storage for latest result
LATEST_RESULT_FILE = "latest_detection.json"
LATEST_IMAGE_FILE = "latest_image.jpg"

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
            "latest": "/latest - View latest detection result",
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
        
        # Save latest result for web view
        try:
            latest_data = {
                "success": True,
                "total_defects": len(detections),
                "defect_types": list(set([d["class_name"] for d in detections])),
                "confidence_threshold": conf_threshold,
                "timestamp": datetime.now().isoformat()
            }
            with open(LATEST_RESULT_FILE, 'w') as f:
                json.dump(latest_data, f, indent=2)
            
            # Save original image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_img.save(LATEST_IMAGE_FILE)
        except Exception as e:
            print(f"Warning: Could not save latest result: {e}")
        
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

@app.get("/latest", response_class=HTMLResponse)
async def get_latest_result():
    """Display the latest detection result in a web page"""
    
    # Check if result file exists
    if not Path(LATEST_RESULT_FILE).exists():
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>PCB Detection - Latest Result</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        text-align: center; 
                        padding: 50px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        min-height: 100vh;
                        margin: 0;
                    }
                    .message { 
                        font-size: 20px;
                        background: rgba(255,255,255,0.1);
                        padding: 30px;
                        border-radius: 15px;
                        max-width: 600px;
                        margin: 0 auto;
                    }
                </style>
            </head>
            <body>
                <h1>🔍 PCB Defect Detection</h1>
                <div class="message">
                    <p>⏳ No detection results yet</p>
                    <p>Waiting for ESP32-CAM...</p>
                </div>
            </body>
        </html>
        """
    
    # Load latest result
    try:
        with open(LATEST_RESULT_FILE, 'r') as f:
            result = json.load(f)
        
        # Convert image to base64
        image_base64 = ""
        if Path(LATEST_IMAGE_FILE).exists():
            with open(LATEST_IMAGE_FILE, 'rb') as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode()
        
        # Generate HTML
        defect_status = "✓ PCB is GOOD - No Defects!" if result['total_defects'] == 0 else f"⚠️ {result['total_defects']} DEFECT(S) FOUND"
        status_color = "#28a745" if result['total_defects'] == 0 else "#dc3545"
        
        defect_types_html = ""
        if result['defect_types']:
            defect_types_html = f"<p><strong>Defect Types:</strong> {', '.join(result['defect_types'])}</p>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>PCB Detection - Latest Result</title>
                <meta http-equiv="refresh" content="5">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1000px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        padding: 30px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    }}
                    h1 {{
                        color: #333;
                        text-align: center;
                        margin-bottom: 30px;
                        font-size: 32px;
                    }}
                    .status {{
                        text-align: center;
                        font-size: 28px;
                        font-weight: bold;
                        color: {status_color};
                        margin: 30px 0;
                        padding: 20px;
                        background: rgba(0,0,0,0.05);
                        border-radius: 15px;
                        border: 3px solid {status_color};
                    }}
                    .info {{
                        background: #f8f9fa;
                        padding: 25px;
                        border-radius: 15px;
                        margin: 25px 0;
                    }}
                    .info p {{
                        margin: 12px 0;
                        font-size: 18px;
                        color: #495057;
                    }}
                    .info strong {{
                        color: #212529;
                    }}
                    .image-container {{
                        text-align: center;
                        margin: 30px 0;
                    }}
                    .image-container h3 {{
                        margin-bottom: 20px;
                        color: #495057;
                    }}
                    .image-container img {{
                        max-width: 100%;
                        height: auto;
                        border: 4px solid #dee2e6;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    }}
                    .timestamp {{
                        text-align: center;
                        color: #6c757d;
                        font-size: 14px;
                        margin-top: 25px;
                        padding: 15px;
                        background: #f8f9fa;
                        border-radius: 10px;
                    }}
                    .refresh {{
                        text-align: center;
                        color: #6c757d;
                        font-size: 13px;
                        margin-top: 15px;
                        font-style: italic;
                    }}
                    .badge {{
                        display: inline-block;
                        padding: 8px 16px;
                        background: {status_color};
                        color: white;
                        border-radius: 20px;
                        font-size: 16px;
                        margin: 10px 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔍 PCB Defect Detection System</h1>
                    
                    <div class="status">
                        {defect_status}
                    </div>
                    
                    <div class="info">
                        <p><strong>📊 Total Defects Detected:</strong> <span class="badge">{result['total_defects']}</span></p>
                        {defect_types_html}
                        <p><strong>🎯 Confidence Threshold:</strong> {result['confidence_threshold']}</p>
                    </div>
                    
                    <div class="image-container">
                        <h3>📸 Captured PCB Image</h3>
                        <img src="data:image/jpeg;base64,{image_base64}" alt="PCB Image">
                    </div>
                    
                    <div class="timestamp">
                        ⏰ Last Updated: {result['timestamp']}
                    </div>
                    
                    <div class="refresh">
                        🔄 Page auto-refreshes every 5 seconds
                    </div>
                </div>
            </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
            <head><title>Error</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>❌ Error Loading Result</h1>
                <p style="color: #dc3545;">{str(e)}</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PCB Defect Detection API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("🌐 Latest Result View: http://localhost:8000/latest")
    
    uvicorn.run("pcb_detection_api:app", host="0.0.0.0", port=8000, reload=False)
