

import os
import io
import base64
import sqlite3
import hashlib
import hmac
import tempfile
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import csv
import smtplib
import threading
import copy
import shutil
import glob
from queue import Queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel

# ReportLab for PDF export
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR / "skin_disease_mtl_best.pth"),
)

MODEL_DIR = BASE_DIR / "model_versions"
MODEL_DIR.mkdir(exist_ok=True)

# Path to your HAM10000 dataset for evaluation
EVAL_DATASET_PATH = os.environ.get("EVAL_DATASET_PATH", str(BASE_DIR / "ham10000_dataset"))

DB_PATH = os.environ.get(
    "DB_PATH",
    str(Path(tempfile.gettempdir()) / "stratum_users.db"),
)

TARGET_CLASSES = [
    "seborrheic keratosis",
    "nevus",
    "squamous cell carcinoma",
    "melanoma",
    "basal cell carcinoma",
    "actinic keratosis",
    "pigmented benign keratosis",
]

CLASS_DESCRIPTIONS = {
    "nevus": "Symmetric border and even pigment are consistent with a common, benign mole.",
    "melanoma": "Irregular border and pigment variation are patterns associated with this finding.",
    "basal cell carcinoma": "Pearly, translucent texture pattern is characteristic of this finding.",
    "seborrheic keratosis": 'Waxy, "stuck-on" texture pattern shares traits with this finding.',
    "squamous cell carcinoma": "Scaly, crusted surface pattern overlaps with this finding.",
    "actinic keratosis": "Rough, sandpaper-like surface pattern partially overlaps with this finding.",
    "pigmented benign keratosis": "Flat, evenly pigmented patch pattern is consistent with this finding.",
}

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================

EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8000")

email_queue = Queue()

def send_email_worker():
    while True:
        try:
            email_data = email_queue.get()
            if email_data is None:
                break
            
            to_email = email_data.get("to")
            subject = email_data.get("subject")
            html_body = email_data.get("html_body")
            
            if not EMAIL_ENABLED or not EMAIL_USER:
                print(f"[EMAIL] Would send to {to_email}: {subject}")
                continue
            
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = EMAIL_FROM
            msg['To'] = to_email
            
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            print(f"[EMAIL] Sent to {to_email}: {subject}")
            
        except Exception as e:
            print(f"[EMAIL] Error sending to {email_data.get('to')}: {e}")
        finally:
            email_queue.task_done()

email_thread = threading.Thread(target=send_email_worker, daemon=True)
email_thread.start()

def queue_email(to_email: str, subject: str, html_body: str):
    email_queue.put({"to": to_email, "subject": subject, "html_body": html_body})

# ============================================================================
# MODEL
# ============================================================================

class AIDermatologist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Linear(768, num_classes)
        self.malignant_head = nn.Linear(768, 1)

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        return self.type_head(f), self.malignant_head(f)


def load_trained_model():
    global TARGET_CLASSES

    if not os.path.isfile(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Run '/api/predict' will fail.")
        return None, None

    model = AIDermatologist(len(TARGET_CLASSES))

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            checkpoint_names = checkpoint.get("names")
            if checkpoint_names is not None:
                try:
                    if hasattr(checkpoint_names, "tolist"):
                        checkpoint_names = checkpoint_names.tolist()
                    else:
                        checkpoint_names = list(checkpoint_names)
                except TypeError:
                    checkpoint_names = [checkpoint_names]

                checkpoint_names = [str(name).strip() for name in checkpoint_names if str(name).strip()]
                if len(checkpoint_names) > 0:
                    TARGET_CLASSES = checkpoint_names
                    model.type_head = nn.Linear(768, len(TARGET_CLASSES))
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load trained model weights from {MODEL_PATH}: {exc}") from exc

    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"[OK] Loaded trained model: {MODEL_PATH}")
    print(f"[OK] Device: {DEVICE}")

    return model, transform


try:
    GLOBAL_MODEL, GLOBAL_TRANSFORM = load_trained_model()
except Exception as e:
    print(f"[ERROR] {e}")
    GLOBAL_MODEL, GLOBAL_TRANSFORM = None, None


# ============================================================================
# CLIP VALIDATOR
# ============================================================================

CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
CLIP_THRESHOLD = float(os.environ.get("CLIP_THRESHOLD", "0.60"))
CLIP_MARGIN = float(os.environ.get("CLIP_MARGIN", "0.05"))
CLIP_AVAILABLE = False
CLIP_MODEL = None
CLIP_PROCESSOR = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


def load_clip_validator():
    global CLIP_AVAILABLE, CLIP_MODEL, CLIP_PROCESSOR
    if CLIPModel is None:
        print("[WARNING] transformers is not installed; CLIP gate disabled.")
        return
    try:
        print(f"[INFO] Loading pretrained skin validator: {CLIP_MODEL_NAME}")
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        CLIP_AVAILABLE = True
        print("[OK] Pretrained CLIP skin validator loaded.")
    except Exception as exc:
        CLIP_AVAILABLE = False
        CLIP_MODEL = None
        CLIP_PROCESSOR = None
        print(f"[WARNING] CLIP validator could not be loaded: {exc}")


CLIP_PROMPTS = [
    "a close-up clinical photograph of a skin lesion on human skin",
    "a close-up photograph of normal human skin",
    "a non-skin image such as an object, animal, food, document, building, or scenery",
]


def clip_skin_gate(img_pil):
    if not CLIP_AVAILABLE:
        return {"available": False, "reason": "clip_unavailable"}
    try:
        inputs = CLIP_PROCESSOR(
            text=CLIP_PROMPTS,
            images=img_pil,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = CLIP_MODEL(**inputs)
            scores = F.softmax(outputs.logits_per_image, dim=1)[0].detach().cpu().numpy()
        lesion, normal_skin, non_skin = [float(x) for x in scores]
        skin = lesion + normal_skin
        margin = skin - non_skin
        is_skin = skin >= CLIP_THRESHOLD and margin >= CLIP_MARGIN
        return {
            "available": True,
            "is_skin": bool(is_skin),
            "lesion_score": round(lesion, 4),
            "normal_skin_score": round(normal_skin, 4),
            "non_skin_score": round(non_skin, 4),
            "skin_score": round(skin, 4),
            "margin": round(margin, 4),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ============================================================================
# IMAGE QUALITY VALIDATION
# ============================================================================

MIN_WIDTH = int(os.environ.get("MIN_IMAGE_WIDTH", "256"))
MIN_HEIGHT = int(os.environ.get("MIN_IMAGE_HEIGHT", "256"))
MIN_BRIGHTNESS = float(os.environ.get("MIN_BRIGHTNESS", "35"))
MAX_BRIGHTNESS = float(os.environ.get("MAX_BRIGHTNESS", "235"))
MIN_CONTRAST = float(os.environ.get("MIN_CONTRAST", "8"))
MIN_DETAIL = float(os.environ.get("MIN_DETAIL", "5"))


def image_quality(img_np):
    h, w = img_np.shape[:2]
    gray = np.asarray(Image.fromarray(img_np).convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if min(h, w) >= 3:
        detail = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    else:
        detail = 0.0
    passed = (
        w >= MIN_WIDTH and h >= MIN_HEIGHT
        and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        and contrast >= MIN_CONTRAST
        and detail >= MIN_DETAIL
    )
    return {
        "passed": bool(passed),
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "detail": round(detail, 2),
    }


# ============================================================================
# AUTHENTICATION + DATABASE
# ============================================================================

SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "24"))
ADMIN_EMAIL = os.environ.get("STRATUM_ADMIN_EMAIL", "admin@stratum.local").strip().lower()
ADMIN_PASSWORD = os.environ.get("STRATUM_ADMIN_PASSWORD", "ChangeMe123!")


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    elif isinstance(salt, str):
        salt = bytes.fromhex(salt)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240000)
    return f"pbkdf2$240000${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        prefix, iterations, salt_hex, digest_hex = stored.split("$", 3)
        if prefix != "pbkdf2":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except Exception:
        return False


def token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def init_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_base64 TEXT NOT NULL,
            status TEXT NOT NULL,
            skin_status TEXT,
            disease TEXT,
            confidence REAL,
            malignancy_risk TEXT,
            malignancy_probability REAL,
            validation_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            correct BOOLEAN NOT NULL,
            corrected_disease TEXT,
            comments TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(analysis_id) REFERENCES analyses(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
            ("Stratum Administrator", ADMIN_EMAIL, hash_password(ADMIN_PASSWORD), "admin", now_iso()),
        )
        print(f"[OK] Created admin account: {ADMIN_EMAIL}")
    conn.commit()
    conn.close()


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def create_session(user_id):
    token = base64.urlsafe_b64encode(os.urandom(48)).decode().rstrip("=")
    created = datetime.utcnow()
    expires = created + timedelta(hours=SESSION_TTL_HOURS)
    conn = db_connect()
    conn.execute(
        "INSERT INTO sessions(user_id,token_hash,created_at,expires_at) VALUES(?,?,?,?)",
        (user_id, token_hash(token), created.replace(microsecond=0).isoformat()+"Z", expires.replace(microsecond=0).isoformat()+"Z"),
    )
    conn.commit(); conn.close()
    return token


def get_user_from_token(token):
    if not token:
        return None
    conn = db_connect()
    row = conn.execute("""
        SELECT u.id,u.name,u.email,u.role,u.created_at,s.expires_at
        FROM sessions s JOIN users u ON u.id=s.user_id
        WHERE s.token_hash=?
    """, (token_hash(token),)).fetchone()
    if row is None:
        conn.close(); return None
    try:
        expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if expires <= datetime.now(expires.tzinfo):
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))
            conn.commit(); conn.close(); return None
    except Exception:
        conn.close(); return None
    user = dict(row)
    conn.close()
    return user


def get_request_user(request):
    token = request.cookies.get("stratum_session")
    return get_user_from_token(token)


def require_user(request):
    user = get_request_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_admin(request):
    user = require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return user


def save_analysis(user_id, img_np, status, skin_status=None, disease=None, confidence=None,
                  malignancy_risk=None, malignancy_probability=None, validation=None):
    pil = Image.fromarray(img_np).convert("RGB")
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=88)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analyses(user_id,image_base64,status,skin_status,disease,confidence,
            malignancy_risk,malignancy_probability,validation_json,created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (user_id,image_b64,status,skin_status,disease,confidence,malignancy_risk,
          malignancy_probability,json.dumps(validation or {}),now_iso()))
    analysis_id = cur.lastrowid
    conn.commit(); conn.close()
    return analysis_id


def admin_analysis(row):
    return {
        "id": row["id"], "user_id": row["user_id"], "user_name": row["user_name"],
        "user_email": row["user_email"], "status": row["status"], "skin_status": row["skin_status"],
        "disease": row["disease"], "confidence": row["confidence"],
        "malignancy_risk": row["malignancy_risk"], "malignancy_probability": row["malignancy_probability"],
        "validation": json.loads(row["validation_json"] or "{}"), "created_at": row["created_at"],
        "image_base64": row["image_base64"],
    }


# ============================================================================
# PREDICTION PIPELINE
# ============================================================================

def predict_pro(img_np, user_id, user_name=None, user_email=None, target_model=None):
    img_pil = Image.fromarray(img_np).convert("RGB")
    gate = clip_skin_gate(img_pil)
    quality = image_quality(img_np)

    if target_model is None:
        target_model = GLOBAL_MODEL

    if gate.get("available") and not gate.get("is_skin"):
        aid = save_analysis(user_id, img_np, "rejected", "non_skin", validation={"gate": gate, "quality": quality})
        return {
            "status": "invalid", "reason": "non_skin_image", "analysis_id": aid,
            "message": "This image does not appear to contain a suitable human skin or skin-lesion region.",
            "gate": gate, "quality": quality, "timestamp": now_iso()
        }

    if not quality["passed"]:
        aid = save_analysis(user_id, img_np, "rejected", "skin" if gate.get("is_skin") else "unknown",
                            validation={"gate": gate, "quality": quality})
        return {
            "status": "invalid", "reason": "image_quality_failed", "analysis_id": aid,
            "message": "The image quality is not sufficient for analysis.",
            "gate": gate, "quality": quality, "timestamp": now_iso()
        }

    input_tensor = GLOBAL_TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        type_logits, mal_logits = target_model(input_tensor)
        probs = F.softmax(type_logits, dim=1).cpu().numpy()[0]
        mal_prob = torch.sigmoid(mal_logits).item()

    confidences = {TARGET_CLASSES[i]: float(probs[i]) for i in range(len(TARGET_CLASSES))}
    ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_conf = ranked[0][0], ranked[0][1] * 100

    if top_conf < 50.0:
        aid = save_analysis(user_id, img_np, "uncertain", "skin", validation={"gate": gate, "quality": quality})
        return {
            "status": "uncertain", "analysis_id": aid,
            "message": "The AI cannot confidently identify a disease pattern.",
            "gate": gate, "quality": quality, "timestamp": now_iso()
        }

    risk = "HIGH" if mal_prob > 0.5 else "LOW"
    aid = save_analysis(user_id, img_np, "completed", "skin", top_label, top_conf, risk, mal_prob * 100,
                        {"gate": gate, "quality": quality})
    
    results = [
        {"name": name.title(), "confidence": round(conf * 100, 1), "primary": i == 0,
         "description": CLASS_DESCRIPTIONS.get(name, "Pattern match based on the model's learned features.")}
        for i, (name, conf) in enumerate(ranked[:3])
    ]
    return {
        "status": "ok", "analysis_id": aid, "results": results,
        "malignancy_risk": risk, "malignancy_probability": round(mal_prob * 100, 1),
        "gate": gate, "quality": quality, "timestamp": now_iso()
    }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Stratum Prediction API", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AccountUpdateRequest(BaseModel):
    current_password: str
    name: Optional[str] = None
    email: Optional[str] = None
    new_password: Optional[str] = None


class AnalysisUpdateRequest(BaseModel):
    status: Optional[str] = None
    skin_status: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    malignancy_risk: Optional[str] = None
    malignancy_probability: Optional[float] = None


class FeedbackRequest(BaseModel):
    correct: bool
    corrected_disease: Optional[str] = None
    comments: Optional[str] = None


@app.on_event("startup")
def startup():
    init_db()
    load_clip_validator()
    print("\n" + "=" * 70)
    print("STRATUM BACKEND STARTED")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Disease model: {MODEL_PATH}")
    print(f"CLIP validator: {CLIP_MODEL_NAME} | available={CLIP_AVAILABLE}")
    print(f"Database: {DB_PATH}")
    print(f"Admin email: {ADMIN_EMAIL}")
    print("=" * 70)


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/api/auth/register")
def register(payload: RegisterRequest):
    name = payload.name.strip()
    email = str(payload.email).strip().lower()
    password = payload.password
    if len(name) < 2:
        raise HTTPException(400, "Name must contain at least 2 characters.")
    if len(password) < 8:
        raise HTTPException(400, "Password must contain at least 8 characters.")
    conn = db_connect()
    try:
        conn.execute("INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                     (name, email, hash_password(password), "user", now_iso()))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists.")
    finally:
        conn.close()
    return {"status": "ok", "message": "Account created successfully."}


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    email = str(payload.email).strip().lower()
    conn = db_connect()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(401, "Invalid email or password.")
    token = create_session(row["id"])
    response = JSONResponse({"status": "ok", "user": {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}})
    response.set_cookie("stratum_session", token, httponly=True, samesite="lax", max_age=SESSION_TTL_HOURS*3600)
    return response


@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.cookies.get("stratum_session")
    if token:
        conn = db_connect()
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))
        conn.commit()
        conn.close()
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("stratum_session")
    return response


@app.get("/api/auth/me")
def me(request: Request):
    user = require_user(request)
    return {"status": "ok", "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}}


@app.post("/api/account")
def update_account(payload: AccountUpdateRequest, request: Request):
    user = require_user(request)
    conn = db_connect()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    if row is None or not verify_password(payload.current_password, row["password_hash"]):
        conn.close()
        raise HTTPException(401, "Current password is incorrect.")

    new_name = payload.name.strip() if payload.name else row["name"]
    new_email = str(payload.email).strip().lower() if payload.email else row["email"]
    new_hash = hash_password(payload.new_password) if payload.new_password else row["password_hash"]

    if payload.new_password and len(payload.new_password) < 8:
        conn.close()
        raise HTTPException(400, "New password must contain at least 8 characters.")

    if new_email != row["email"]:
        exists = conn.execute("SELECT id FROM users WHERE email=? AND id!=?", (new_email, user["id"])).fetchone()
        if exists:
            conn.close()
            raise HTTPException(409, "That email is already in use by another account.")

    conn.execute("UPDATE users SET name=?, email=?, password_hash=? WHERE id=?", (new_name, new_email, new_hash, user["id"]))
    conn.commit()
    conn.close()
    return {"status": "ok", "user": {"id": user["id"], "name": new_name, "email": new_email, "role": user["role"]}}


# ============================================================================
# PREDICT ENDPOINT (STRICT ADMIN CHECK)
# ============================================================================

@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...), model_version: Optional[str] = Form(None)):
    user = require_user(request)
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(contents) > 10*1024*1024:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        target_model = GLOBAL_MODEL
        if model_version:
            if user["role"] != "admin":
                raise HTTPException(403, "Forbidden: Only Administrators can select model versions.")
            target_model = load_model_by_version(model_version)
        
        full_result = predict_pro(img_np, user["id"], user["name"], user["email"], target_model=target_model)
        
        if user["role"] != "admin":
            safe = {
                "status": full_result["status"],
                "analysis_id": full_result.get("analysis_id"),
                "message": ("Your image was submitted successfully." if full_result["status"] == "ok" else full_result.get("message", "The image could not be accepted.")),
                "timestamp": full_result.get("timestamp")
            }
            return safe
        return full_result
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Server error: {exc}")


# ============================================================================
# FEEDBACK ENDPOINT
# ============================================================================

@app.post("/api/feedback/{analysis_id}")
async def submit_feedback(analysis_id: int, payload: FeedbackRequest, request: Request):
    user = require_user(request)
    
    conn = db_connect()
    analysis = conn.execute("SELECT id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if analysis is None:
        conn.close()
        raise HTTPException(404, "Analysis not found.")
    
    conn.execute("""
        INSERT INTO feedback(analysis_id, user_id, correct, corrected_disease, comments, created_at)
        VALUES(?,?,?,?,?,?)
    """, (analysis_id, user["id"], payload.correct, payload.corrected_disease, payload.comments, now_iso()))
    conn.commit()
    conn.close()
    
    return {"status": "ok", "message": "Feedback submitted successfully."}


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.get("/api/my-submissions")
def my_submissions(request: Request):
    user = require_user(request)
    conn = db_connect()
    rows = conn.execute("SELECT id,status,skin_status,created_at FROM analyses WHERE user_id=? ORDER BY id DESC", (user["id"],)).fetchall()
    conn.close()
    return {"status": "ok", "submissions": [dict(row) for row in rows]}


@app.get("/api/admin/analyses")
def admin_analyses(request: Request):
    require_admin(request)
    conn = db_connect()
    rows = conn.execute("""
        SELECT a.*,u.name user_name,u.email user_email
        FROM analyses a JOIN users u ON u.id=a.user_id ORDER BY a.id DESC
    """).fetchall()
    conn.close()
    return {"status": "ok", "analyses": [admin_analysis(r) for r in rows]}


@app.get("/api/admin/patients/list")
def admin_patients_list(request: Request):
    require_admin(request)
    conn = db_connect()
    rows = conn.execute("""
        SELECT u.id, u.name, u.email, u.created_at,
               COUNT(a.id) as analysis_count,
               SUM(CASE WHEN a.status = 'completed' THEN 1 ELSE 0 END) as completed_count
        FROM users u
        LEFT JOIN analyses a ON a.user_id = u.id
        WHERE u.role = 'user'
        GROUP BY u.id
        ORDER BY u.name ASC
    """).fetchall()
    conn.close()
    return {"status": "ok", "patients": [dict(r) for r in rows]}


@app.get("/api/admin/analyses/list")
def admin_analyses_list(request: Request):
    require_admin(request)
    conn = db_connect()
    rows = conn.execute("""
        SELECT a.id, a.status, a.disease, a.created_at, u.name as patient_name, u.email as patient_email
        FROM analyses a JOIN users u ON u.id = a.user_id ORDER BY a.id DESC LIMIT 100
    """).fetchall()
    conn.close()
    return {"status": "ok", "analyses": [dict(r) for r in rows]}


@app.put("/api/admin/analysis/{analysis_id}")
def admin_update_analysis(analysis_id: int, payload: AnalysisUpdateRequest, request: Request):
    require_admin(request)
    conn = db_connect()
    row = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "Analysis not found.")

    updates = payload.dict(exclude_unset=True)
    if not updates:
        conn.close()
        raise HTTPException(400, "No fields provided.")

    fields = list(updates.keys())
    values = [updates[f] for f in fields]
    set_clause = ", ".join(f"{f}=?" for f in fields)
    conn.execute(f"UPDATE analyses SET {set_clause} WHERE id=?", (*values, analysis_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/admin/analysis/{analysis_id}")
def admin_delete_analysis(analysis_id: int, request: Request):
    require_admin(request)
    conn = db_connect()
    row = conn.execute("SELECT id FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "Analysis not found.")
    conn.execute("DELETE FROM feedback WHERE analysis_id=?", (analysis_id,))
    conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted_id": analysis_id}

@app.delete("/api/admin/clear-all-data")
def admin_clear_all_data(request: Request):
    """Clears ALL analyses and feedback data from the database."""
    require_admin(request)
    
    conn = db_connect()
    # Delete feedback first (foreign key constraint)
    conn.execute("DELETE FROM feedback")
    # Delete all analyses
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
    
    return {"status": "ok", "message": "All patient analyses and feedback deleted successfully."}
@app.get("/api/admin/stats")
def admin_stats(request: Request):
    require_admin(request)
    conn = db_connect()
    users = conn.execute("SELECT COUNT(*) c FROM users WHERE role='user'").fetchone()["c"]
    total = conn.execute("SELECT COUNT(*) c FROM analyses").fetchone()["c"]
    completed = conn.execute("SELECT COUNT(*) c FROM analyses WHERE status='completed'").fetchone()["c"]
    conn.close()
    return {"status": "ok", "users": users, "total_analyses": total, "completed": completed}


# ============================================================================
# MODEL REGISTRY & HUMAN-IN-THE-LOOP
# ============================================================================

def get_available_models():
    models = []
    if os.path.isfile(MODEL_PATH):
        models.append({"version": "best", "name": "Original Model (best)", "path": MODEL_PATH})
    for model_file in sorted(MODEL_DIR.glob("*.pth")):
        version = model_file.stem.split("_v")[-1]
        models.append({"version": version, "name": f"Fine-tuned v{version}", "path": str(model_file)})
    return models

def load_model_by_version(version):
    models = get_available_models()
    model_path = None
    for m in models:
        if m["version"] == str(version):
            model_path = m["path"]
            break
    if not model_path:
        raise HTTPException(404, "Model version not found.")
    try:
        model = AIDermatologist(len(TARGET_CLASSES))
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        return model.to(DEVICE).eval()
    except Exception as e:
        raise HTTPException(500, f"Failed to load model v{version}: {e}")

@app.get("/api/admin/models")
def admin_list_models(request: Request):
    require_admin(request)
    return {"status": "ok", "models": get_available_models()}

# ============================================================================
# FINE-TUNING LOGIC
# ============================================================================

FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4

def get_training_data():
    conn = db_connect()
    rows = conn.execute("""
        SELECT a.image_base64, LOWER(f.corrected_disease) as corrected_disease
        FROM feedback f
        JOIN analyses a ON f.analysis_id = a.id
        WHERE f.correct = 0 
          AND f.corrected_disease IS NOT NULL 
          AND f.corrected_disease != ''
          AND LOWER(f.corrected_disease) IN ({})
        GROUP BY a.id
        HAVING f.id = MAX(f.id)
    """.format(','.join('?' * len(TARGET_CLASSES))), TARGET_CLASSES).fetchall()
    conn.close()
    return rows

def fine_tune_model():
    if GLOBAL_MODEL is None:
        return {"status": "error", "message": "Base model not loaded."}
    
    training_rows = get_training_data()
    if len(training_rows) < 5:
        return {"status": "error", "message": "Not enough verified feedback samples (min 5 required)."}

    # CRITICAL FIX: Check if duplicates were loaded!
    unique_images = set(row[0] for row in training_rows)  # Get unique Base64 images
    if len(unique_images) != len(training_rows):
        return {"status": "error", "message": "Duplicate images found in feedback. Please delete duplicates and try again."}

    print(f"[FINE-TUNE] Starting fine-tuning with {len(training_rows)} verified samples...")

    # ... rest of your function ...

    model_copy = copy.deepcopy(GLOBAL_MODEL)
    
    for param in model_copy.parameters():
        param.requires_grad = False
    
    for param in model_copy.features[-2:].parameters():
        param.requires_grad = True
    for param in model_copy.type_head.parameters():
        param.requires_grad = True
    for param in model_copy.malignant_head.parameters():
        param.requires_grad = True

    from torch.utils.data import Dataset, DataLoader

    class HardExampleDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.transform = GLOBAL_TRANSFORM
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            img_b64, label = self.rows[idx]
            img_data = base64.b64decode(img_b64)
            img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_tensor = self.transform(img_pil)
            label_idx = TARGET_CLASSES.index(label)
            return img_tensor, torch.tensor(label_idx)

    dataset = HardExampleDataset(training_rows)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_copy.parameters()), lr=FINE_TUNE_LR)
    criterion = nn.CrossEntropyLoss()

    model_copy = model_copy.to(DEVICE)
    model_copy.train()
    
    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            type_logits, _ = model_copy(inputs)
            loss = criterion(type_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[FINE-TUNE] Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")

    version = len(list(MODEL_DIR.glob("*.pth"))) + 1
    new_model_path = MODEL_DIR / f"skin_disease_mtl_v{version}.pth"
    torch.save(model_copy.state_dict(), new_model_path)
    print(f"[FINE-TUNE] New model saved as: {new_model_path}")

    return {"status": "success", "message": f"Fine-tuned model saved as v{version}.", "path": str(new_model_path)}

@app.post("/api/admin/fine-tune")
def trigger_fine_tune(request: Request):
    require_admin(request)
    result = fine_tune_model()
    return result

@app.post("/api/admin/promote-model/{version}")
def promote_model(version: int, request: Request):
    require_admin(request)
    
    model_file = MODEL_DIR / f"skin_disease_mtl_v{version}.pth"
    if not model_file.is_file():
        raise HTTPException(404, "Model version not found.")

    global MODEL_PATH, GLOBAL_MODEL
    MODEL_PATH = str(model_file)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        new_model = AIDermatologist(len(TARGET_CLASSES))
        new_model.load_state_dict(state_dict, strict=True)
        GLOBAL_MODEL = new_model.to(DEVICE).eval()
        print(f"[PROMOTE] Successfully promoted version {version} to production!")
    except Exception as e:
        raise HTTPException(500, f"Failed to load model version {version}: {e}")

    return {"status": "success", "message": f"Model v{version} is now live in production."}


# ============================================================================
# MODEL EVALUATION GATE (A/B TESTING vs HAM10000)
# ============================================================================

import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_on_dataset(model, dataset_path, max_images=1500):
    """Runs a model on a sample of the validation set and returns metrics. Crash-proof."""
    if not os.path.isdir(dataset_path):
        return {"error": f"Evaluation dataset not found at {dataset_path}."}
    
    if model is None:
        return {"error": "The model could not be loaded (None)."}

    try:
        # THE KEY FIX: Use the model's exact class order
        model_classes = TARGET_CLASSES

        # Manually create dataset structure to avoid ImageFolder hanging on empty folders
        class_dirs = {}
        for cls in model_classes:
            cls_path = os.path.join(dataset_path, cls)
            if os.path.isdir(cls_path):
                files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    class_dirs[cls] = files

        if len(class_dirs) != len(model_classes):
            missing = [c for c in model_classes if c not in class_dirs]
            return {"error": f"Missing or empty folders for classes: {missing}. Please add at least 1 image to each."}

        # Get random samples
        sample_list = []
        for cls, files in class_dirs.items():
            chosen = files[:max_images]
            for f in chosen:
                sample_list.append((os.path.join(dataset_path, cls, f), cls))

        # Shuffle
        import random
        random.seed(42)  # <--- ADD THIS LINE! The "42" can be any number.
        random.shuffle(sample_list)
        sample_list = sample_list[:max_images]
        
        total_images = len(sample_list)
        print(f"[EVAL] Starting evaluation on {total_images} images...")
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, (img_path, label_name) in enumerate(sample_list):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = GLOBAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
                    type_logits, _ = model(img_tensor)
                    pred_idx = torch.argmax(type_logits, dim=1).item()
                    
                    all_preds.append(pred_idx)
                    all_labels.append(model_classes.index(label_name))
                except Exception as e:
                    continue  # Skip corrupt images
                
                # Print progress every 50 images
                if (i + 1) % 50 == 0:
                    print(f"[EVAL] Progress: {i+1}/{total_images} images processed...")

        if len(all_labels) == 0:
            return {"error": "No images were processed."}

        print(f"[EVAL] Evaluation complete. Processing {len(all_labels)} images.")

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_images": len(all_labels)
        }
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

@app.get("/api/admin/evaluate-models")
def evaluate_models(request: Request):
    """Compares Original Model vs Fine-tuned models on HAM10000."""
    require_admin(request)

    results = {"models": []}

    # 1. Evaluate Original Model
    if GLOBAL_MODEL is not None:
        original_metrics = evaluate_model_on_dataset(GLOBAL_MODEL, EVAL_DATASET_PATH)
        results["models"].append({"version": "best", "name": "Original Model", "metrics": original_metrics})

    # 2. Evaluate all Fine-tuned versions
    for model_file in sorted(MODEL_DIR.glob("*.pth")):
        version = model_file.stem.split("_v")[-1]
        try:
            fine_tuned_model = load_model_by_version(version)
            metrics = evaluate_model_on_dataset(fine_tuned_model, EVAL_DATASET_PATH)
            results["models"].append({"version": version, "name": f"Fine-tuned v{version}", "metrics": metrics})
        except Exception as e:
            results["models"].append({"version": version, "name": f"Fine-tuned v{version}", "metrics": {"error": str(e)}})

    return results

@app.post("/api/admin/auto-promote/{version}")
def auto_promote(version: int, request: Request):
    """Only promotes the new model if it beats the original on the validation set."""
    require_admin(request)
    
    # MUST BE AT THE TOP OF THE FUNCTION!
    global MODEL_PATH, GLOBAL_MODEL
    
    # Load original
    original_metrics = evaluate_model_on_dataset(GLOBAL_MODEL, EVAL_DATASET_PATH)
    
    # Load new
    try:
        new_model = load_model_by_version(version)
        new_metrics = evaluate_model_on_dataset(new_model, EVAL_DATASET_PATH)
    except Exception as e:
        raise HTTPException(404, f"Failed to load model v{version}: {e}")

    # Compare F1-score (the most balanced metric for medical)
    original_f1 = original_metrics.get("f1", 0)
    new_f1 = new_metrics.get("f1", 0)

    if new_f1 >= original_f1:
        MODEL_PATH = str(MODEL_DIR / f"skin_disease_mtl_v{version}.pth")
        GLOBAL_MODEL = new_model
        return {"status": "success", "message": f"Model v{version} promoted! (Old F1: {original_f1}, New F1: {new_f1})"}
    else:
        return {"status": "rejected", "message": f"Model v{version} REJECTED. It performs worse. (Old F1: {original_f1}, New F1: {new_f1})"}


# ============================================================================
# ANALYTICS & EVALUATION
# ============================================================================

@app.get("/api/admin/analytics")
def admin_analytics(request: Request, days: int = 30):
    require_admin(request)
    
    conn = db_connect()
    cutoff = (datetime.utcnow() - timedelta(days=days)).replace(microsecond=0).isoformat() + "Z"
    
    timeline_rows = conn.execute("""
        SELECT substr(created_at, 1, 10) as date, 
               COUNT(*) as total,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
        FROM analyses 
        WHERE created_at >= ? 
        GROUP BY date ORDER BY date ASC
    """, (cutoff,)).fetchall()
    
    disease_rows = conn.execute("""
        SELECT COALESCE(NULLIF(disease, ''), 'Unknown') as disease, COUNT(*) as count
        FROM analyses
        WHERE status = 'completed'
        GROUP BY disease ORDER BY count DESC
    """).fetchall()
    
    confidence_rows = conn.execute("""
        SELECT 
            CASE 
                WHEN confidence < 20 THEN '0-20%'
                WHEN confidence < 40 THEN '20-40%'
                WHEN confidence < 60 THEN '40-60%'
                WHEN confidence < 80 THEN '60-80%'
                ELSE '80-100%'
            END as bucket,
            COUNT(*) as count
        FROM analyses
        WHERE status = 'completed' AND confidence IS NOT NULL
        GROUP BY bucket
    """).fetchall()
    
    malignancy_rows = conn.execute("""
        SELECT COALESCE(NULLIF(malignancy_risk, ''), 'Unknown') as malignancy_risk, COUNT(*) as count
        FROM analyses
        WHERE status = 'completed'
        GROUP BY malignancy_risk
    """).fetchall()
    
    conn.close()

    return {
        "status": "ok",
        "timeline": [dict(r) for r in timeline_rows],
        "diseases": [dict(r) for r in disease_rows],
        "confidence_buckets": [dict(r) for r in confidence_rows],
        "malignancy": [dict(r) for r in malignancy_rows]
    }

@app.get("/api/admin/feedback/stats")
def admin_feedback_stats(request: Request):
    require_admin(request)
    
    conn = db_connect()
    total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
    correct = conn.execute("SELECT COUNT(*) c FROM feedback WHERE correct = 1").fetchone()["c"]
    conn.close()
    
    accuracy_rate = round((correct / total) * 100, 1) if total > 0 else 0.0
    
    return {
        "status": "ok",
        "accuracy_rate": accuracy_rate,
        "total": total,
        "correct": correct
    }

@app.get("/api/admin/feedback/list")
def admin_feedback_list(request: Request):
    require_admin(request)
    
    conn = db_connect()
    rows = conn.execute("""
        SELECT 
            f.id as feedback_id,
            f.correct,
            f.corrected_disease,
            f.comments,
            f.created_at as feedback_date,
            a.id as analysis_id,
            a.disease as predicted_disease,
            u.name as user_name,
            u.email as user_email
        FROM feedback f
        JOIN analyses a ON f.analysis_id = a.id
        JOIN users u ON f.user_id = u.id
        ORDER BY f.id DESC
    """).fetchall()
    conn.close()
    
    feedback_list = []
    for row in rows:
        feedback_list.append({
            "feedback_id": row["feedback_id"],
            "correct": bool(row["correct"]),
            "corrected_disease": row["corrected_disease"] or "N/A",
            "comments": row["comments"] or "No comments provided",
            "feedback_date": row["feedback_date"],
            "analysis_id": row["analysis_id"],
            "predicted_disease": row["predicted_disease"] or "N/A",
            "user_name": row["user_name"],
            "user_email": row["user_email"]
        })
    
    return {"status": "ok", "feedback": feedback_list}


# ============================================================================
# NEW: DELETE FEEDBACK ENDPOINT
# ============================================================================

@app.delete("/api/admin/feedback/{feedback_id}")
def admin_delete_feedback(feedback_id: int, request: Request):
    """Deletes a single feedback record."""
    require_admin(request)
    
    conn = db_connect()
    row = conn.execute("SELECT id FROM feedback WHERE id=?", (feedback_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "Feedback not found.")
    
    conn.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
    conn.commit()
    conn.close()
    
    return {"status": "ok", "deleted_id": feedback_id}


# ============================================================================
# PDF EXPORT
# ============================================================================

@app.get("/api/admin/export/patient-pdf/{patient_identifier}")
def export_patient_pdf(patient_identifier: str, request: Request):
    require_admin(request)
    
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(503, "PDF export not available. Install reportlab: pip install reportlab")
    
    patient_email = urllib.parse.unquote(patient_identifier).strip().lower()
    
    conn = db_connect()
    patient = conn.execute(
        "SELECT id, name, email, created_at FROM users WHERE email = ?",
        (patient_email,)
    ).fetchone()
    
    if patient is None:
        conn.close()
        raise HTTPException(404, f"Patient not found: {patient_email}")
    
    rows = conn.execute("""
        SELECT id, status, skin_status, disease, confidence, malignancy_risk, 
               malignancy_probability, created_at
        FROM analyses WHERE user_id = ? ORDER BY id DESC
    """, (patient["id"],)).fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(404, f"No analyses found for patient '{patient_email}'")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    styles = getSampleStyleSheet()
    
    TEAL = colors.HexColor('#00695C')
    CORAL = colors.HexColor('#D32F2F')
    GREEN = colors.HexColor('#2E7D32')
    GOLD = colors.HexColor('#F57C00')
    DARK = colors.HexColor('#1a1a2e')
    MUTED = colors.HexColor('#666666')
    
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontSize=24, textColor=TEAL,
        alignment=TA_CENTER, spaceAfter=10, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=14, textColor=MUTED,
        alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica'
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'], fontSize=16, textColor=DARK,
        spaceAfter=8, spaceBefore=16, fontName='Helvetica-Bold'
    )
    subheading_style = ParagraphStyle(
        'SubHeading', parent=styles['Heading3'], fontSize=13, textColor=TEAL,
        spaceAfter=6, spaceBefore=12, fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontSize=10, textColor=MUTED,
        leading=14, fontName='Helvetica'
    )
    
    story = []
    
    story.append(Paragraph("🧬 Stratum Patient Report", title_style))
    story.append(Paragraph("Complete Analysis History", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Patient Information", heading_style))
    info_data = [
        ["Patient Name:", patient["name"]],
        ["Patient Email:", patient["email"]],
        ["Patient ID:", f"#{patient['id']}"],
        ["Account Created:", patient["created_at"]],
        ["Total Analyses:", str(len(rows))],
        ["Report Generated:", now_iso()],
    ]
    
    info_table = Table(info_data, colWidths=[1.5*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ('BACKGROUND', (1,0), (1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Summary Statistics", heading_style))
    completed = [r for r in rows if r["status"] == "completed"]
    rejected = [r for r in rows if r["status"] == "rejected"]
    uncertain = [r for r in rows if r["status"] == "uncertain"]
    
    confidences = [r["confidence"] for r in completed if r["confidence"] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    high_risk = [r for r in completed if r["malignancy_risk"] == "HIGH"]
    low_risk = [r for r in completed if r["malignancy_risk"] == "LOW"]
    
    stats_data = [
        ["Metric", "Value"],
        ["Total Analyses", str(len(rows))],
        ["Completed", f"{len(completed)} ({len(completed)/len(rows)*100:.0f}%)"],
        ["Rejected", f"{len(rejected)} ({len(rejected)/len(rows)*100:.0f}%)"],
        ["Uncertain", f"{len(uncertain)} ({len(uncertain)/len(rows)*100:.0f}%)"],
        ["Average Confidence", f"{avg_confidence:.1f}%" if avg_confidence else "N/A"],
        ["High Risk", str(len(high_risk))],
        ["Low Risk", str(len(low_risk))],
    ]
    
    stats_table = Table(stats_data, colWidths=[2*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0,0), (1,0), TEAL),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('BACKGROUND', (0,1), (1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 15))
    
    if completed:
        story.append(Paragraph("Disease Distribution", heading_style))
        disease_counts = {}
        for r in completed:
            disease = r["disease"] or "Unknown"
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        disease_data = [["Disease", "Count", "Percentage"]]
        for disease, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(completed) * 100) if len(completed) > 0 else 0
            disease_data.append([disease, str(count), f"{pct:.1f}%"])
        
        disease_table = Table(disease_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
        disease_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,0), (-1,0), TEAL),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ]))
        story.append(disease_table)
        story.append(Spacer(1, 15))
    
    if completed:
        story.append(Paragraph("Malignancy Risk Overview", heading_style))
        risk_data = [
            ["Risk Level", "Count", "Percentage"],
            ["HIGH", str(len(high_risk)), f"{len(high_risk)/len(completed)*100:.1f}%"],
            ["LOW", str(len(low_risk)), f"{len(low_risk)/len(completed)*100:.1f}%"],
        ]
        
        risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,0), (-1,0), CORAL),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 20))
    
    story.append(PageBreak())
    story.append(Paragraph("Individual Analysis Records", heading_style))
    story.append(Paragraph(f"Showing all {len(rows)} analyses (newest first).", body_style))
    story.append(Spacer(1, 10))
    
    for idx, row in enumerate(rows):
        story.append(Paragraph(f"<b>Analysis #{row['id']}</b> — {row['created_at']}", subheading_style))
        status_text = row["status"].upper()
        status_color = GREEN if row["status"] == "completed" else (CORAL if row["status"] == "rejected" else GOLD)
        story.append(Paragraph(f"<b>Status:</b> <font color='{status_color.hexval()}'>{status_text}</font>", body_style))
        story.append(Spacer(1, 3))
        
        if row["status"] == "completed":
            details = [
                ["Disease:", row["disease"] or "Not specified"],
                ["Confidence:", f"{row['confidence']:.1f}%" if row["confidence"] else "N/A"],
                ["Malignancy Risk:", row["malignancy_risk"] or "N/A"],
                ["Malignancy Probability:", f"{row['malignancy_probability']:.1f}%" if row["malignancy_probability"] else "N/A"],
                ["Skin Status:", row["skin_status"] or "N/A"],
            ]
            details_table = Table(details, colWidths=[1.5*inch, 2.5*inch])
            details_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BACKGROUND', (0,0), (1,-1), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(details_table)
            if row["confidence"]:
                story.append(Spacer(1, 3))
                conf = row["confidence"] / 100
                bar_len = int(conf * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                story.append(Paragraph(f"<b>Confidence:</b> {bar} {row['confidence']:.1f}%", body_style))
            if row["malignancy_risk"] == "HIGH":
                story.append(Paragraph("<font color='#D32F2F'><b>⚠️ HIGH RISK</b></font> - This analysis shows elevated malignancy risk indicators.", body_style))
            elif row["malignancy_risk"] == "LOW":
                story.append(Paragraph("<font color='#2E7D32'><b>✓ LOW RISK</b></font> - This analysis shows low malignancy risk indicators.", body_style))
        elif row["status"] == "rejected":
            story.append(Paragraph(f"<b>Reason:</b> {row['skin_status'] or 'Image not suitable for analysis'}", body_style))
        elif row["status"] == "uncertain":
            story.append(Paragraph("<b>⚠️ Uncertain</b> - The AI could not confidently identify a pattern.", body_style))
        
        if idx < len(rows) - 1:
            story.append(Spacer(1, 8))
            story.append(Paragraph("─" * 80, ParagraphStyle('Separator', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
            story.append(Spacer(1, 5))
    
    story.append(PageBreak())
    story.append(Spacer(1, 50))
    story.append(Paragraph(
        "<b>DISCLAIMER</b><br/><br/>This report is for informational purposes only and is not a medical diagnosis. "
        "All analyses should be reviewed by a qualified healthcare professional. "
        "Stratum is a pattern-recognition tool and not a substitute for professional medical advice.",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=10, textColor=MUTED, alignment=TA_CENTER, leading=14)
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Report generated on {now_iso()} · Stratum v6.0", ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    
    doc.build(story)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=stratum_patient_{patient['id']}_{datetime.now().strftime('%Y%m%d')}.pdf"}
    )

@app.get("/api/admin/export/pdf/{analysis_id}")
def export_single_analysis_pdf(analysis_id: int, request: Request):
    require_admin(request)
    
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(503, "PDF export not available. Install reportlab: pip install reportlab")
    
    conn = db_connect()
    row = conn.execute("""
        SELECT a.*, u.name user_name, u.email user_email
        FROM analyses a JOIN users u ON u.id = a.user_id 
        WHERE a.id = ?
    """, (analysis_id,)).fetchone()
    conn.close()
    
    if row is None:
        raise HTTPException(404, "Analysis not found.")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#00695C'), alignment=TA_CENTER, spaceAfter=10, fontName='Helvetica-Bold')
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#1a1a2e'), spaceAfter=8, spaceBefore=16, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#666666'), leading=14, fontName='Helvetica')
    
    story = []
    story.append(Paragraph("🧬 Stratum Single Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Analysis Details", heading_style))
    info_data = [
        ["Analysis ID:", f"#{row['id']}"],
        ["Patient Name:", row["user_name"]],
        ["Patient Email:", row["user_email"]],
        ["Status:", row["status"].upper()],
        ["Created At:", row["created_at"]],
    ]
    if row["status"] == "completed":
        info_data.extend([
            ["Disease:", row["disease"] or "N/A"],
            ["Confidence:", f"{row['confidence']:.1f}%" if row["confidence"] else "N/A"],
            ["Malignancy Risk:", row["malignancy_risk"] or "N/A"],
            ["Malignancy Probability:", f"{row['malignancy_probability']:.1f}%" if row["malignancy_probability"] else "N/A"],
        ])
    
    info_table = Table(info_data, colWidths=[1.5*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10), ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ('BACKGROUND', (1,0), (1,-1), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    story.append(PageBreak())
    story.append(Paragraph("<b>DISCLAIMER</b><br/><br/>This report is for informational purposes only and is not a medical diagnosis.", ParagraphStyle('Disc', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#666666'), alignment=TA_CENTER)))
    
    doc.build(story)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=stratum_analysis_{row['id']}.pdf"}
    )


# ============================================================================
# HEALTH & FRONTEND
# ============================================================================

@app.get("/api/health")
def health():
    model_status = "loaded" if GLOBAL_MODEL is not None else "missing"
    return {"status": "ok", "device": str(DEVICE), "model_status": model_status}

FRONTEND_CANDIDATES = [BASE_DIR / "stratum-skin-analysis.html"]

@app.get("/")
def serve_frontend():
    for frontend_path in FRONTEND_CANDIDATES:
        if frontend_path.is_file():
            return FileResponse(str(frontend_path))
    return {"message": "Frontend HTML not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)