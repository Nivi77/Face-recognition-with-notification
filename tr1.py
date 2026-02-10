import cv2
import numpy as np
import os
import time
from deepface import DeepFace
import onnxruntime as ort
from numpy.linalg import norm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------
MODEL = "SFace"
BASE_THRESHOLD = 0.40
RTSP_URL = "rtsp://username:password@ip_address:554/stream"
FRAME_RESIZE_WIDTH = 640
SKIP_FRAMES = 3

STORE_DIR = "face_store"
NAMES_FILE = os.path.join(STORE_DIR, "names.npy")
EMB_FILE = os.path.join(STORE_DIR, "embeddings.npy")

# Email
SENDER_EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

EMAIL_RESET_SECONDS = 60  # 1 minutes

# -----------------------------------------------------------
# LOAD DATABASE
# -----------------------------------------------------------
if not os.path.exists(NAMES_FILE) or not os.path.exists(EMB_FILE):
    print("Database not found. Run build_database.py first.")
    exit()

known_names = np.load(NAMES_FILE, allow_pickle=True).tolist()
known_embeddings = np.load(EMB_FILE)

print(f"Loaded {len(known_names)} faces from database.")

# -----------------------------------------------------------
# GPU CHECK
# -----------------------------------------------------------
model_obj = DeepFace.build_model(MODEL)

try:
    session = ort.InferenceSession(
        model_obj.onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    providers = session.get_providers()
except:
    providers = ["CPUExecutionProvider"]

USING_GPU = "CUDAExecutionProvider" in providers
print("GPU Enabled:", USING_GPU)

# -----------------------------------------------------------
# CAMERA (RTSP → Webcam Fallback)
# -----------------------------------------------------------
print("Attempting RTSP connection...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP unavailable. Switching to webcam...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No camera available.")
    exit()

print("Camera initialized successfully.")

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def adaptive_threshold(gray, face_area_ratio):
    dynamic = BASE_THRESHOLD

    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)

    if brightness < 60 or brightness > 200:
        dynamic += 0.2

    if blur < 100:
        dynamic += 0.35

    if contrast < 25:
        dynamic += 0.1

    if face_area_ratio < 0.02:
        dynamic += 0.1

    print(
        f"[THR {dynamic:.3f}] "
        
    )

    return dynamic


def send_email(person_name, confidence):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"[ALERT] {person_name} Identified"
    body = f"""
Person Identified: {person_name}
Time: {time_now}
Confidence: {confidence:.2f}%
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"[EMAIL SENT] {person_name}")
    except Exception as e:
        print("[EMAIL ERROR]:", e)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

frame_count = 0
display_label = "No Face"
display_color = (0, 255, 0)

last_emailed_identity = None
last_email_time = 0

# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
while True:
    current_time = time.time()

    # Reset email identity after 3 minutes
    if last_emailed_identity is not None and \
       current_time - last_email_time > EMAIL_RESET_SECONDS:
        print("[INFO] Email identity reset after 1 minute.")
        last_emailed_identity = None

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    scale = FRAME_RESIZE_WIDTH / w
    small = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0 and frame_count % SKIP_FRAMES == 0:

        (x, y, fw, fh) = faces[0]

        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + fw) / scale)
        y1 = int((y + fh) / scale)

        face_img = frame[y0:y1, x0:x1]

        try:
            emb = np.array(
                DeepFace.represent(
                    face_img,
                    model_name=MODEL,
                    enforce_detection=False
                )[0]["embedding"]
            )
        except:
            emb = None

        if emb is not None:

            face_area_ratio = (fw * fh) / (gray.shape[0] * gray.shape[1])
            dynamic_threshold = adaptive_threshold(gray, face_area_ratio)

            dists = [cosine_distance(emb, e) for e in known_embeddings]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            confidence = (1 - best_dist) * 100

            if best_dist <= dynamic_threshold:
                person_name = known_names[best_idx]
                display_label = f"{person_name} | {dists}"
                display_color = (0, 0, 255)

                if person_name != last_emailed_identity:
                    send_email(person_name, confidence)
                    last_emailed_identity = person_name
                    last_email_time = time.time()
            else:
                display_label = f"Unknown | {dists}"
                display_color = (0, 255, 0)

            cv2.rectangle(frame, (x0, y0), (x1, y1), display_color, 2)

    frame_count += 1

    cv2.putText(frame, display_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)

    cv2.imshow("Adaptive Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
