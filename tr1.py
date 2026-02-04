import cv2
from deepface import DeepFace
import numpy as np
import os
import time
from numpy.linalg import norm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------
MODEL = "SFace"
THRESHOLD = 0.56   # Static cosine distance threshold
STORE_DIR = "face_store"
SKIP_FRAMES = 5
FACE_CHANGE_THRESHOLD = 0.8
FRAME_RESIZE_WIDTH = 320


# -----------------------------------------------------------
# LOAD KNOWN FACES FROM .NPY DATABASE
# -----------------------------------------------------------
print("[INFO] Loading known faces from database...")

NAMES_FILE = os.path.join(STORE_DIR, "names.npy")
EMB_FILE = os.path.join(STORE_DIR, "embeddings.npy")

if os.path.exists(NAMES_FILE) and os.path.exists(EMB_FILE):
    known_names = np.load(NAMES_FILE, allow_pickle=True).tolist()
    known_embeddings = np.load(EMB_FILE)
    print(f"[INFO] Loaded {len(known_names)} known faces.\n")
else:
    print("[ERROR] Face database not found.")
    known_names = []
    known_embeddings = []


# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


# -----------------------------------------------------------
# SEND EMAIL NOTIFICATION
# -----------------------------------------------------------
def send_email(person_name, confidence, dist):
    sender_email = "@gmail.com"
    app_password = ""
    receiver_email = "gmail.com"

    subject = f"[ALERT] {person_name} Identified"
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""Person Identified: {person_name}
Time Detected: {time_now}
Confidence: {confidence:.2f}%
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"[EMAIL SENT] Alert sent for: {person_name}")
    except Exception as e:
        print("[EMAIL ERROR]:", e)


# -----------------------------------------------------------
# CAMERA
# -----------------------------------------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cached_emb = None
frame_count = 0

prev_name = "No Face"
prev_color = (0, 255, 0)
prev_conf = 0
prev_dist = None


# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
while True:
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

        emb = np.array(
            DeepFace.represent(face_img, model_name=MODEL, enforce_detection=False)[0]["embedding"]
        )

        if cached_emb is None or cosine_distance(emb, cached_emb) > FACE_CHANGE_THRESHOLD:
            cached_emb = emb

            if len(known_embeddings) > 0:
                dists = [cosine_distance(emb, e) for e in known_embeddings]
                best_idx = int(np.argmin(dists))
                best_dist = dists[best_idx]
                confidence = (1 - best_dist) * 100

                if best_dist <= THRESHOLD:
                    prev_name = known_names[best_idx]
                    prev_color = (0, 0, 255)
                    send_email(prev_name, confidence, best_dist)
                else:
                    prev_name = "Unknown"
                    prev_color = (0, 255, 0)

                prev_conf = confidence
                prev_dist = best_dist

    frame_count += 1

    if prev_dist is not None:
        label = f"{prev_name} | {prev_conf:.1f}% | Dist: {prev_dist:.3f}"
    else:
        label = f"{prev_name}"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, prev_color, 2)

    if len(faces) > 0:
        (x, y, fw, fh) = faces[0]
        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + fw) / scale)
        y1 = int((y + fh) / scale)
        cv2.rectangle(frame, (x0, y0), (x1, y1), prev_color, 2)

    cv2.imshow("Fast Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
