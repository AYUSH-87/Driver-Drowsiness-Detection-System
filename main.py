import cv2
import numpy as np
import time
import winsound
from tensorflow.keras.models import load_model

# LOAD MODELS
eye_model = load_model("eye_model.keras")
mouth_model = load_model("mouth_model.keras")
IMG_SIZE = 64

# ALERT CONTROL
last_alert_time = 0
ALERT_COOLDOWN = 2  # seconds

def play_alert(msg):
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time > ALERT_COOLDOWN:
        print("[ALERT]", msg)
        winsound.Beep(1500, 700)
        last_alert_time = current_time

# CASCADES
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# PREDICTION FUNCTIONS
def predict_eye(img):
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = img.reshape(1,64,64,1)

    pred = eye_model.predict(img, verbose=0)[0][0]
    return "closed" if pred < 0.5 else "open"

def predict_mouth(img):
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = img.reshape(1,64,64,1)

    pred = mouth_model.predict(img, verbose=0)[0][0]
    return "closed" if pred < 0.5 else "open"

# CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not working")
    exit()

eye_counter = 0
mouth_counter = 0

EYE_THRESH = 5     # frames (~0.3–0.5 sec)
MOUTH_THRESH = 10   # frames (~1 sec)

print("[INFO] Press Q to quit")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    status = "ALERT"

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]

        # EYES 
        eyes = eye_cascade.detectMultiScale(face)

        for (ex,ey,ew,eh) in eyes[:1]:  # take one eye
            eye_roi = face[ey:ey+eh, ex:ex+ew]

            eye_state = predict_eye(eye_roi)

            if eye_state == "closed":
                eye_counter += 1
            else:
                eye_counter = 0

            cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)
            cv2.putText(frame, eye_state, (x+ex,y+ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        # MOUTH 
        mouths = mouth_cascade.detectMultiScale(face, 1.7, 11)

        for (mx,my,mw,mh) in mouths[:1]:
            mouth_roi = face[my:my+mh, mx:mx+mw]

            mouth_state = predict_mouth(mouth_roi)

            if mouth_state == "open":
                mouth_counter += 1
            else:
                mouth_counter = 0

            cv2.rectangle(frame, (x+mx,y+my), (x+mx+mw,y+my+mh), (255,0,0), 2)
            cv2.putText(frame, mouth_state, (x+mx,y+my-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)

        # STATUS & ALERT 
        if eye_counter > EYE_THRESH:
            status = "DROWSY (Eyes Closed)"
            play_alert(status)

        elif mouth_counter > MOUTH_THRESH:
            status = "YAWNING"
            play_alert(status)

        else:
            status = "ALERT"

    # DISPLAY
    color = (0,0,255) if status != "ALERT" else (0,180,0)

    cv2.rectangle(frame, (0,0), (frame.shape[1],50), color, -1)
    cv2.putText(frame, status, (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()