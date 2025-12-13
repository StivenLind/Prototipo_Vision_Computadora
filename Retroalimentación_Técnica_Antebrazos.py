# =========================================================
# RETROALIMENTACIÓN EN TIEMPO REAL – GOLPE DE ANTEBRAZOS
# Cámara web + MediaPipe + Modelo IA entrenado
# =========================================================

# Requisitos:
# pip install opencv-python mediapipe numpy pandas scikit-learn joblib

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# ---------------- CARGA DEL MODELO ----------------
modelo = joblib.load("modelo_tecnica_antebrazos.pkl")

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)

# ---------------- FUNCIONES ----------------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def evaluar_tecnica(ang_brazos, ang_rodilla, ang_tronco):
    entrada = pd.DataFrame([{
        "angulo_brazos": ang_brazos,
        "angulo_rodilla": ang_rodilla,
        "angulo_tronco": ang_tronco
    }])
    pred = modelo.predict(entrada)[0]
    return "CORRECTO" if pred == 1 else "INCORRECTO"

# ---------------- CÁMARA ----------------
cap = cv2.VideoCapture(0)  # Iriun Cam funciona como webcam

captura_realizada = False
resultado = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # -------- KEYPOINTS NECESARIOS --------
        pts = {
            "hombro": lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "codo": lm[mp_pose.PoseLandmark.LEFT_ELBOW],
            "muñeca": lm[mp_pose.PoseLandmark.LEFT_WRIST],
            "cadera": lm[mp_pose.PoseLandmark.LEFT_HIP],
            "rodilla": lm[mp_pose.PoseLandmark.LEFT_KNEE],
            "tobillo": lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        }

        # Coordenadas
        def xy(p): return [int(p.x * w), int(p.y * h)]

        hombro, codo, muñeca = xy(pts["hombro"]), xy(pts["codo"]), xy(pts["muñeca"])
        cadera, rodilla, tobillo = xy(pts["cadera"]), xy(pts["rodilla"]), xy(pts["tobillo"])

        # -------- ÁNGULOS --------
        ang_brazos = calcular_angulo(hombro, codo, muñeca)
        ang_rodilla = calcular_angulo(cadera, rodilla, tobillo)
        ang_tronco = calcular_angulo(hombro, cadera, rodilla)

        # -------- DIBUJO SIMPLIFICADO --------
        for p in [hombro, codo, muñeca, cadera, rodilla, tobillo]:
            cv2.circle(frame, tuple(p), 6, (0, 255, 0), -1)

        cv2.line(frame, tuple(hombro), tuple(codo), (255, 255, 255), 2)
        cv2.line(frame, tuple(codo), tuple(muñeca), (255, 255, 255), 2)
        cv2.line(frame, tuple(cadera), tuple(rodilla), (255, 255, 255), 2)
        cv2.line(frame, tuple(rodilla), tuple(tobillo), (255, 255, 255), 2)

        # -------- TEXTO EN PANTALLA --------
        cv2.putText(frame, f"Brazos: {int(ang_brazos)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Rodilla: {int(ang_rodilla)}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tronco: {int(ang_tronco)}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------- CAPTURA MANUAL (TECLA C) --------
        if captura_realizada:
            cv2.putText(frame, resultado, (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            resultado = evaluar_tecnica(ang_brazos, ang_rodilla, ang_tronco)
            captura_realizada = True

    cv2.imshow("Retroalimentacion Tecnica – Antebrazos", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
