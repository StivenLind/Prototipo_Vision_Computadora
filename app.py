import base64
import os
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_tecnica_antebrazos.pkl")
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontro el modelo en: {MODEL_PATH}. Ejecuta primero modelo_IA.py"
    )

model = joblib.load(MODEL_PATH)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def select_visible_side(landmarks):
    sides = {
        "LEFT": {
            "hombro": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "codo": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            "muneca": landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
            "cadera": landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            "rodilla": landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            "tobillo": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
        },
        "RIGHT": {
            "hombro": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "codo": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            "muneca": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
            "cadera": landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            "rodilla": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            "tobillo": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
        },
    }

    def score(points):
        return sum(getattr(p, "visibility", 0.0) for p in points.values())

    selected_side = max(sides, key=lambda s: score(sides[s]))
    return selected_side, sides[selected_side]


def feedback_messages(angles, result):
    feedback = []

    if angles["brazos"] < 150:
        feedback.append("Extiende mas los brazos para formar una plataforma firme.")
    elif angles["brazos"] > 180:
        feedback.append("Relaja ligeramente los codos para evitar hiperextension.")

    if angles["rodilla"] < 90:
        feedback.append("Sube un poco la postura: la flexion de rodilla es muy cerrada.")
    elif angles["rodilla"] > 135:
        feedback.append("Flexiona mas las rodillas para estabilizar el centro de gravedad.")

    if angles["tronco"] < 110:
        feedback.append("Endereza un poco el tronco, hay demasiada inclinacion.")
    elif angles["tronco"] > 160:
        feedback.append("Inclina un poco mas el tronco para una mejor recepcion.")

    if result == "CORRECTO" and not feedback:
        feedback.append("Tecnica correcta. Mantener postura y sincronizacion.")
    elif result == "CORRECTO":
        feedback.append("Buena ejecucion general. Ajusta detalles para mayor precision.")

    if result == "INCORRECTO" and not feedback:
        feedback.append("Se detectaron errores tecnicos. Ajusta postura y repite.")

    return feedback


def parse_image_from_data_url(data_url):
    if not data_url:
        return None

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/api/evaluar_frame", methods=["POST", "OPTIONS"])
def evaluate_frame():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    frame = parse_image_from_data_url(payload.get("image"))

    if frame is None:
        return jsonify({"ok": False, "message": "Frame invalido."}), 400

    original_height, original_width, _ = frame.shape

    frame_small = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return jsonify({
            "ok": False,
            "message": "No se detecto pose. Ubicate lateral a la camara.",
            "timestamp": datetime.utcnow().isoformat()
        })

    landmarks = results.pose_landmarks.landmark
    side_name, side_points = select_visible_side(landmarks)

    scale_x = original_width / PROCESS_WIDTH
    scale_y = original_height / PROCESS_HEIGHT

    def to_xy(point):
        return [
            int(point.x * PROCESS_WIDTH * scale_x),
            int(point.y * PROCESS_HEIGHT * scale_y),
        ]

    hombro = to_xy(side_points["hombro"])
    codo = to_xy(side_points["codo"])
    muneca = to_xy(side_points["muneca"])
    cadera = to_xy(side_points["cadera"])
    rodilla = to_xy(side_points["rodilla"])
    tobillo = to_xy(side_points["tobillo"])

    angles = {
        "brazos": round(calculate_angle(hombro, codo, muneca), 1),
        "rodilla": round(calculate_angle(cadera, rodilla, tobillo), 1),
        "tronco": round(calculate_angle(hombro, cadera, rodilla), 1),
    }

    model_input = pd.DataFrame([{
        "angulo_brazos": angles["brazos"],
        "angulo_rodilla": angles["rodilla"],
        "angulo_tronco": angles["tronco"],
    }])

    prediction = int(model.predict(model_input)[0])
    result = "CORRECTO" if prediction == 1 else "INCORRECTO"

    return jsonify({
        "ok": True,
        "resultado": result,
        "lado": side_name,
        "angulos": angles,
        "feedback": feedback_messages(angles, result),
        "puntos": {
            "hombro": hombro,
            "codo": codo,
            "muneca": muneca,
            "cadera": cadera,
            "rodilla": rodilla,
            "tobillo": tobillo,
        },
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
