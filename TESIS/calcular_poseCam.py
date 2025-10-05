import cv2
import mediapipe as mp
import numpy as np

# Inicializar pose y dibujado
mp_pose = mp.solutions.pose
mp_dibujo = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Usa cámara del computador

def calcular_angulo(a, b, c):
    """
    Calcula el ángulo formado por tres puntos (a-b-c).
    b es el vértice del ángulo.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.arccos(coseno)
    return np.degrees(angulo)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar pose
    results = pose.process(rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Dibujar cuerpo completo
        mp_dibujo.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_dibujo.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Puntos para los codos
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        left_elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
        left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))

        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        right_elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
        right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

        # Calcular ángulos
        angulo_izq = calcular_angulo(left_shoulder, left_elbow, left_wrist)
        angulo_der = calcular_angulo(right_shoulder, right_elbow, right_wrist)

        # Mostrar ángulos sobre los codos
        cv2.putText(frame, f"{int(angulo_izq)}°",
                    (left_elbow[0] - 50, left_elbow[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"{int(angulo_der)}°",
                    (right_elbow[0] + 10, right_elbow[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Mostrar imagen con todo el cuerpo y ángulos
    cv2.imshow("Keypoints del Cuerpo + Angulos de los Brazos", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
