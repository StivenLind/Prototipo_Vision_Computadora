import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Inicializar detector
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar video (o usa 0 para cámara web)
cap = cv2.VideoCapture("datos/prueba.mp4")

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

    # Producto punto para calcular ángulo
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.arccos(coseno)

    return np.degrees(angulo)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar pose
    results = pose.process(rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        # Coordenadas de articulaciones
        landmarks = results.pose_landmarks.landmark

        # Extraer hombros, codos y muñecas
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

        # Calcular ángulos en los codos
        angulo_izq = calcular_angulo(left_shoulder, left_elbow, left_wrist)
        angulo_der = calcular_angulo(right_shoulder, right_elbow, right_wrist)

        # Dibujar brazos
        for punto in [left_shoulder, left_elbow, left_wrist,
                      right_shoulder, right_elbow, right_wrist]:
            cv2.circle(frame, punto, 6, (0, 0, 255), -1)

        for start, end in [(left_shoulder, left_elbow), (left_elbow, left_wrist),
                           (right_shoulder, right_elbow), (right_elbow, right_wrist)]:
            cv2.line(frame, start, end, (0, 255, 0), 3)

        # Mostrar ángulos en pantalla
        cv2.putText(frame, f"Angulo codo izq: {int(angulo_izq)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Angulo codo der: {int(angulo_der)}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Mostrar ventana
    cv2.imshow("Calculo de Angulos en Codos", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
