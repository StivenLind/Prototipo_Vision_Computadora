import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Inicializar detector
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar video (o usa 0 para cámara web)
cap = cv2.VideoCapture("datos/prueba.mp4")

# Conexiones que nos interesan (solo brazos)
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
]

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

        # Dibujar solo los brazos
        for conn in arm_connections:
            start = results.pose_landmarks.landmark[conn[0].value]
            end = results.pose_landmarks.landmark[conn[1].value]

            # Convertir coordenadas normalizadas a pixeles
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))

            # Dibujar línea entre puntos
            cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

            # Dibujar círculos en las articulaciones
            cv2.circle(frame, start_point, 6, (0, 0, 255), -1)
            cv2.circle(frame, end_point, 6, (0, 0, 255), -1)

    # Mostrar ventana
    cv2.imshow("Detección de Brazos", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
