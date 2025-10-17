import cv2
import mediapipe as mp
import numpy as np

# --- Inicializar Mediapipe Pose ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Función para calcular ángulo entre tres puntos ---
def calcular_angulo(a, b, c):
    a = np.array(a)  # Primer punto
    b = np.array(b)  # Punto central
    c = np.array(c)  # Último punto
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radians * 180.0 / np.pi)
    
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

# --- Configuración de la cámara ---
# Índice 1 normalmente es la cámara del celular vía DroidCam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara. Verifica DroidCam.")
    exit()

print("✅ Cámara conectada correctamente.")

# --- Inicializar pose ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara.")
            break

        # Convertir a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Procesar pose
        results = pose.process(image)

        # Volver a BGR para visualización
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Coordenadas de hombro, codo y muñeca (brazo derecho e izquierdo)
            hombro_d = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            codo_d = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            muñeca_d = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            hombro_i = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            codo_i = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            muñeca_i = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calcular ángulos
            angulo_d = calcular_angulo(hombro_d, codo_d, muñeca_d)
            angulo_i = calcular_angulo(hombro_i, codo_i, muñeca_i)

            # Dibujar puntos y líneas del cuerpo
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Mostrar los ángulos en pantalla
            cv2.putText(image, f'Derecho: {int(angulo_d)}°',
                        tuple(np.multiply(codo_d, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, f'Izquierdo: {int(angulo_i)}°',
                        tuple(np.multiply(codo_i, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass

        # Mostrar video
        cv2.imshow('Detección de Técnica - Tesis', image)

        # Salir con tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
