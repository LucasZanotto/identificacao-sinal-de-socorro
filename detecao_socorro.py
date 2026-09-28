import cv2
import mediapipe as mp
import math
import time
import os
import urllib.request
from ultralytics import YOLO

# 1. Download do modelo oficial do MediaPipe Tasks
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Baixando arquivo do modelo Hand Landmarker do MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download concluído com sucesso!")

# 2. Carrega o YOLOv8 (Modelo Nano: ultraleve para tempo real)
print("Carregando modelo YOLOv8 para detecção de pessoas...")
yolo_model = YOLO("yolov8n.pt") 

# 3. Configuração do MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2 # Busca até 2 mãos por pessoa detectada pelo YOLO
)

landmarker = HandLandmarker.create_from_options(options)

# Funções Geométricas (Lógica original do Protótipo 2)
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def dedo_estendido(landmarks, tip_idx, pip_idx, wrist_idx=0):
    dist_tip_wrist = calcular_distancia(landmarks[tip_idx], landmarks[wrist_idx])
    dist_pip_wrist = calcular_distancia(landmarks[pip_idx], landmarks[wrist_idx])
    return dist_tip_wrist > dist_pip_wrist

def polegar_recolhido(landmarks):
    polegar_tip = landmarks[4]
    mcp_indicador = landmarks[5]
    mcp_minimo = landmarks[17]
    dist_polegar_minimo = calcular_distancia(polegar_tip, mcp_minimo)
    largura_palma = calcular_distancia(mcp_indicador, mcp_minimo)
    return dist_polegar_minimo < (largura_palma * 1.2)

# Classe para Rastreamento e MEF Individual
class RastreioMao:
    def __init__(self, hand_id, wrist_pos_global):
        self.id = hand_id
        self.wrist = wrist_pos_global
        self.estado = 0
        self.tempo_ultimo_estado = time.time()
        self.ultimo_visto = time.time()

    def atualizar_posicao(self, wrist_pos_global):
        self.wrist = wrist_pos_global
        self.ultimo_visto = time.time()

maos_rastreadas = []
proximo_id_mao = 1
TIMEOUT_ESTADO = 3.0
DISTANCA_MAX_TRACKING = 100 # Em pixels na imagem global
alerta_geral_disparado = False
tempo_exibicao_alerta = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando Monitoramento Híbrido: YOLOv8 (ROI) + MediaPipe (Gestos)...")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    tempo_agora = time.time()

    # 1. ETAPA YOLO: Detecta Pessoas na cena
    # (conf=0.4 ignora detecções fracas; classes=0 filtra apenas 'pessoa')
    yolo_results = yolo_model(frame, conf=0.4, classes=0, verbose=False)[0]

    # Limpeza de mãos ausentes
    maos_rastreadas = [m for m in maos_rastreadas if (tempo_agora - m.ultimo_visto) < 1.0]

    # 2. ETAPA DE RECORTE (ROI) E MEDIAPIPE
    for box in yolo_results.boxes:
        # Coordenadas da pessoa na imagem
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Garante limites válidos na imagem
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)

        # Desenha a caixa da pessoa (Azul)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, "Pessoa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # Recorta a região da pessoa (Region of Interest - ROI)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        crop_h, crop_w, _ = person_crop.shape
        crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Passa a imagem RECORTADA para o MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        resultado = landmarker.detect(mp_image)

        if resultado.hand_landmarks:
            for lm in resultado.hand_landmarks:
                # Converte coordenada relativa do recorte para PIXELS GLOBAIS da tela
                pulso_crop = lm[0]
                pulso_global_x = int(x1 + (pulso_crop.x * crop_w))
                pulso_global_y = int(y1 + (pulso_crop.y * crop_h))
                pulso_global = (pulso_global_x, pulso_global_y)

                # Associação por proximidade espacial (Pixels)
                mao_encontrada = None
                menor_distancia = float('inf')

                for obj_mao in maos_rastreadas:
                    dist = math.hypot(pulso_global[0] - obj_mao.wrist[0], pulso_global[1] - obj_mao.wrist[1])
                    if dist < DISTANCA_MAX_TRACKING and dist < menor_distancia:
                        menor_distancia = dist
                        mao_encontrada = obj_mao

                if mao_encontrada is None:
                    mao_encontrada = RastreioMao(proximo_id_mao, pulso_global)
                    maos_rastreadas.append(mao_encontrada)
                    proximo_id_mao += 1
                else:
                    mao_encontrada.atualizar_posicao(pulso_global)

                # Reset por Timeout
                if mao_encontrada.estado > 0 and (tempo_agora - mao_encontrada.tempo_ultimo_estado) > TIMEOUT_ESTADO:
                    mao_encontrada.estado = 0

                # Lógica de Classificação dos Dedos (Protótipo 2)
                indicador = dedo_estendido(lm, 8, 6)
                medio = dedo_estendido(lm, 12, 10)
                anelar = dedo_estendido(lm, 16, 14)
                minimo = dedo_estendido(lm, 20, 18)
                
                quatro_dedos_estendidos = indicador and medio and anelar and minimo
                quatro_dedos_dobrados = (not indicador) and (not medio) and (not anelar) and (not minimo)
                pol_recolhido = polegar_recolhido(lm)

                # Transição da MEF
                if mao_encontrada.estado == 0:
                    if quatro_dedos_estendidos and not pol_recolhido:
                        mao_encontrada.estado = 1
                        mao_encontrada.tempo_ultimo_estado = tempo_agora
                elif mao_encontrada.estado == 1:
                    if quatro_dedos_estendidos and pol_recolhido:
                        mao_encontrada.estado = 2
                        mao_encontrada.tempo_ultimo_estado = tempo_agora
                elif mao_encontrada.estado == 2:
                    if quatro_dedos_dobrados and pol_recolhido:
                        mao_encontrada.estado = 3
                        alerta_geral_disparado = True
                        tempo_exibicao_alerta = tempo_agora + 4.0
                        mao_encontrada.estado = 0

                # Desenha os pontos da mão projetados de volta na imagem principal
                for point in lm:
                    pt_x = int(x1 + (point.x * crop_w))
                    pt_y = int(y1 + (point.y * crop_h))
                    cv2.circle(frame, (pt_x, pt_y), 3, (0, 255, 0), -1)

                info_texto = f"Mao #{mao_encontrada.id} | Est: {mao_encontrada.estado}"
                cv2.putText(frame, info_texto, (pulso_global_x - 30, pulso_global_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Exibição do Alerta Geral
    if alerta_geral_disparado:
        if time.time() < tempo_exibicao_alerta:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 255), -1)
            cv2.putText(frame, "!!! ALERTA: SINAL DE SOCORRO DETECTADO !!!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        else:
            alerta_geral_disparado = False

    cv2.imshow("Sistema Hibrido de Monitoramento (YOLO + MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()