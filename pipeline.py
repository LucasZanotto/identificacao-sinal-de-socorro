"""Etapa de visão: descobre onde olhar e extrai os pontos das mãos com o MediaPipe.

O sistema usa a detecção de movimento do OpenCV, que não tem rede neural, para procurar mãos novas.
Cada mão encontrada ganha um recorte só dela, e o modo de vídeo do MediaPipe passa a acompanhá-la sem
precisar do detector de palma, que costuma perder a mão virada, fechada ou coberta. O YOLOv8 só entra
quando pedido, para comparar o gasto dos dois na avaliação.
"""
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field

# Só faz diferença na comparação com o YOLO e precisa vir antes do PyTorch. Sem isso as threads
# dele ficam em espera ativa entre uma inferência e outra e ocupam a CPU à toa.
os.environ.setdefault("KMP_BLOCKTIME", "0")

import cv2
import mediapipe as mp
import numpy as np

from gesto import calcular_distancia, PULSO, tamanho_palma

PASTA = os.path.dirname(os.path.abspath(__file__))
MODELO_MEDIAPIPE = os.path.join(PASTA, "hand_landmarker.task")
MODELO_YOLO = os.path.join(PASTA, "yolov8n.pt")
URL_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
LUMINANCIA_MINIMA = 90  # Abaixo disso (de 0 a 255) o quadro é clareado
LUMINANCIA_ALVO = 110

TAMANHO_MOVIMENTO = (320, 180)  # O movimento é analisado em uma cópia pequena do quadro
AREA_MINIMA_MOVIMENTO = 12  # Em pixels dessa cópia pequena
MUDANCA_GLOBAL = 0.4  # Se mais do que isso do quadro mudou, foi a exposição da câmera e não movimento
MAX_RECORTES = 2  # O MediaPipe roda em no máximo tantos recortes por quadro
CONFIANCA_MAO = 0.3  # Abaixo do padrão de 0,5 para achar a mão pequena e coberta. A sequência do gesto filtra os enganos
MS_POR_QUADRO = 66  # Tempo informado ao modo de vídeo. Fixo para o resultado não depender da velocidade do computador
QUADROS_ACOMPANHANDO = 8  # Por quantos quadros mantém o recorte de uma mão que sumiu
QUADROS_SEGURANDO = 15  # Por quantos quadros mantém a última região que se mexeu
INTEIRO_A_CADA = 5  # Sem nada para olhar, analisa o quadro inteiro de tempos em tempos
# Menor lado dos recortes, como fração da largura do quadro. Não há limite de cima para não cortar uma mão perto da câmera
LADO_MINIMO_MAO = 0.17
LADO_MINIMO_MOVIMENTO = 0.20


def ajustar_iluminacao(frame):
    """Clareia o quadro com correção gama quando a cena está escura. Quadro claro passa direto."""
    media = float(cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).mean())
    if media >= LUMINANCIA_MINIMA:
        return frame
    gama = max(0.35, math.log(LUMINANCIA_ALVO / 255) / math.log(max(media, 5) / 255))
    tabela = np.clip(((np.arange(256) / 255) ** gama) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, tabela)


def recorte_centrado(cx, cy, lado, largura, altura):
    """Recorte quadrado em volta do centro. Se não couber na imagem, fica do tamanho que couber."""
    w, h = int(min(lado, largura)), int(min(lado, altura))
    x1 = int(np.clip(cx - w / 2, 0, largura - w))
    y1 = int(np.clip(cy - h / 2, 0, altura - h))
    return (x1, y1, x1 + w, y1 + h)


def _girar(lista, passo):
    if not lista:
        return lista
    k = passo % len(lista)
    return lista[k:] + lista[:k]


@dataclass
class Regiao:
    caixa: tuple  # (x1, y1, x2, y2)
    origem: str  # "mao", "movimento", "pessoa" ou "inteiro"


@dataclass
class Quadro:
    """Resultado de um quadro. A imagem em si não fica guardada aqui."""
    maos: list = field(default_factory=list)  # Lista de (pontos, lado, confiança), com os pontos em pixels
    regioes: list = field(default_factory=list)  # Onde o MediaPipe olhou neste quadro
    clareado: bool = False
    ms_localizar: float = 0.0
    ms_mediapipe: float = 0.0


class LocalizadorMovimento:
    """Compara cada quadro com um modelo do fundo da cena (MOG2) e aponta o que se mexeu."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.fundo = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
        self.mascara = None

    def caixas(self, frame):
        """Retorna as regiões em movimento como (x, y, largura, altura), da maior para a menor."""
        altura, largura = frame.shape[:2]
        mascara = self.fundo.apply(cv2.resize(frame, TAMANHO_MOVIMENTO))
        self.mascara = mascara
        if (mascara > 0).mean() > MUDANCA_GLOBAL:
            return []
        limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        limpa = cv2.dilate(limpa, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        caixas = [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= AREA_MINIMA_MOVIMENTO]
        caixas.sort(key=lambda b: -b[2] * b[3])
        ex, ey = largura / TAMANHO_MOVIMENTO[0], altura / TAMANHO_MOVIMENTO[1]
        return [(x * ex, y * ey, w * ex, h * ey) for x, y, w, h in caixas]


class JanelaMao:
    """Recorte que segue uma mão já encontrada, com um MediaPipe em modo de vídeo só para ela."""

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self.caixa = None  # Sem caixa, a janela está livre
        self.lado = 0.0
        self.sem_ver = 0


class DetectorMaos:
    def __init__(self, localizador="movimento", yolo_a_cada=3, conf_yolo=0.4, margem_roi=0.15, num_maos=2, corrigir_luz=True, rastrear=True):
        """localizador: como procurar mãos novas. "movimento" (o sistema), "yolo" (só para comparar) ou None (quadro inteiro).

        rastrear=False roda o detector de palma em todo quadro, só para medir o que o rastreamento acrescenta.
        """
        # Download do modelo oficial do MediaPipe Tasks
        if not os.path.exists(MODELO_MEDIAPIPE):
            print("Baixando arquivo do modelo Hand Landmarker do MediaPipe...")
            urllib.request.urlretrieve(URL_MEDIAPIPE, MODELO_MEDIAPIPE)

        self.localizador = localizador
        self.corrigir_luz = corrigir_luz
        self.rastrear = rastrear
        self.yolo_a_cada = yolo_a_cada
        self.conf_yolo = conf_yolo
        self.margem_roi = margem_roi
        self.movimento = LocalizadorMovimento() if localizador == "movimento" else None

        self.yolo = None
        if localizador == "yolo":
            # Importado só aqui porque o PyTorch pesa na memória e no tempo de partida
            from ultralytics import YOLO
            self.yolo = YOLO(MODELO_YOLO)

        def opcoes(modo, maos):
            return mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_MEDIAPIPE),
                running_mode=modo,
                num_hands=maos,
                min_hand_detection_confidence=CONFIANCA_MAO,
                min_hand_presence_confidence=CONFIANCA_MAO,
                min_tracking_confidence=CONFIANCA_MAO,
            )

        criar = mp.tasks.vision.HandLandmarker.create_from_options
        modos = mp.tasks.vision.RunningMode
        self.landmarker = criar(opcoes(modos.IMAGE, num_maos))
        self.janelas = [JanelaMao(criar(opcoes(modos.VIDEO, 1))) for _ in range(MAX_RECORTES)]
        self.tempo_ms = 0
        self.reiniciar()

    def fechar(self):
        # Precisa ser explícito: se o coletor de lixo do Python fechar, o MediaPipe pode travar
        self.landmarker.close()
        for janela in self.janelas:
            janela.landmarker.close()

    def reiniciar(self):
        self.contador = 0
        self.caixas_yolo = []
        self.ultima_regiao = None
        self.quadro_regiao = -QUADROS_SEGURANDO - 1
        if self.movimento is not None:
            self.movimento.reiniciar()
        # Uma imagem vazia faz o modo de vídeo esquecer a mão do vídeo anterior
        self.tempo_ms += MS_POR_QUADRO
        vazia = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), dtype=np.uint8))
        for janela in self.janelas:
            janela.caixa = None
            janela.sem_ver = 0
            janela.landmarker.detect_for_video(vazia, self.tempo_ms)

    def _regioes_por_movimento(self, frame, ocupadas):
        altura, largura = frame.shape[:2]
        regioes = []
        # Precisa rodar em todo quadro, senão o modelo do fundo fica desatualizado.
        # As regiões se revezam para que alguém andando ao fundo não esconda um gesto pequeno.
        caixas = self.movimento.caixas(frame)
        candidatas = _girar(caixas, self.contador) if ocupadas else caixas[:1] + _girar(caixas[1:], self.contador)
        for x, y, w, h in candidatas:
            if len(regioes) >= MAX_RECORTES - len(ocupadas):
                break
            cx, cy = x + w / 2, y + h / 2
            if any(j.caixa[0] <= cx <= j.caixa[2] and j.caixa[1] <= cy <= j.caixa[3] for j in ocupadas):
                continue
            lado = max(1.3 * max(w, h), LADO_MINIMO_MOVIMENTO * largura)
            regiao = Regiao(recorte_centrado(cx, cy, lado, largura, altura), "movimento")
            regioes.append(regiao)
            self.ultima_regiao, self.quadro_regiao = regiao, self.contador

        if ocupadas:
            return regioes
        if not regioes and self.contador - self.quadro_regiao <= QUADROS_SEGURANDO:
            regioes.append(self.ultima_regiao)
        # Mão parada em primeiro plano não gera movimento, mas o MediaPipe acha no quadro inteiro
        if not regioes and self.contador % INTEIRO_A_CADA == 0:
            regioes.append(Regiao((0, 0, largura, altura), "inteiro"))
        return regioes

    def _regioes_por_yolo(self, frame, ocupadas):
        altura, largura = frame.shape[:2]
        rodou = self.contador % self.yolo_a_cada == 0
        if rodou:
            # (conf ignora detecções fracas; classes=0 filtra apenas 'pessoa'; device força CPU, sem placa de vídeo)
            resultado = self.yolo(frame, conf=self.conf_yolo, classes=0, device="cpu", verbose=False)[0]
            self.caixas_yolo = []
            for box in resultado.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # A margem evita cortar a mão levantada para fora da caixa da pessoa
                mx, my = int((x2 - x1) * self.margem_roi), int((y2 - y1) * self.margem_roi)
                self.caixas_yolo.append((max(0, x1 - mx), max(0, y1 - my), min(largura, x2 + mx), min(altura, y2 + my)))
        # Com mão já acompanhada, só procura outra nos quadros em que o YOLO rodou
        if len(ocupadas) >= MAX_RECORTES or (ocupadas and not rodou):
            return []
        # Sem pessoa na cena (mão em primeiro plano, por exemplo) analisa o quadro inteiro
        if not self.caixas_yolo:
            return [Regiao((0, 0, largura, altura), "inteiro")]
        return [Regiao(c, "pessoa") for c in _girar(self.caixas_yolo, self.contador)[:MAX_RECORTES - len(ocupadas)]]

    def _regioes_no_quadro_inteiro(self, frame, ocupadas):
        altura, largura = frame.shape[:2]
        # Com mão já acompanhada, a busca por outra fica mais espaçada para não gastar à toa
        if len(ocupadas) >= MAX_RECORTES or (ocupadas and self.contador % INTEIRO_A_CADA):
            return []
        return [Regiao((0, 0, largura, altura), "inteiro")]

    def _seguir_mao(self, janela, pontos, largura, altura):
        """Centraliza o recorte na mão. Seguir a mão faz ela quase não mudar de lugar dentro do recorte, o que ajuda o rastreio."""
        xs, ys = [p[0] for p in pontos], [p[1] for p in pontos]
        alvo = max(2.4 * max(max(xs) - min(xs), max(ys) - min(ys)), LADO_MINIMO_MAO * largura)
        # O tamanho só muda quando a mão cresce ou encolhe bastante: cada mudança atrapalha o rastreio
        if alvo > janela.lado or alvo < 0.6 * janela.lado:
            janela.lado = alvo
        janela.caixa = recorte_centrado((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, janela.lado, largura, altura)

    def _abrir_janela(self, mao, largura, altura):
        janela = next((j for j in self.janelas if j.caixa is None), None)
        if janela is not None:
            janela.lado = 0.0
            janela.sem_ver = 0
            self._seguir_mao(janela, mao[0], largura, altura)

    def _acompanhar(self, frame, ocupadas, quadro):
        altura, largura = frame.shape[:2]
        maos = []
        for janela in ocupadas:
            quadro.regioes.append(Regiao(janela.caixa, "mao"))
            achadas = self._extrair_maos(frame, janela.caixa, janela)
            if not achadas:
                janela.sem_ver += 1
                if janela.sem_ver > QUADROS_ACOMPANHANDO:
                    janela.caixa = None
            elif self._ja_vista(achadas[0], maos):
                # Duas janelas foram parar na mesma mão
                janela.caixa = None
            else:
                maos.append(achadas[0])
                janela.sem_ver = 0
                self._seguir_mao(janela, achadas[0][0], largura, altura)
        return maos

    def _extrair_maos(self, frame, caixa, janela=None):
        x1, y1, x2, y2 = caixa
        recorte = frame[y1:y2, x1:x2]
        if recorte.size == 0:
            return []
        crop_h, crop_w = recorte.shape[:2]
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        if janela is None or not self.rastrear:
            resultado = self.landmarker.detect(imagem)
        else:
            resultado = janela.landmarker.detect_for_video(imagem, self.tempo_ms)
        maos = []
        for lm, lado in zip(resultado.hand_landmarks, resultado.handedness):
            # Converte a coordenada relativa do recorte para PIXELS GLOBAIS. Sem isso a
            # proporção do recorte distorce as distâncias.
            pontos = [(x1 + p.x * crop_w, y1 + p.y * crop_h, p.z * crop_w) for p in lm]
            maos.append((pontos, lado[0].category_name, lado[0].score))
        return maos

    @staticmethod
    def _ja_vista(mao, maos):
        limite = 0.5 * tamanho_palma(mao[0])
        return any(calcular_distancia(mao[0][PULSO][:2], m[0][PULSO][:2]) <= limite for m in maos)

    @staticmethod
    def _remover_repetidas(maos):
        # Recortes sobrepostos podem entregar a mesma mão duas vezes
        unicas = []
        for mao in sorted(maos, key=lambda m: -m[2]):
            if not DetectorMaos._ja_vista(mao, unicas):
                unicas.append(mao)
        return unicas

    def processar(self, frame):
        quadro = Quadro()
        self.tempo_ms += MS_POR_QUADRO

        # 1. ETAPA DE LOCALIZAÇÃO: decide em que parte da imagem procurar mãos novas
        inicio = time.perf_counter()
        if self.corrigir_luz:
            claro = ajustar_iluminacao(frame)
            quadro.clareado = claro is not frame
            frame = claro
        altura, largura = frame.shape[:2]
        ocupadas = [j for j in self.janelas if j.caixa is not None]
        if self.localizador == "movimento":
            busca = self._regioes_por_movimento(frame, ocupadas)
        elif self.localizador == "yolo":
            busca = self._regioes_por_yolo(frame, ocupadas)
        else:
            busca = self._regioes_no_quadro_inteiro(frame, ocupadas)
        quadro.ms_localizar = (time.perf_counter() - inicio) * 1000

        # 2. ETAPA DO MEDIAPIPE: acompanha as mãos que já conhece e procura as novas
        inicio = time.perf_counter()
        maos = self._acompanhar(frame, ocupadas, quadro)
        for regiao in busca:
            quadro.regioes.append(regiao)
            for mao in self._extrair_maos(frame, regiao.caixa):
                if not self._ja_vista(mao, maos):
                    maos.append(mao)
                    self._abrir_janela(mao, largura, altura)
        quadro.maos = self._remover_repetidas(maos)
        quadro.ms_mediapipe = (time.perf_counter() - inicio) * 1000

        self.contador += 1
        return quadro
