"""Roda o sistema inteiro (movimento, MediaPipe, geometria e MEF) num vídeo real do gesto.

Vídeo do Wikimedia Commons, autor GemoWiki, licença CC BY 4.0, "Signal for Help( demonstrative
example )". Ele fica em testes/dados e não vem com o repositório, é baixado na primeira execução.
Mostra uma pessoa fazendo o sinal duas vezes.
"""
import importlib.util
import os
import sys
import urllib.request

import cv2
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from gesto import RastreadorMaos
from pipeline import DetectorMaos

URL = "https://upload.wikimedia.org/wikipedia/commons/5/59/Signal_for_Help%28_demonstrative_example_%29.ogg"
VIDEO = os.path.join(RAIZ, "testes", "dados", "signal_for_help_wikimedia.ogg")
LOCALIZADORES = ["movimento", None]
if importlib.util.find_spec("ultralytics"):
    LOCALIZADORES.append("yolo")


@pytest.fixture(scope="module")
def video():
    if not os.path.exists(VIDEO):
        os.makedirs(os.path.dirname(VIDEO), exist_ok=True)
        pedido = urllib.request.Request(URL, headers={"User-Agent": "tcc-unoesc/1.0"})
        try:
            with urllib.request.urlopen(pedido, timeout=60) as resposta, open(VIDEO, "wb") as arquivo:
                arquivo.write(resposta.read())
        except OSError as erro:
            pytest.skip(f"Sem o vídeo de teste e sem como baixar: {erro}")
    return VIDEO


def alertas_no_video(caminho, localizador, espelhar=False):
    detector = DetectorMaos(localizador=localizador)
    rastreador = RastreadorMaos()
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS)
    alertas = []
    indice = 0
    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break
        if espelhar:
            frame = cv2.flip(frame, 1)
        tempo = indice / fps
        quadro = detector.processar(frame)
        if rastreador.atualizar([(p, l) for p, l, _ in quadro.maos], tempo):
            alertas.append(tempo)
        indice += 1
    cap.release()
    detector.fechar()
    return alertas


@pytest.mark.parametrize("localizador", LOCALIZADORES, ids=[str(l) for l in LOCALIZADORES])
@pytest.mark.parametrize("espelhar", [False, True], ids=["normal", "espelhado"])
def test_reconhece_as_duas_vezes_que_a_pessoa_faz_o_sinal(video, localizador, espelhar):
    alertas = alertas_no_video(video, localizador, espelhar)
    assert len(alertas) == 2
    # A pessoa fecha a mão perto de 1,3 s e de 4,1 s e abre o polegar só depois de 6 s
    assert 1.0 < alertas[0] < 2.5
    assert 3.7 < alertas[1] < 5.0
