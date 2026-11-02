import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesto import calcular_distancia
from pipeline import DetectorMaos, INTEIRO_A_CADA, LUMINANCIA_MINIMA, LocalizadorMovimento, MAX_RECORTES, QUADROS_ACOMPANHANDO, ajustar_iluminacao, recorte_centrado
import mao_sintetica as ms


def fundo(altura=720, largura=1280):
    return np.full((altura, largura, 3), 200, dtype=np.uint8)


def com_mancha(x, y, tamanho=60, cor=30):
    frame = fundo()
    frame[y:y + tamanho, x:x + tamanho] = cor
    return frame


@pytest.fixture(scope="module")
def detector():
    d = DetectorMaos()
    yield d
    d.fechar()


def preparar(detector, resultados=None):
    """Troca o MediaPipe por uma resposta pronta, para testar só a escolha das regiões."""
    detector.reiniciar()
    chamadas = []

    def extrair_falso(frame, caixa, janela=None):
        chamadas.append(caixa)
        return resultados.pop(0) if resultados else []

    detector._extrair_maos = extrair_falso
    return chamadas


def test_quadro_claro_passa_sem_alteracao():
    claro = np.full((720, 1280, 3), 150, dtype=np.uint8)
    assert ajustar_iluminacao(claro) is claro


def test_quadro_escuro_fica_mais_claro():
    escuro = np.full((720, 1280, 3), 25, dtype=np.uint8)
    resultado = ajustar_iluminacao(escuro)
    assert resultado.mean() > LUMINANCIA_MINIMA
    assert resultado.shape == escuro.shape


def test_correcao_aguenta_quadro_totalmente_preto():
    preto = np.zeros((36, 64, 3), dtype=np.uint8)
    assert ajustar_iluminacao(preto).max() == 0


def test_mesma_mao_em_caixas_sobrepostas_conta_uma_vez():
    mao = ms.polegar_dobrado(x=500.0)
    deslocada = ms.polegar_dobrado(x=510.0)
    outra = ms.polegar_dobrado(x=900.0)
    unicas = DetectorMaos._remover_repetidas([(deslocada, "Left", 0.7), (mao, "Left", 0.9), (outra, "Left", 0.8)])
    assert len(unicas) == 2
    # Fica a leitura com maior confiança
    assert unicas[0][0] is mao
    assert calcular_distancia(unicas[0][0][0], unicas[1][0][0]) > 300


@pytest.mark.parametrize("cx, cy", [(640, 360), (0, 0), (1280, 720)])
def test_recorte_fica_dentro_da_imagem(cx, cy):
    x1, y1, x2, y2 = recorte_centrado(cx, cy, 300, 1280, 720)
    assert x2 - x1 == 300 and y2 - y1 == 300
    assert 0 <= x1 and x2 <= 1280 and 0 <= y1 and y2 <= 720


def test_recorte_maior_que_a_imagem_vira_a_imagem_inteira():
    assert recorte_centrado(100, 100, 5000, 640, 360) == (0, 0, 640, 360)


def test_mao_perto_da_camera_nao_e_cortada(detector):
    # Mão grande em um vídeo em pé, como em um celular: o recorte precisa conter a mão inteira
    mao = ms.polegar_dobrado(x=540.0, y=1200.0, escala=520.0)
    preparar(detector)
    frame = np.full((1920, 1080, 3), 200, dtype=np.uint8)
    for _ in range(3):
        detector.processar(frame)
    detector._abrir_janela((mao, "Left", 0.9), 1080, 1920)
    x1, y1, x2, y2 = detector.processar(frame).regioes[0].caixa
    assert x1 <= min(p[0] for p in mao) and max(p[0] for p in mao) <= x2
    assert y1 <= min(p[1] for p in mao) and max(p[1] for p in mao) <= y2


def test_movimento_aponta_onde_a_mancha_se_mexeu():
    local = LocalizadorMovimento()
    for _ in range(10):
        local.caixas(fundo())
    x, y, w, h = local.caixas(com_mancha(800, 300))[0]
    assert x <= 830 <= x + w
    assert y <= 330 <= y + h


def test_cena_parada_nao_gera_movimento():
    local = LocalizadorMovimento()
    for _ in range(10):
        caixas = local.caixas(fundo())
    assert caixas == []


def test_mudanca_de_exposicao_nao_conta_como_movimento():
    local = LocalizadorMovimento()
    for _ in range(10):
        local.caixas(fundo())
    assert local.caixas(np.full((720, 1280, 3), 90, dtype=np.uint8)) == []


def test_sem_movimento_olha_o_quadro_inteiro_de_tempos_em_tempos(detector):
    preparar(detector)
    origens = [[r.origem for r in detector.processar(fundo()).regioes] for _ in range(2 * INTEIRO_A_CADA)]
    assert origens.count(["inteiro"]) == 2
    assert origens.count([]) == 2 * INTEIRO_A_CADA - 2


def test_regiao_em_movimento_vai_para_o_mediapipe(detector):
    chamadas = preparar(detector)
    for _ in range(10):
        detector.processar(fundo())
    chamadas.clear()
    quadro = detector.processar(com_mancha(800, 300))
    assert [r.origem for r in quadro.regioes] == ["movimento"]
    x1, y1, x2, y2 = chamadas[0]
    assert x1 <= 830 <= x2 and y1 <= 330 <= y2


def test_guarda_a_ultima_regiao_quando_o_movimento_para(detector):
    preparar(detector)
    for _ in range(10):
        detector.processar(fundo())
    detector.processar(com_mancha(800, 300))
    # A mancha fica parada no mesmo lugar e a região continua valendo por um tempo
    quadro = detector.processar(com_mancha(800, 300))
    assert quadro.regioes and quadro.regioes[0].origem == "movimento"


def test_mao_vista_continua_acompanhada_mesmo_parada(detector):
    mao = ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0)
    resultados = []
    preparar(detector, resultados)
    for _ in range(10):
        detector.processar(fundo())
    resultados.append([(mao, "Left", 0.9)])
    detector._abrir_janela((mao, "Left", 0.9), 1280, 720)
    quadro = detector.processar(fundo())
    assert [r.origem for r in quadro.regioes] == ["mao"]
    x1, y1, x2, y2 = quadro.regioes[0].caixa
    assert x1 <= mao[0][0] <= x2 and y1 <= mao[0][1] <= y2


def test_mao_some_e_deixa_de_ser_acompanhada(detector):
    mao = ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0)
    preparar(detector)
    for _ in range(10):
        detector.processar(fundo())
    detector._abrir_janela((mao, "Left", 0.9), 1280, 720)
    for _ in range(QUADROS_ACOMPANHANDO + 2):
        quadro = detector.processar(fundo())
    assert all(r.origem != "mao" for r in quadro.regioes)
    assert all(j.caixa is None for j in detector.janelas)


def test_mao_some_por_poucos_quadros_continua_com_a_janela(detector):
    mao = ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0)
    preparar(detector)
    detector._abrir_janela((mao, "Left", 0.9), 1280, 720)
    for _ in range(QUADROS_ACOMPANHANDO - 1):
        quadro = detector.processar(fundo())
    assert [r.origem for r in quadro.regioes] == ["mao"]


def test_mao_encontrada_na_busca_passa_a_ser_acompanhada(detector):
    mao = ms.polegar_dobrado(x=830.0, y=330.0, escala=60.0)
    chamadas = []
    resposta = []

    def extrair_falso(frame, caixa, janela=None):
        chamadas.append(janela)
        return list(resposta)

    preparar(detector)
    detector._extrair_maos = extrair_falso
    for _ in range(10):
        detector.processar(fundo())
    resposta.append((mao, "Left", 0.9))
    chamadas.clear()
    primeiro = detector.processar(com_mancha(800, 300))
    assert [r.origem for r in primeiro.regioes] == ["movimento"]
    assert chamadas == [None]
    chamadas.clear()
    # A mancha continua no mesmo lugar: quem olha agora é a janela da mão, em modo de vídeo
    segundo = detector.processar(com_mancha(800, 300))
    assert [r.origem for r in segundo.regioes] == ["mao"]
    assert chamadas == [detector.janelas[0]]
    assert len(segundo.maos) == 1


def test_janela_segue_a_mao_que_anda(detector):
    resultados = [[(ms.polegar_dobrado(x=500.0 + 40 * i, y=400.0, escala=60.0), "Left", 0.9)] for i in range(5)]
    preparar(detector, resultados)
    detector._abrir_janela((ms.polegar_dobrado(x=500.0, y=400.0, escala=60.0), "Left", 0.9), 1280, 720)
    centros = []
    for _ in range(5):
        detector.processar(fundo())
        x1, _, x2, _ = detector.janelas[0].caixa
        centros.append((x1 + x2) / 2)
    assert centros == sorted(centros)
    assert centros[-1] - centros[0] == pytest.approx(160, abs=2)


def test_janela_so_muda_de_tamanho_quando_a_mao_muda_bastante(detector):
    grande = ms.polegar_dobrado(x=640.0, y=500.0, escala=300.0)
    preparar(detector, [[(ms.polegar_dobrado(x=640.0, y=500.0, escala=240.0), "Left", 0.9)], [(ms.polegar_dobrado(x=640.0, y=500.0, escala=120.0), "Left", 0.9)]])
    detector._abrir_janela((grande, "Left", 0.9), 1920, 1080)
    lado = detector.janelas[0].lado
    # Encolheu um pouco, como a mão que fecha: o recorte continua do mesmo tamanho
    detector.processar(np.full((1080, 1920, 3), 200, dtype=np.uint8))
    assert detector.janelas[0].lado == lado
    # Encolheu muito, como a pessoa que se afasta: o recorte acompanha
    detector.processar(np.full((1080, 1920, 3), 200, dtype=np.uint8))
    assert detector.janelas[0].lado < 0.6 * lado


def test_duas_janelas_na_mesma_mao_viram_uma(detector):
    mao = ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0)
    preparar(detector, [[(mao, "Left", 0.9)], [(mao, "Left", 0.9)]])
    detector._abrir_janela((mao, "Left", 0.9), 1280, 720)
    detector._abrir_janela((ms.polegar_dobrado(x=905.0, y=400.0, escala=60.0), "Left", 0.9), 1280, 720)
    quadro = detector.processar(fundo())
    assert len(quadro.maos) == 1
    assert sum(j.caixa is not None for j in detector.janelas) == 1


def test_com_todas_as_janelas_ocupadas_nao_procura_outras_maos(detector):
    primeira = ms.polegar_dobrado(x=300.0, y=400.0, escala=60.0)
    segunda = ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0)
    resultados = []
    preparar(detector, resultados)
    for _ in range(10):
        detector.processar(fundo())
    resultados.extend([[(primeira, "Left", 0.9)], [(segunda, "Left", 0.9)]])
    detector._abrir_janela((primeira, "Left", 0.9), 1280, 720)
    detector._abrir_janela((segunda, "Left", 0.9), 1280, 720)
    quadro = detector.processar(com_mancha(100, 600))
    assert [r.origem for r in quadro.regioes] == ["mao"] * MAX_RECORTES


def test_movimento_dentro_da_janela_nao_gera_outra_busca(detector):
    chamadas = preparar(detector)
    for _ in range(10):
        detector.processar(fundo())
    detector._abrir_janela((ms.polegar_dobrado(x=830.0, y=330.0, escala=60.0), "Left", 0.9), 1280, 720)
    chamadas.clear()
    quadro = detector.processar(com_mancha(800, 280))
    assert [r.origem for r in quadro.regioes] == ["mao"]
    assert len(chamadas) == 1


@pytest.mark.parametrize("rastrear, esperado", [(True, "video"), (False, "imagem")])
def test_rastrear_decide_o_modo_do_mediapipe_na_janela(monkeypatch, rastrear, esperado):
    # Sem rastrear é a comparação da avaliação: mesmo recorte, mas com o detector de palma em todo quadro
    detector = DetectorMaos(rastrear=rastrear)
    usados = []

    def espiar(nome, original):
        def chamada(*argumentos):
            usados.append(nome)
            return original(*argumentos)
        return chamada

    monkeypatch.setattr(detector.landmarker, "detect", espiar("imagem", detector.landmarker.detect))
    monkeypatch.setattr(detector.janelas[0].landmarker, "detect_for_video", espiar("video", detector.janelas[0].landmarker.detect_for_video))
    detector.tempo_ms += 66
    detector._extrair_maos(fundo(), (0, 0, 200, 200), detector.janelas[0])
    detector.fechar()
    assert usados == [esperado]


def test_reiniciar_libera_as_janelas(detector):
    preparar(detector)
    detector._abrir_janela((ms.polegar_dobrado(x=900.0, y=400.0, escala=60.0), "Left", 0.9), 1280, 720)
    detector.reiniciar()
    assert all(j.caixa is None for j in detector.janelas)


def test_movimento_grande_nao_esconde_gesto_pequeno(detector):
    """Duas pessoas andando e uma mão pequena: em poucos quadros a mão também é analisada."""
    preparar(detector)
    for _ in range(10):
        detector.processar(fundo())
    pequena_vista = False
    for i in range(12):
        frame = fundo()
        frame[100:400, 100 + 5 * i:300 + 5 * i] = 30
        frame[300:600, 500 + 5 * i:700 + 5 * i] = 60
        # A mão muda de um lado para o outro a cada quadro para continuar sendo movimento
        frame[100:140, 1100 + (i % 2) * 30:1140 + (i % 2) * 30] = 90
        quadro = detector.processar(frame)
        assert len(quadro.regioes) <= MAX_RECORTES
        pequena_vista = pequena_vista or any(r.caixa[2] > 1100 for r in quadro.regioes)
    assert pequena_vista


def test_sem_localizador_olha_sempre_o_quadro_inteiro():
    detector = DetectorMaos(localizador=None)
    preparar(detector)
    quadro = detector.processar(fundo())
    detector.fechar()
    assert [r.origem for r in quadro.regioes] == ["inteiro"]
    assert quadro.regioes[0].caixa == (0, 0, 1280, 720)
