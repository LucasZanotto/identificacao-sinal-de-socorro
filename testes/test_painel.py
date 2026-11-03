import os
import runpy
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detecao_socorro
from alerta import Alerta
from gesto import Configuracao, RastreadorMaos
from painel import ALTURA, CINZA, ESPERANDO, LARANJA, LARGURA_LATERAL, Painel, VERDE
from pipeline import Quadro, Regiao
import mao_sintetica as ms

MEDIDAS = {"fps": 25.0, "ms_total": 38.0, "ms_localizar": 1.5, "ms_mediapipe": 30.0, "ms_gesto": 0.05}
REGIOES = [Regiao((100, 100, 500, 500), "mao"), Regiao((600, 100, 900, 400), "movimento"), Regiao((0, 0, 1280, 720), "inteiro")]


def frame(largura=1280, altura=720):
    return np.full((altura, largura, 3), 180, dtype=np.uint8)


def montar(tmp_path):
    rastreador = RastreadorMaos(Configuracao())
    return rastreador, Painel(rastreador.cfg), Alerta(caminho_log=str(tmp_path / "alertas.csv"), com_som=False)


def desenhar(painel, rastreador, alerta, tela=None, tempo=1.0, mascara=None, regioes=None, clareado=False):
    quadro = Quadro(regioes=regioes or [], clareado=clareado)
    return painel.desenhar(frame() if tela is None else tela, quadro, rastreador, alerta, mascara, MEDIDAS, tempo)


def test_tela_sem_nenhuma_mao(tmp_path):
    rastreador, painel, alerta = montar(tmp_path)
    tela = desenhar(painel, rastreador, alerta)
    assert tela.shape == (ALTURA, 1280 + LARGURA_LATERAL, 3)


@pytest.mark.parametrize("pontos", [ms.aberta, ms.polegar_dobrado, ms.punho, ms.meio_fechada])
def test_tela_com_cada_pose(tmp_path, pontos):
    rastreador, painel, alerta = montar(tmp_path)
    for i in range(3):
        rastreador.atualizar([(pontos(), "Left")], i * 0.1)
    mascara = np.zeros((180, 320), dtype=np.uint8)
    antes = frame()
    tela = desenhar(painel, rastreador, alerta, tela=antes, tempo=0.3, mascara=mascara, regioes=REGIOES, clareado=True)
    assert tela.shape[0] == ALTURA
    # O vídeo original continua limpo: só a cópia recebe os desenhos
    assert (antes == 180).all()


def tem_laranja(img):
    # A tolerância pega também a borda suavizada do texto
    return bool((np.abs(img.astype(int) - LARANJA).sum(axis=2) < 60).any())


def test_versao_limpa_mostra_so_o_quadrado_da_mao(tmp_path):
    rastreador, painel, alerta = montar(tmp_path)
    tela = desenhar(painel, rastreador, alerta, regioes=REGIOES)
    assert (tela[300, 100] == VERDE).all()
    assert (tela[300, 600] == 180).all()
    assert (tela[300, 0] == 180).all()
    assert not tem_laranja(tela)


def test_versao_profunda_mostra_tudo_o_que_o_sistema_examina(tmp_path):
    rastreador, _, alerta = montar(tmp_path)
    tela = desenhar(Painel(rastreador.cfg, profundo=True), rastreador, alerta, regioes=REGIOES)
    assert (tela[300, 100] == VERDE).all()
    assert (tela[300, 600] == LARANJA).all()
    assert (tela[300, 0] == CINZA).all()
    assert tem_laranja(tela[:, 1280:])


@pytest.mark.parametrize("largura, altura", [(405, 720), (1080, 1920), (1920, 1080), (640, 480)])
def test_tela_se_ajusta_a_qualquer_tamanho_de_video(tmp_path, largura, altura):
    rastreador, painel, alerta = montar(tmp_path)
    tela = desenhar(painel, rastreador, alerta, tela=frame(largura, altura))
    assert tela.shape == (ALTURA, int(largura * ALTURA / altura) + LARGURA_LATERAL, 3)


def test_alerta_aparece_na_tela_e_nao_passa_da_largura(tmp_path):
    rastreador, painel, alerta = montar(tmp_path)
    sem = desenhar(painel, rastreador, alerta, tela=frame(405, 720))
    alerta.disparar(1, ms.aberta())
    com = desenhar(painel, rastreador, alerta, tela=frame(405, 720))
    assert not np.array_equal(sem, com)
    assert (com[10, 10] == (0, 0, 255)).all()


def test_eventos_contam_a_historia_do_gesto(tmp_path, capsys):
    rastreador, painel, alerta = montar(tmp_path)

    def passo(pontos, tempo):
        concluidas = rastreador.atualizar([(pontos, "Left")], tempo)
        painel.registrar(rastreador, concluidas, tempo)

    tempo = 0.0
    for pontos in [ms.aberta()] * 3 + [ms.polegar_dobrado()] * 3 + [ms.punho()] * 3:
        passo(pontos, tempo)
        tempo += 0.1
    saida = capsys.readouterr().out
    assert "Mao #1 apareceu" in saida
    assert "Repouso -> Mao aberta" in saida
    assert "Mao aberta -> Polegar recolhido" in saida
    assert "GESTO CONCLUIDO" in saida


def test_eventos_explicam_por_que_a_sequencia_voltou(tmp_path, capsys):
    rastreador, painel, alerta = montar(tmp_path)
    tempo = 0.0
    for pontos in [ms.aberta()] * 3 + [ms.punho()] * 3:
        painel.registrar(rastreador, rastreador.atualizar([(pontos, "Left")], tempo), tempo)
        tempo += 0.1
    assert "fechou sem o polegar" in capsys.readouterr().out

    rastreador, painel, alerta = montar(tmp_path)
    tempo = 0.0
    for pontos in [ms.aberta()] * 3 + [ms.meio_fechada()] * 40:
        painel.registrar(rastreador, rastreador.atualizar([(pontos, "Left")], tempo), tempo)
        tempo += 0.1
    assert "tempo esgotado" in capsys.readouterr().out


def test_evento_de_mao_que_sai_de_vista(tmp_path, capsys):
    rastreador, painel, alerta = montar(tmp_path)
    painel.registrar(rastreador, rastreador.atualizar([(ms.aberta(), "Left")], 0.0), 0.0)
    painel.registrar(rastreador, rastreador.atualizar([], 2.0), 2.0)
    assert "saiu de vista" in capsys.readouterr().out
    assert painel.estados == {}


def test_toda_etapa_tem_instrucao():
    assert set(ESPERANDO) == {0, 1, 2}


def criar_video(caminho, quadros=20, fps=20.0):
    escritor = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(quadros):
        escritor.write(np.full((240, 320, 3), 50 + i, dtype=np.uint8))
    escritor.release()


class CameraFalsa:
    """Entrega um quadro a cada 5 ms com o número dele na imagem. Pode falhar para simular queda."""
    abertas = 0

    def __init__(self, fonte, falhar_depois=None):
        CameraFalsa.abertas += 1
        self.contador = 0
        self.falhar_depois = falhar_depois if CameraFalsa.abertas == 1 else None

    def set(self, *a):
        return True

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.005)
        self.contador += 1
        if self.falhar_depois is not None and self.contador > self.falhar_depois:
            return False, None
        return True, np.full((10, 10, 3), min(self.contador, 255), dtype=np.uint8)

    def release(self):
        pass


def test_camera_ao_vivo_entrega_so_o_quadro_mais_recente(monkeypatch):
    CameraFalsa.abertas = 0
    monkeypatch.setattr(cv2, "VideoCapture", CameraFalsa)
    camera = detecao_socorro.CameraAoVivo(0)
    sucesso, primeiro = camera.ler()
    assert sucesso
    # Como a leitura corre em paralelo, ler devagar pula quadros em vez de acumular atraso
    time.sleep(0.2)
    sucesso, depois = camera.ler()
    camera.liberar()
    assert sucesso
    assert depois[0, 0, 0] >= primeiro[0, 0, 0] + 10


def test_camera_que_cai_conecta_de_novo(monkeypatch):
    CameraFalsa.abertas = 0
    monkeypatch.setattr(cv2, "VideoCapture", lambda fonte: CameraFalsa(fonte, falhar_depois=3))
    monkeypatch.setattr(detecao_socorro, "FALHAS_PARA_RECONECTAR", 3)
    monkeypatch.setattr(detecao_socorro, "ESPERA_RECONEXAO", 0.01)
    camera = detecao_socorro.CameraAoVivo("rtsp://camera/teste")
    assert camera.ler()[0]
    time.sleep(0.5)
    camera.liberar()
    assert CameraFalsa.abertas >= 2


def test_camera_sem_imagem_nao_trava_para_sempre(monkeypatch):
    CameraFalsa.abertas = 0
    monkeypatch.setattr(cv2, "VideoCapture", lambda fonte: CameraFalsa(fonte, falhar_depois=0))
    monkeypatch.setattr(detecao_socorro, "FALHAS_PARA_RECONECTAR", 1000)
    camera = detecao_socorro.CameraAoVivo("rtsp://camera/teste")
    assert camera.ler(espera=0.05) == (False, None)
    camera.liberar()


def test_programa_completo_em_um_arquivo_com_a_tela(tmp_path, monkeypatch):
    video = str(tmp_path / "video.mp4")
    criar_video(video)
    mostradas = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda _, tela: mostradas.append(tela.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda _: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["detecao", "--fonte", video, "--sem-som"])
    detecao_socorro.main()
    assert len(mostradas) == 20
    assert mostradas[0] == (ALTURA, 960 + LARGURA_LATERAL, 3)


def test_programa_profundo_abre_a_mesma_tela_na_janela_dele(tmp_path, monkeypatch):
    video = str(tmp_path / "video.mp4")
    criar_video(video)
    janelas = set()
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda nome, _: janelas.add(nome))
    monkeypatch.setattr(cv2, "waitKey", lambda _: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["detecao", "--fonte", video, "--sem-som"])
    programa = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detecao_socorro_profundo.py")
    runpy.run_path(programa, run_name="__main__")
    assert janelas == {detecao_socorro.JANELA_PROFUNDA}


def test_q_encerra_o_programa(tmp_path, monkeypatch):
    video = str(tmp_path / "video.mp4")
    criar_video(video)
    mostradas = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda _, tela: mostradas.append(1))
    monkeypatch.setattr(cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["detecao", "--fonte", video, "--sem-som"])
    detecao_socorro.main()
    assert len(mostradas) == 1


def test_fonte_invalida_avisa(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detecao", "--fonte", str(tmp_path / "nao_existe.mp4"), "--sem-janela"])
    with pytest.raises(SystemExit) as erro:
        detecao_socorro.main()
    assert "Não foi possível abrir" in str(erro.value)
