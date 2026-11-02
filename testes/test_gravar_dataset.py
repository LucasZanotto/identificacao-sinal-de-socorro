import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gravar_dataset as gd


def criar_video(caminho, quadros=300, fps=30.0):
    escritor = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(quadros):
        escritor.write(np.full((240, 320, 3), i % 255, dtype=np.uint8))
    escritor.release()


def test_roteiro_tem_a_quantidade_esperada_por_pessoa():
    clipes = gd.listar_clipes()
    assert len(clipes) == 62
    assert sum(1 for c in clipes if c[2] == gd.ACAO_POSITIVA) == 33
    assert {c[0] for c in clipes} == {"plena", "baixa", "contraluz", "oclusao", "brusco", "longe", "virada", "costas", "manga_longe", "camera_alta"}
    assert all(c[2] in gd.INSTRUCOES for c in clipes)


def test_registrar_pessoa_atualiza_sem_apagar_as_outras(tmp_path):
    gd.registrar_pessoa(str(tmp_path), "P01", "II")
    gd.registrar_pessoa(str(tmp_path), "P02", "V")
    gd.registrar_pessoa(str(tmp_path), "P01", "III")
    texto = (tmp_path / "pessoas.csv").read_text(encoding="utf-8").split()
    assert texto == ["pessoa,fitzpatrick", "P01,III", "P02,V"]


def test_gravar_clipe_gera_video_legivel(tmp_path):
    origem = str(tmp_path / "origem.mp4")
    criar_video(origem)
    cap = cv2.VideoCapture(origem)
    destino = str(tmp_path / "P01" / "plena" / "socorro_01.mp4")
    total = gd.gravar_clipe(cap, destino, duracao=5.0, fps=30.0)
    cap.release()
    assert total > 0
    lido = cv2.VideoCapture(destino)
    assert int(lido.get(cv2.CAP_PROP_FRAME_COUNT)) == total
    lido.release()


def test_sessao_completa_com_interrupcao_e_retomada(tmp_path, monkeypatch):
    """Simula as teclas: grava dois clipes, refaz o último, sai e depois retoma de onde parou."""
    origem = str(tmp_path / "camera.mp4")
    criar_video(origem, quadros=5000)
    dataset = str(tmp_path / "dataset")
    monkeypatch.setattr(gd, "DURACAO_CLIPE", 0.02)
    monkeypatch.setattr(gd, "CONTAGEM", 0)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    def rodar(teclas):
        fila = list(teclas)

        def tecla_falsa(_):
            # Só o laço de espera da tela de instruções olha a tecla: as outras chamadas ignoram
            return ord(fila.pop(0)) if fila else ord("q")

        monkeypatch.setattr(cv2, "waitKey", tecla_falsa)
        monkeypatch.setattr(gd, "gravar_clipe", gravar_sem_consumir_teclas)
        monkeypatch.setattr(sys, "argv", ["gravar", "--pessoa", "P01", "--fitzpatrick", "III", "--fonte", origem, "--dataset", dataset])
        gd.main()

    original = gd.gravar_clipe

    def gravar_sem_consumir_teclas(cap, caminho, duracao, fps, mostrar=None):
        return original(cap, caminho, duracao, fps, None)

    rodar("  q")
    pasta = os.path.join(dataset, "P01", "plena")
    assert sorted(os.listdir(pasta)) == ["socorro_01.mp4", "socorro_02.mp4"]

    # Retoma: refaz o clipe anterior (apaga o socorro_02) e sai antes de gravar de novo
    rodar("rq")
    assert sorted(os.listdir(pasta)) == ["socorro_01.mp4"]

    rodar(" q")
    assert sorted(os.listdir(pasta)) == ["socorro_01.mp4", "socorro_02.mp4"]
    assert (tmp_path / "dataset" / "pessoas.csv").exists()
