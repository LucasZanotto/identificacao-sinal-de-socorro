import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import avaliar
from avaliar import Clipe, RegraAbreFecha, RegraSoPunho, contar, mcnemar_exato, metricas, reproduzir, wilson
from gesto import Configuracao, MaquinaEstados, POLEGAR_RECOLHIDO, REPOUSO
import mao_sintetica as ms


def dados_de(sequencia, fps=30.0):
    quadros = []
    for pontos, n in sequencia:
        for _ in range(n):
            mao = [{"p": pontos, "l": "Left", "c": 0.9}] if pontos else []
            quadros.append({"t": len(quadros) / fps, "m": mao, "y": 1.0, "d": 2.0})
    return {"fps": fps, "quadros": quadros}


def test_wilson_bate_com_valor_de_referencia():
    baixo, alto = wilson(8, 10)
    assert baixo == pytest.approx(0.4902, abs=1e-3)
    assert alto == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(10, 10)[1] == pytest.approx(1.0)


def test_mcnemar_exato():
    assert mcnemar_exato(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exato(0, 0) == 1.0
    assert mcnemar_exato(5, 5) == 1.0


def test_contagem_da_matriz_de_confusao():
    pares = [(True, True), (True, False), (False, False), (False, False), (False, True)]
    assert contar(pares) == (1, 1, 2, 1)


def test_metricas_conferidas_a_mao():
    pares = [(True, True)] * 8 + [(True, False)] * 2 + [(False, False)] * 9 + [(False, True)] * 1
    m = metricas(pares)
    assert m["acuracia"] == pytest.approx(17 / 20)
    assert m["precisao"] == pytest.approx(8 / 9)
    assert m["sensibilidade"] == pytest.approx(0.8)
    assert m["especificidade"] == pytest.approx(0.9)
    assert m["f1"] == pytest.approx(2 * 8 / (2 * 8 + 1 + 2))
    assert m["ic_f1"][0] <= m["f1"] <= m["ic_f1"][1]


def test_metricas_sem_positivos_nao_quebra():
    m = metricas([(False, False)] * 3)
    assert m["sensibilidade"] == 0.0
    assert m["f1"] == 0.0


def test_clipe_entende_o_nome_do_arquivo():
    assert Clipe("x", "P01", "plena", "socorro_03").positivo
    negativo = Clipe("x", "P01", "plena", "cocar_cabeca_01")
    assert negativo.acao == "cocar_cabeca"
    assert not negativo.positivo


def test_listar_clipes_ignora_o_que_nao_e_video(tmp_path):
    (tmp_path / "P01" / "plena").mkdir(parents=True)
    (tmp_path / "P01" / "plena" / "socorro_01.mp4").write_bytes(b"")
    (tmp_path / "P01" / "plena" / "notas.txt").write_text("x")
    (tmp_path / "_landmarks" / "plena").mkdir(parents=True)
    (tmp_path / "_landmarks" / "plena" / "socorro_01.mp4").write_bytes(b"")
    (tmp_path / "pessoas.csv").write_text("pessoa,fitzpatrick\nP01,III\n")
    clipes = avaliar.listar_clipes(str(tmp_path))
    assert [c.id for c in clipes] == ["P01/plena/socorro_01"]
    assert avaliar.ler_pessoas(str(tmp_path)) == {"P01": "III"}


def test_clipe_regravado_nao_reaproveita_os_pontos_antigos(tmp_path):
    video = tmp_path / "P01" / "baixa" / "socorro_01.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    clipe = Clipe(str(video), "P01", "baixa", "socorro_01")
    assert not avaliar.pontos_em_dia(str(tmp_path), "movimento", clipe)
    destino = avaliar.caminho_landmarks(str(tmp_path), "movimento", clipe)
    os.makedirs(os.path.dirname(destino))
    open(destino, "w").close()
    assert avaliar.pontos_em_dia(str(tmp_path), "movimento", clipe)
    # Regravado depois da extração, o clipe precisa ser extraído de novo
    depois = os.path.getmtime(destino) + 100
    os.utime(video, (depois, depois))
    assert not avaliar.pontos_em_dia(str(tmp_path), "movimento", clipe)
    # Vídeo apagado de propósito: os pontos guardados continuam valendo
    video.unlink()
    assert avaliar.pontos_em_dia(str(tmp_path), "movimento", clipe)


def test_reproducao_respeita_a_taxa_de_analise():
    dados = dados_de([(ms.polegar_dobrado(), 30), (ms.punho(), 30)])
    assert reproduzir(dados, Configuracao()).alerta
    # A 1 quadro por segundo nunca há dois quadros seguidos para confirmar cada pose
    assert not reproduzir(dados, Configuracao(), pular=30).alerta


def test_encolher_mantem_o_tamanho_do_quadro():
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[300:420, 600:680] = 255
    menor = avaliar.encolher(frame, 0.5)
    assert menor.shape == frame.shape
    # A mancha branca fica com metade do tamanho, ainda no centro
    ys, xs = np.nonzero(menor[:, :, 0] > 127)
    assert xs.max() - xs.min() + 1 == pytest.approx(40, abs=3)
    assert ys.max() - ys.min() + 1 == pytest.approx(60, abs=3)
    assert abs((xs.max() + xs.min()) / 2 - 640) < 5


def test_faixa_da_palma():
    assert avaliar.faixa_palma(80) == "60 ou mais"
    assert avaliar.faixa_palma(34) == "30 a 40"
    assert avaliar.faixa_palma(5) == "menos de 10"


def test_grafico_de_custo(tmp_path):
    medida = {"ms_medio": 27.2, "cpu_percentual": 4.1, "ram_mb": 180.0}
    outra = {"ms_medio": 44.0, "cpu_percentual": 8.6, "ram_mb": 514.0}
    destino = tmp_path / "custo.png"
    assert not avaliar.figura_custo([(avaliar.NOME_PADRAO, medida)], str(destino))
    assert not destino.exists()
    assert avaliar.figura_custo([(avaliar.NOME_PADRAO, medida), (avaliar.NOME_YOLO, outra)], str(destino))
    assert destino.stat().st_size > 0


def test_mao_normal_fechando_engana_as_regras_simples_mas_nao_o_sistema():
    dados = dados_de([(ms.aberta(), 20), (ms.punho(), 20)])
    assert not reproduzir(dados, Configuracao()).alerta
    assert reproduzir(dados, Configuracao(), RegraAbreFecha).alerta
    assert reproduzir(dados, Configuracao(), RegraSoPunho).alerta


def test_prototipo2_perde_quem_levanta_a_mao_com_polegar_dobrado():
    dados = dados_de([(ms.polegar_dobrado(), 20), (ms.punho(), 20)])
    assert reproduzir(dados, Configuracao()).alerta
    assert not reproduzir(dados, Configuracao(), avaliar.MaquinaPrototipo2, avaliar.pose_prototipo2).alerta


def test_leitura_registra_quanto_a_mao_apareceu():
    leitura = reproduzir(dados_de([(None, 30), (ms.polegar_dobrado(), 30)]), Configuracao())
    assert leitura.frac_mao == pytest.approx(0.5)
    assert leitura.estado_maximo == POLEGAR_RECOLHIDO
    assert not leitura.alerta


def test_motivo_de_falha():
    assert "quase não" in avaliar.motivo_falha(reproduzir(dados_de([(None, 60)]), Configuracao()))
    assert "nenhuma pose" in avaliar.motivo_falha(reproduzir(dados_de([(ms.meio_fechada(), 60)]), Configuracao()))
    assert "polegar dobrado" in avaliar.motivo_falha(reproduzir(dados_de([(ms.aberta(), 60)]), Configuracao()))
    assert "punho" in avaliar.motivo_falha(reproduzir(dados_de([(ms.polegar_dobrado(), 60)]), Configuracao()))


def montar_dataset(pasta, pessoas=(("P01", "II"), ("P02", "V"), ("P03", "III"))):
    """Dataset falso: arquivos de vídeo vazios e pontos já extraídos, para o programa rodar sem modelos."""
    sinal = dados_de([(ms.polegar_dobrado(), 30), (ms.punho(), 30)])
    fechar = dados_de([(ms.aberta(), 30), (ms.punho(), 30)])
    nada = dados_de([(None, 60)])
    cenarios = {
        "plena": [("socorro_01", sinal), ("socorro_02", sinal), ("abrir_fechar_01", fechar), ("acenar_01", dados_de([(ms.aberta(), 60)]))],
        "oclusao": [("socorro_01", sinal), ("socorro_02", nada), ("segurar_copo_01", nada)],
        "baixa": [("socorro_01", sinal), ("acenar_01", nada)],
    }
    for pessoa, tom in pessoas:
        for cenario, clipes in cenarios.items():
            os.makedirs(os.path.join(pasta, pessoa, cenario), exist_ok=True)
            for nome, dados in clipes:
                open(os.path.join(pasta, pessoa, cenario, nome + ".mp4"), "wb").close()
                for modo in avaliar.MODOS:
                    destino = avaliar.caminho_landmarks(pasta, modo, Clipe("", pessoa, cenario, nome))
                    os.makedirs(os.path.dirname(destino), exist_ok=True)
                    with open(destino, "w", encoding="utf-8") as arquivo:
                        json.dump(dados, arquivo)
    with open(os.path.join(pasta, "pessoas.csv"), "w", encoding="utf-8") as arquivo:
        arquivo.write("pessoa,fitzpatrick\n" + "".join(f"{pessoa},{tom}\n" for pessoa, tom in pessoas))
    open(os.path.join(pasta, "SINTETICO.txt"), "w").close()


def rodar_main(monkeypatch, *argumentos):
    monkeypatch.setattr(sys, "argv", ["avaliar.py", *argumentos])
    avaliar.main()


def test_relatorio_completo_com_dados_falsos(tmp_path, monkeypatch):
    dataset, saida = str(tmp_path / "dataset"), str(tmp_path / "resultados")
    montar_dataset(dataset)
    rodar_main(monkeypatch, "--dataset", dataset, "--saida", saida, "--sem-desempenho")
    with open(os.path.join(saida, "relatorio.md"), encoding="utf-8") as arquivo:
        texto = arquivo.read()
    assert "DADOS DE TESTE DE SOFTWARE" in texto
    for trecho in ("## 1. Preparação", "## 2. Medição", "## 3. Comparação", "## 4. Estresse", "## 5. Erros", "## 6. Respostas"):
        assert trecho in texto
    # 27 clipes: 15 marcados com o sinal, dos quais 12 têm o gesto de verdade, e 12 sem o sinal
    assert "VP = 12, FP = 0, VN = 12, FN = 3" in texto
    assert "Só punho" in texto
    assert "correção de iluminação" in texto
    assert "Com e sem o rastreamento" in texto
    # Sem --calibrar, o relatório avisa que os limiares padrão vieram dos clipes do P01
    assert "prévia otimista" in texto
    for arquivo_saida in ("matriz_confusao.png", "metricas_por_cenario.png", "por_clipe.csv"):
        assert os.path.getsize(os.path.join(saida, arquivo_saida)) > 0


def test_distancia_compara_a_busca_por_movimento_com_o_quadro_inteiro(tmp_path, monkeypatch):
    dataset, saida = str(tmp_path / "dataset"), str(tmp_path / "resultados")
    montar_dataset(dataset)
    grupos = {"movimento": {"20 a 30": [(True, True), (True, True), (False, False)]}, "quadro_inteiro": {"20 a 30": [(True, False), (True, True), (False, True)]}}
    monkeypatch.setattr(avaliar, "simular_distancia", lambda *a: grupos)
    rodar_main(monkeypatch, "--dataset", dataset, "--saida", saida, "--sem-desempenho", "--distancia")
    with open(os.path.join(saida, "relatorio.md"), encoding="utf-8") as arquivo:
        texto = arquivo.read()
    assert "| 20 a 30 | 2 de 2 | 0 de 1 | 1 de 2 | 1 de 1 |" in texto


def test_calibracao_deixa_as_pessoas_fora_do_teste(tmp_path, monkeypatch):
    dataset, saida = str(tmp_path / "dataset"), str(tmp_path / "resultados")
    montar_dataset(dataset)
    rodar_main(monkeypatch, "--dataset", dataset, "--saida", saida, "--sem-desempenho", "--calibrar", "P01")
    with open(os.path.join(saida, "por_clipe.csv"), encoding="utf-8") as arquivo:
        texto = arquivo.read()
    assert "P01/" not in texto
    assert "P02/" in texto


def test_calibrar_sem_sobrar_teste_explica_o_que_fazer(tmp_path, monkeypatch):
    dataset, saida = str(tmp_path / "dataset"), str(tmp_path / "resultados")
    montar_dataset(dataset)
    with pytest.raises(SystemExit) as erro:
        rodar_main(monkeypatch, "--dataset", dataset, "--saida", saida, "--sem-desempenho", "--calibrar", "P01,P02,P03")
    assert "P01, P02, P03" in str(erro.value)
    assert "sem --calibrar" in str(erro.value)


def test_segunda_sessao_conta_como_a_mesma_pessoa(tmp_path, monkeypatch):
    dataset, saida = str(tmp_path / "dataset"), str(tmp_path / "resultados")
    montar_dataset(dataset, pessoas=(("P01", "III"), ("P01_s2", "III")))
    rodar_main(monkeypatch, "--dataset", dataset, "--saida", saida, "--sem-desempenho", "--calibrar", "P01")
    with open(os.path.join(saida, "relatorio.md"), encoding="utf-8") as arquivo:
        texto = arquivo.read()
    assert "de 1 pessoa." in texto
    assert "Esses clipes ficam fora do teste" in texto
    assert "O teste tem uma pessoa só" in texto
    with open(os.path.join(saida, "por_clipe.csv"), encoding="utf-8") as arquivo:
        clipes = arquivo.read()
    assert "P01/" not in clipes
    assert "P01_s2/" in clipes


def test_dataset_vazio_avisa(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        rodar_main(monkeypatch, "--dataset", str(tmp_path), "--saida", str(tmp_path / "r"), "--sem-desempenho")
