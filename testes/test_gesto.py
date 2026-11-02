import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesto import (
    Configuracao, MAO_ABERTA, MaquinaEstados, POLEGAR_RECOLHIDO, RastreadorMaos, REPOUSO,
    aberturas_dedos, classificar_pose, razao_polegar, rotular_pose, tamanho_palma,
)
import mao_sintetica as ms

CFG = Configuracao()
QUADRO = 1 / 15


def rodar(sequencia, cfg=CFG, tempo_inicial=0.0):
    """sequencia: lista de (pontos, quantos quadros). Retorna os tempos em que houve alerta."""
    rastreador = RastreadorMaos(cfg)
    tempo = tempo_inicial
    alertas = []
    for pontos, quadros in sequencia:
        for _ in range(quadros):
            if rastreador.atualizar([] if pontos is None else [(pontos, "Left")], tempo):
                alertas.append(tempo)
            tempo += QUADRO
    return alertas, rastreador


def rotulo(pontos, lado="Left", cfg=CFG):
    return rotular_pose(classificar_pose(pontos, lado, cfg), cfg)


def test_abertura_do_dedo_nao_quebra_com_pontos_repetidos():
    assert aberturas_dedos([(0.0, 0.0, 0.0)] * 21) == [0.0, 0.0, 0.0, 0.0]


def test_rotulos_das_poses_basicas():
    assert rotulo(ms.aberta()) == "aberta"
    assert rotulo(ms.polegar_dobrado()) == "polegar"
    assert rotulo(ms.punho()) == "punho"
    assert rotulo(ms.meio_fechada()) is None


def test_mao_sintetica_fica_dentro_dos_valores_medidos_no_video_real():
    assert min(aberturas_dedos(ms.aberta())) > 1.6
    assert max(aberturas_dedos(ms.punho())) < 1.1
    assert razao_polegar(ms.aberta()) > 1.5
    assert razao_polegar(ms.polegar_dobrado()) < 0.7


@pytest.mark.parametrize("escala", [40, 120, 300, 900])
@pytest.mark.parametrize("rotacao", [-30, 0, 25])
def test_pose_nao_depende_da_distancia_nem_da_inclinacao(escala, rotacao):
    kw = {"escala": escala, "rotacao": rotacao}
    assert rotulo(ms.aberta(**kw)) == "aberta"
    assert rotulo(ms.polegar_dobrado(**kw)) == "polegar"
    assert rotulo(ms.punho(**kw)) == "punho"


@pytest.mark.parametrize("lateral", [-50, -30, 0, 30, 50])
@pytest.mark.parametrize("frente", [-50, -30, 0, 30, 50])
def test_pose_nao_depende_de_a_palma_estar_de_frente_para_a_camera(lateral, frente):
    def inclinada(**kw):
        return ms.transformar(ms.inclinar(ms.mao(**kw), lateral, frente))

    assert rotulo(inclinada(polegar="aberto")) == "aberta"
    assert rotulo(inclinada(polegar="recolhido")) == "polegar"
    assert rotulo(inclinada(fechamento=1.0, polegar="recolhido")) == "punho"


def test_sinal_feito_com_a_palma_inclinada_dispara():
    def inclinada(**kw):
        return ms.transformar(ms.inclinar(ms.mao(**kw), 45, 30))

    alertas, _ = rodar([(inclinada(polegar="aberto"), 8), (inclinada(polegar="recolhido"), 8), (inclinada(fechamento=1.0, polegar="recolhido"), 8)])
    assert len(alertas) == 1


def test_costas_da_mao_tambem_contam():
    costas = ms.transformar(ms.mao(polegar="aberto", costas=True))
    assert rotulo(costas) == "aberta"
    assert rotulo(costas, cfg=Configuracao(exigir_palma=True)) is None
    polegar = ms.transformar(ms.mao(polegar="recolhido", costas=True))
    punho = ms.transformar(ms.mao(fechamento=1.0, polegar="recolhido", costas=True))
    alertas, _ = rodar([(polegar, 8), (punho, 8)])
    assert len(alertas) == 1


def test_palma_inclinada_nao_faz_a_mao_parecer_pequena():
    # O tamanho decide se é a mesma mão de um quadro para o outro
    frente = tamanho_palma(ms.aberta())
    assert tamanho_palma(ms.transformar(ms.inclinar(ms.mao(), 0, 70))) > 0.6 * frente
    assert tamanho_palma(ms.transformar(ms.inclinar(ms.mao(), 70, 0))) > 0.6 * frente


def test_palma_depende_do_lado_informado():
    assert classificar_pose(ms.aberta(), "Left", CFG).palma_para_camera
    assert not classificar_pose(ms.aberta(), "Right", CFG).palma_para_camera
    espelhada = ms.transformar(ms.mao(polegar="aberto", costas=True))
    assert classificar_pose(espelhada, "Right", CFG).palma_para_camera


def test_sequencia_completa_dispara_uma_vez():
    alertas, _ = rodar([(ms.aberta(), 8), (ms.polegar_dobrado(), 8), (ms.meio_fechada(), 2), (ms.punho(), 8), (ms.punho(), 8)])
    assert len(alertas) == 1


def test_levantar_a_mao_com_polegar_ja_dobrado_tambem_dispara():
    # Foi assim que o vídeo real começou e o Protótipo 2 perdia esse caso
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.punho(), 8)])
    assert len(alertas) == 1


def test_dois_gestos_seguidos_disparam_duas_vezes():
    gesto = [(ms.polegar_dobrado(), 8), (ms.punho(), 8)]
    alertas, _ = rodar(gesto + gesto)
    assert len(alertas) == 2


def test_fechar_a_mao_normal_nao_dispara():
    fechando = ms.transformar(ms.mao(polegar="aberto", fechamentos={"anelar": 1.0, "minimo": 1.0}))
    punho_com_polegar_fora = ms.transformar(ms.mao(fechamento=1.0, polegar="aberto"))
    alertas, _ = rodar([(ms.aberta(), 8), (fechando, 3), (punho_com_polegar_fora, 10)])
    assert alertas == []


def test_polegar_colado_ao_indicador_nao_conta_como_recolhido():
    colado = ms.transformar(ms.mao(polegar="colado"))
    assert rotulo(colado) == "aberta"
    alertas, _ = rodar([(colado, 8), (ms.punho(), 8)])
    assert alertas == []


def test_ordem_invertida_nao_dispara():
    alertas, _ = rodar([(ms.punho(), 8), (ms.polegar_dobrado(), 8), (ms.aberta(), 8)])
    assert alertas == []


def test_soltar_o_polegar_antes_de_fechar_cancela():
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.aberta(), 8), (ms.punho(), 8)])
    assert alertas == []


def test_leitura_breve_de_mao_aberta_nao_apaga_o_polegar_recolhido():
    # Enquanto os dedos fecham rápido, a imagem borra e o modelo lê mão aberta por alguns quadros
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.aberta(), 3), (ms.punho(), 8)])
    assert len(alertas) == 1


def test_mao_aberta_por_mais_tempo_apaga_o_polegar_recolhido():
    quadros = int(CFG.tempo_cancelar / QUADRO) + 2
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.aberta(), quadros), (ms.punho(), 8)])
    assert alertas == []


def test_leitura_isolada_nao_muda_o_estado():
    # Um único quadro errado de punho no meio não pode concluir o gesto
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.punho(), 1), (ms.polegar_dobrado(), 8)])
    assert alertas == []


def test_polegar_lido_errado_em_quadro_borrado_nao_prepara_o_alerta():
    # Mão aberta se movendo rápido: o modelo lê polegar dobrado por 2 quadros e logo depois a mão fecha
    alertas, _ = rodar([(ms.aberta(), 8), (ms.polegar_dobrado(), 2), (ms.aberta(), 3), (ms.punho(), 8)])
    assert alertas == []


@pytest.mark.parametrize("fps", [8, 15, 30, 60])
def test_confirmacao_nao_depende_da_taxa_de_quadros(fps):
    # O mesmo gesto, com a mesma duração, em câmeras diferentes
    rastreador = RastreadorMaos(CFG)
    alertas = []
    tempo = 0.0
    for pontos, segundos in ((ms.polegar_dobrado(), 0.5), (ms.punho(), 0.5)):
        for _ in range(int(segundos * fps)):
            alertas += rastreador.atualizar([(pontos, "Left")], tempo)
            tempo += 1 / fps
    assert len(alertas) == 1


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_engano_rapido_nao_passa_em_nenhuma_taxa_de_quadros(fps):
    # 0,1 segundo de polegar lido errado é engano com qualquer câmera, mesmo sendo muitos quadros a 60 FPS
    rastreador = RastreadorMaos(CFG)
    alertas = []
    tempo = 0.0
    for pontos, segundos in ((ms.aberta(), 0.5), (ms.polegar_dobrado(), 0.1), (ms.aberta(), 0.2), (ms.punho(), 0.5)):
        for _ in range(int(segundos * fps)):
            alertas += rastreador.atualizar([(pontos, "Left")], tempo)
            tempo += 1 / fps
    assert alertas == []


def test_passou_do_tempo_entre_as_etapas_nao_dispara():
    pausa = int(3.5 / QUADRO)
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (ms.meio_fechada(), pausa), (ms.punho(), 8)])
    assert alertas == []


def test_segurar_a_pose_nao_estoura_o_tempo():
    longo = int(10 / QUADRO)
    alertas, _ = rodar([(ms.polegar_dobrado(), longo), (ms.punho(), 8)])
    assert len(alertas) == 1


def test_mao_some_por_pouco_tempo_mantem_o_progresso():
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (None, int(0.5 / QUADRO)), (ms.punho(), 8)])
    assert len(alertas) == 1


def test_mao_some_por_muito_tempo_perde_o_progresso():
    alertas, _ = rodar([(ms.polegar_dobrado(), 8), (None, int(1.5 / QUADRO)), (ms.punho(), 8)])
    assert alertas == []


def test_duas_maos_tem_estados_independentes():
    rastreador = RastreadorMaos(CFG)
    direita = {"x": 300.0}
    esquerda = {"x": 1000.0}
    tempo = 0.0
    alertas = []
    for _ in range(4):
        rastreador.atualizar([(ms.polegar_dobrado(**direita), "Left"), (ms.aberta(**esquerda), "Left")], tempo)
        tempo += QUADRO
    assert sorted(m.estado for m in rastreador.maos) == [MAO_ABERTA, POLEGAR_RECOLHIDO]
    for _ in range(4):
        alertas += rastreador.atualizar([(ms.punho(**direita), "Left"), (ms.aberta(**esquerda), "Left")], tempo)
        tempo += QUADRO
    assert len(alertas) == 1
    assert sorted(m.estado for m in rastreador.maos) == [REPOUSO, MAO_ABERTA]


def test_mao_que_salta_para_longe_vira_outra():
    rastreador = RastreadorMaos(CFG)
    rastreador.atualizar([(ms.polegar_dobrado(x=200.0), "Left")], 0.0)
    rastreador.atualizar([(ms.polegar_dobrado(x=1100.0), "Left")], QUADRO)
    assert [m.id for m in rastreador.maos] == [1, 2]


def test_rastreio_acompanha_movimento_rapido_proporcional_ao_tamanho():
    rastreador = RastreadorMaos(CFG)
    rastreador.atualizar([(ms.polegar_dobrado(x=400.0, escala=300.0), "Left")], 0.0)
    rastreador.atualizar([(ms.polegar_dobrado(x=700.0, escala=300.0), "Left")], QUADRO)
    assert len(rastreador.maos) == 1


def test_mao_que_fecha_continua_sendo_a_mesma():
    # No punho a palma parece bem menor e o pulso costuma saltar na imagem: não pode virar outra mão
    aberta_grande = ms.polegar_dobrado(x=500.0, escala=300.0)
    punho_pequeno = ms.punho(x=800.0, escala=60.0)
    rastreador = RastreadorMaos(CFG)
    for i in range(3):
        rastreador.atualizar([(aberta_grande, "Left")], i * QUADRO)
    alertas = []
    for i in range(3, 6):
        alertas += rastreador.atualizar([(punho_pequeno, "Left")], i * QUADRO)
    assert [m.id for m in rastreador.maos] == [1]
    assert len(alertas) == 1


def test_movimento_durante_o_gesto_nao_atrapalha():
    sequencia = [(ms.polegar_dobrado(x=400.0 + 20 * i), 1) for i in range(8)] + [(ms.punho(x=560.0 + 20 * i), 1) for i in range(8)]
    alertas, _ = rodar(sequencia)
    assert len(alertas) == 1


def test_estado_maximo_registra_ate_onde_a_sequencia_chegou():
    _, rastreador = rodar([(ms.aberta(), 8)])
    assert rastreador.estado_maximo == MAO_ABERTA
    _, rastreador = rodar([(ms.polegar_dobrado(), 8)])
    assert rastreador.estado_maximo == POLEGAR_RECOLHIDO
