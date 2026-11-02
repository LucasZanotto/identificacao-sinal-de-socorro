"""Gera os 21 pontos de uma mão a partir de uma descrição simples da pose.

Serve para testar a lógica sem nenhuma imagem. As medidas ficam próximas do que foi medido em
vídeos reais do gesto: abertura perto de 1,8 com o dedo esticado e perto de 0,9 com o dedo fechado.
A sequência de pontos é um teste do programa, não diz nada sobre a precisão em pessoas reais.
"""
import math

BASES = {"indicador": (-0.35, -1.0), "medio": (-0.10, -1.05), "anelar": (0.15, -1.0), "minimo": (0.38, -0.90)}
COMPRIMENTOS = {"indicador": (0.42, 0.26, 0.22), "medio": (0.46, 0.30, 0.24), "anelar": (0.42, 0.28, 0.24), "minimo": (0.33, 0.20, 0.20)}
# Quanto cada articulação do dedo dobra, em graus, com a mão totalmente fechada
DOBRA_MAXIMA = (85.0, 100.0, 60.0)


def _dedo(base, comprimentos, fechamento):
    """Dedo para cima (y negativo na imagem) dobrando na direção da câmera (z negativo). Fechamento vai de 0 a 1."""
    x, y = base
    z = 0.0
    pontos = [(x, y, z)]
    giro = 0.0
    for comprimento, dobra in zip(comprimentos, DOBRA_MAXIMA):
        giro += math.radians(dobra * fechamento)
        y -= comprimento * math.cos(giro)
        z -= comprimento * math.sin(giro)
        pontos.append((x, y, z))
    return pontos


def mao(fechamento=0.0, polegar="aberto", costas=False, fechamentos=None):
    """polegar: 'aberto', 'recolhido' ou 'colado' (encostado no indicador, sem cruzar a palma)."""
    pontas = {
        "aberto": [(-0.40, -0.25), (-0.65, -0.45), (-0.80, -0.70), (-0.95, -0.92)],
        "recolhido": [(-0.40, -0.25), (-0.50, -0.50), (-0.20, -0.62), (0.08, -0.62)],
        "colado": [(-0.40, -0.25), (-0.50, -0.50), (-0.52, -0.80), (-0.52, -1.08)],
    }[polegar]
    pontos = [(0.0, 0.0, 0.0)] + [(x, y, 0.0) for x, y in pontas]
    fechamentos = fechamentos or {}
    for nome in ("indicador", "medio", "anelar", "minimo"):
        pontos += _dedo(BASES[nome], COMPRIMENTOS[nome], fechamentos.get(nome, fechamento))
    if costas:
        pontos = [(-x, y, z) for x, y, z in pontos]
    return pontos


def inclinar(pontos, lateral=0.0, frente=0.0):
    """Gira a mão em torno do pulso sem mudar a pose. Lateral vira a palma para o lado e frente a deita para trás."""
    a, b = math.radians(lateral), math.radians(frente)
    saida = []
    for x, y, z in pontos:
        x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
        y, z = y * math.cos(b) - z * math.sin(b), y * math.sin(b) + z * math.cos(b)
        saida.append((x, y, z))
    return saida


def transformar(pontos, escala=200.0, rotacao=0.0, x=640.0, y=500.0):
    seno, cosseno = math.sin(math.radians(rotacao)), math.cos(math.radians(rotacao))
    return [
        (x + escala * (cosseno * px - seno * py), y + escala * (seno * px + cosseno * py), escala * pz)
        for px, py, pz in pontos
    ]


def aberta(**kw):
    return transformar(mao(polegar="aberto"), **kw)


def polegar_dobrado(**kw):
    return transformar(mao(polegar="recolhido"), **kw)


def punho(**kw):
    return transformar(mao(fechamento=1.0, polegar="recolhido"), **kw)


def meio_fechada(**kw):
    # No meio do caminho, parte dos dedos ainda está esticada e parte já fechou
    return transformar(mao(polegar="recolhido", fechamentos={"anelar": 1.0, "minimo": 1.0}), **kw)
