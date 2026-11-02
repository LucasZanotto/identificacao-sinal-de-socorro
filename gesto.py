"""Geometria da mão e Máquina de Estados Finitos (MEF) do Sinal de Pedido de Socorro.

Este módulo não depende de câmera nem de modelos, então dá para testar a lógica
sem nenhuma imagem (ver testes/test_gesto.py).
"""
import math
from dataclasses import dataclass

# Índices dos 21 pontos do MediaPipe
PULSO = 0
POLEGAR_PONTA = 4
INDICADOR_BASE = 5
MEDIO_BASE = 9
MINIMO_BASE = 17

# (base, articulação do meio, ponta) dos quatro dedos
DEDOS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))

# Estados da MEF
REPOUSO = 0
MAO_ABERTA = 1
POLEGAR_RECOLHIDO = 2
GESTO_CONCLUIDO = 3


@dataclass(frozen=True)
class Configuracao:
    dedo_estendido: float = 1.5  # Acima dessa abertura o dedo conta como estendido
    dedo_dobrado: float = 1.4  # Abaixo dessa abertura o dedo conta como dobrado
    limiar_polegar: float = 1.1  # Em larguras de palma. Mais do que isso pega a mão reta com o polegar encostado
    exigir_palma: bool = False  # True só aceita a mão aberta com a palma para a câmera
    timeout_estado: float = 3.0  # Segundos sem ver a pose do estado atual (Δt)
    tempo_confirmacao: float = 0.13  # Segundos que uma pose precisa durar para ser aceita
    tempo_cancelar: float = 0.3  # Segundos de mão aberta para desfazer o polegar recolhido
    tempo_perda: float = 1.0  # Segundos sem ver a mão até descartar o rastreio
    fator_rastreio: float = 1.5  # Salto máximo do pulso, em tamanhos de palma


@dataclass(frozen=True)
class Pose:
    quatro_estendidos: bool
    quatro_dobrados: bool
    polegar_recolhido: bool
    palma_para_camera: bool


# Funções Geométricas
def calcular_distancia(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def tamanho_palma(pontos):
    # A palma inclinada encurta em uma direção só, então vale a maior das duas medidas
    return max(calcular_distancia(pontos[PULSO][:2], pontos[MEDIO_BASE][:2]), calcular_distancia(pontos[INDICADOR_BASE][:2], pontos[MINIMO_BASE][:2]))


def aberturas_dedos(pontos):
    """Distância da ponta de cada dedo ao pulso, dividida pela da base do dedo ao pulso.

    Fica perto de 1,8 com o dedo esticado e perto de 0,9 com o dedo fechado. Com a palma inclinada essa
    razão de duas distâncias longas varia menos do que o ângulo de uma articulação.
    """
    aberturas = []
    for base, _, ponta in DEDOS:
        ate_base = calcular_distancia(pontos[base], pontos[PULSO])
        aberturas.append(calcular_distancia(pontos[ponta], pontos[PULSO]) / ate_base if ate_base else 0.0)
    return aberturas


def razao_polegar(pontos):
    """Distância da ponta do polegar à base do mínimo, em larguras de palma."""
    largura_palma = calcular_distancia(pontos[INDICADOR_BASE][:2], pontos[MINIMO_BASE][:2])
    if largura_palma == 0:
        return float("inf")
    return calcular_distancia(pontos[POLEGAR_PONTA][:2], pontos[MINIMO_BASE][:2]) / largura_palma


def palma_voltada(pontos, lado):
    """Usa o sentido da palma na imagem junto com o lado da mão informado pelo MediaPipe."""
    pulso, indicador, minimo = pontos[PULSO], pontos[INDICADOR_BASE], pontos[MINIMO_BASE]
    sentido = (indicador[0] - pulso[0]) * (minimo[1] - pulso[1]) - (indicador[1] - pulso[1]) * (minimo[0] - pulso[0])
    if lado == "Left":
        return sentido > 0
    if lado == "Right":
        return sentido < 0
    return True


def classificar_pose(pontos, lado, cfg):
    """Recebe os 21 pontos em pixels. As razões não mudam com a distância da câmera."""
    aberturas = aberturas_dedos(pontos)
    return Pose(
        quatro_estendidos=all(a >= cfg.dedo_estendido for a in aberturas),
        quatro_dobrados=all(a <= cfg.dedo_dobrado for a in aberturas),
        polegar_recolhido=razao_polegar(pontos) < cfg.limiar_polegar,
        palma_para_camera=palma_voltada(pontos, lado),
    )


def rotular_pose(pose, cfg):
    if pose.quatro_dobrados:
        return "punho"
    if pose.quatro_estendidos and (pose.palma_para_camera or not cfg.exigir_palma):
        return "polegar" if pose.polegar_recolhido else "aberta"
    return None


class MaquinaEstados:
    def __init__(self, cfg, tempo):
        self.cfg = cfg
        self.estado = REPOUSO
        self.tempo_estado = tempo
        self.candidato = None
        self.inicio_candidato = tempo
        self.estado_maximo = REPOUSO

    def _ir_para(self, estado, tempo):
        self.estado = estado
        self.tempo_estado = tempo
        self.estado_maximo = max(self.estado_maximo, estado)

    def _pose_confirmada(self, pose, tempo):
        # Reset por Timeout
        if self.estado > REPOUSO and (tempo - self.tempo_estado) > self.cfg.timeout_estado:
            self.estado = REPOUSO

        # A pose precisa durar um pouco, para ignorar leituras erradas em quadros borrados.
        # Contar em tempo, e não em quadros, mantém o mesmo comportamento em qualquer taxa de quadros.
        rotulo = rotular_pose(pose, self.cfg)
        if rotulo != self.candidato:
            self.candidato = rotulo
            self.inicio_candidato = tempo
        return rotulo if self.pose_firme(tempo) else None

    def pose_firme(self, tempo):
        return tempo - self.inicio_candidato >= self.cfg.tempo_confirmacao

    def atualizar(self, pose, tempo):
        """Avança a MEF com a pose do quadro. Retorna True quando o gesto é concluído."""
        rotulo = self._pose_confirmada(pose, tempo)
        if rotulo is None:
            return False

        # Transição da MEF
        if rotulo == "polegar":
            # A pessoa pode levantar a mão com o polegar já dobrado, sem passar pelo estado 1
            self._ir_para(POLEGAR_RECOLHIDO, tempo)
        elif rotulo == "aberta":
            # Com os dedos fechando rápido, a imagem borra e o modelo às vezes lê mão aberta por alguns quadros
            if self.estado == POLEGAR_RECOLHIDO and tempo - self.inicio_candidato < self.cfg.tempo_cancelar:
                return False
            self._ir_para(MAO_ABERTA, tempo)
        elif self.estado == POLEGAR_RECOLHIDO:
            self._ir_para(GESTO_CONCLUIDO, tempo)
            self.estado = REPOUSO
            return True
        else:
            # Punho sem passar pelo polegar recolhido é só uma mão fechando
            self.estado = REPOUSO
        return False


class RastreioMao:
    def __init__(self, hand_id, pontos, lado, cfg, tempo, fabrica_mef=MaquinaEstados):
        self.id = hand_id
        self.pontos = pontos
        self.lado = lado
        self.mef = fabrica_mef(cfg, tempo)
        self.ultimo_visto = tempo

    @property
    def estado(self):
        return self.mef.estado


class RastreadorMaos:
    """Associa as mãos de cada quadro às do quadro anterior e mantém uma MEF para cada uma."""

    def __init__(self, cfg=None, fabrica_mef=MaquinaEstados, classificador=classificar_pose):
        self.cfg = cfg or Configuracao()
        self.fabrica_mef = fabrica_mef
        self.classificador = classificador
        self.maos = []
        self.proximo_id = 1
        self.estado_maximo = REPOUSO

    def atualizar(self, deteccoes, tempo):
        """deteccoes: lista de (pontos, lado). Retorna as mãos que concluíram o gesto neste quadro."""
        # Limpeza de mãos ausentes
        self.maos = [m for m in self.maos if (tempo - m.ultimo_visto) < self.cfg.tempo_perda]

        concluidas = []
        livres = list(self.maos)
        for pontos, lado in deteccoes:
            mao = self._associar(pontos, livres)
            if mao is None:
                mao = RastreioMao(self.proximo_id, pontos, lado, self.cfg, tempo, self.fabrica_mef)
                self.maos.append(mao)
                self.proximo_id += 1
            else:
                livres.remove(mao)
            mao.pontos = pontos
            mao.lado = lado
            mao.ultimo_visto = tempo
            if mao.mef.atualizar(self.classificador(pontos, lado, self.cfg), tempo):
                concluidas.append(mao)
            self.estado_maximo = max(self.estado_maximo, mao.mef.estado_maximo)
        return concluidas

    def _associar(self, pontos, livres):
        # Associação por proximidade, proporcional ao tamanho da mão na imagem. Usa o maior dos
        # dois tamanhos porque a palma parece menor no punho e a mão não pode virar outra ao fechar.
        melhor, menor = None, float("inf")
        for mao in livres:
            limite = self.cfg.fator_rastreio * max(tamanho_palma(pontos), tamanho_palma(mao.pontos))
            dist = calcular_distancia(pontos[PULSO][:2], mao.pontos[PULSO][:2])
            if dist < limite and dist < menor:
                melhor, menor = mao, dist
        return melhor
