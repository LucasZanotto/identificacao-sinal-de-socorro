"""Avaliação do sistema sobre o dataset. Gera as tabelas e figuras da análise de resultados.

Uso:
    python avaliar.py                       # lê a pasta dataset e escreve na pasta resultados
    python avaliar.py --calibrar P01,P02    # ajusta os limiares nessas pessoas e testa nas demais

O relatório segue as etapas do roteiro de análise de resultados: preparação, medição,
comparação, estresse, erros e conclusões. Os pontos das mãos ficam guardados em
dataset/_landmarks, então a segunda execução é rápida e os vídeos podem até ser apagados.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace

from gesto import (
    Configuracao, DEDOS, GESTO_CONCLUIDO, MAO_ABERTA, MaquinaEstados, POLEGAR_RECOLHIDO, PULSO, Pose, REPOUSO,
    RastreadorMaos, calcular_distancia, classificar_pose, razao_polegar, tamanho_palma,
)

ACAO_POSITIVA = "socorro"
EXTENSOES = (".mp4", ".avi", ".mov", ".mkv", ".ogg", ".webm", ".m4v", ".3gp")
FPS_ANALISE = 15  # Taxa com que os clipes são analisados, próxima do que uma CPU simples entrega
META = 0.80
MODOS = {
    "movimento": {"localizador": "movimento"},
    "quadro_inteiro": {"localizador": None},
    "sem_correcao": {"localizador": "movimento", "corrigir_luz": False},
    "sem_rastreio": {"localizador": "movimento", "rastrear": False},
    "yolo": {"localizador": "yolo"},  # Só para comparar com o sistema
}
CENARIOS_DE_LUZ = ("baixa", "contraluz")
# Mude quando o pipeline.py mudar, senão a avaliação reaproveita pontos extraídos pela versão anterior
VERSAO_PONTOS = "v3"
NOME_PADRAO ="Movimento + MediaPipe (padrão)"
NOME_YOLO = "YOLO a cada 3 quadros + MediaPipe"
NOME_YOLO_TODOS = "YOLO em todo quadro + MediaPipe"
ESCALAS_DISTANCIA = (1.0, 0.75, 0.5, 0.35, 0.25)
MODOS_DISTANCIA = ("movimento", "quadro_inteiro")  # O quadro inteiro mostra o que a detecção de movimento acrescenta
FAIXAS_PALMA = ((60, "60 ou mais"), (40, "40 a 60"), (30, "30 a 40"), (20, "20 a 30"), (15, "15 a 20"), (10, "10 a 15"), (0, "menos de 10"))
VARIACOES = {
    "limiar_polegar": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    "dedo_estendido": [1.4, 1.5, 1.6, 1.7],
    "dedo_dobrado": [1.2, 1.3, 1.35, 1.4],
    "timeout_estado": [2.0, 3.0, 4.0, 5.0],
    "tempo_confirmacao": [0.0, 0.07, 0.13, 0.2, 0.27],
    "tempo_cancelar": [0.0, 0.15, 0.25, 0.3, 0.35, 0.5],
    "exigir_palma": [True, False],
}


def modos_disponiveis():
    # O YOLO é opcional: só serve para comparar com o sistema e pede o PyTorch
    return [m for m in MODOS if m != "yolo" or importlib.util.find_spec("ultralytics")]


@dataclass(frozen=True)
class Clipe:
    caminho: str
    pessoa: str
    cenario: str
    nome: str

    @property
    def acao(self):
        return self.nome.rsplit("_", 1)[0]

    @property
    def positivo(self):
        return self.acao == ACAO_POSITIVA

    @property
    def id(self):
        return f"{self.pessoa}/{self.cenario}/{self.nome}"


def pessoa_real(clipe):
    # P01_s2 é uma segunda sessão da mesma pessoa, gravada para calibrar em uma sessão e testar na outra
    return clipe.pessoa.split("_")[0]


@dataclass
class Leitura:
    alerta: bool
    t_alerta: float
    estado_maximo: int
    frac_mao: float


# Regras simples usadas como comparação
class RegraAbreFecha(MaquinaEstados):
    """Mão aberta seguida de punho, sem olhar o polegar (regra de Mukesh et al., 2025)."""

    def atualizar(self, pose, tempo):
        rotulo = self._pose_confirmada(pose, tempo)
        if rotulo in ("aberta", "polegar"):
            self._ir_para(MAO_ABERTA, tempo)
        elif rotulo == "punho":
            if self.estado == MAO_ABERTA:
                self._ir_para(GESTO_CONCLUIDO, tempo)
                self.estado = REPOUSO
                return True
            self.estado = REPOUSO
        return False


class RegraSoPunho(MaquinaEstados):
    """Qualquer punho fechado dispara, sem nenhuma sequência: o gesto visto como pose estática."""

    def atualizar(self, pose, tempo):
        rotulo = self._pose_confirmada(pose, tempo)
        if rotulo == "punho" and self.estado == REPOUSO:
            self._ir_para(GESTO_CONCLUIDO, tempo)
            self.estado = POLEGAR_RECOLHIDO
            return True
        if rotulo != "punho" and rotulo is not None:
            self.estado = REPOUSO
        return False


class MaquinaPrototipo2(MaquinaEstados):
    """Regra original do Protótipo 2, mantida para medir o ganho da versão atual."""

    def atualizar(self, pose, tempo):
        if self.estado > REPOUSO and (tempo - self.tempo_estado) > self.cfg.timeout_estado:
            self.estado = REPOUSO
        if self.estado == REPOUSO and pose.quatro_estendidos and not pose.polegar_recolhido:
            self._ir_para(MAO_ABERTA, tempo)
        elif self.estado == MAO_ABERTA and pose.quatro_estendidos and pose.polegar_recolhido:
            self._ir_para(POLEGAR_RECOLHIDO, tempo)
        elif self.estado == POLEGAR_RECOLHIDO and pose.quatro_dobrados and pose.polegar_recolhido:
            self._ir_para(GESTO_CONCLUIDO, tempo)
            self.estado = REPOUSO
            return True
        return False


def pose_prototipo2(pontos, lado, cfg):
    estendidos = [
        calcular_distancia(pontos[ponta][:2], pontos[PULSO][:2]) > calcular_distancia(pontos[meio][:2], pontos[PULSO][:2])
        for _, meio, ponta in DEDOS
    ]
    return Pose(all(estendidos), not any(estendidos), razao_polegar(pontos) < 1.2, True)


METODOS = {
    "Sistema proposto (MEF completa)": (MaquinaEstados, classificar_pose),
    "Protótipo 2 (regra anterior)": (MaquinaPrototipo2, pose_prototipo2),
    "Abre e fecha, sem olhar o polegar": (RegraAbreFecha, classificar_pose),
    "Só punho, sem sequência": (RegraSoPunho, classificar_pose),
}


# Estatística
def wilson(acertos, total, z=1.96):
    """Intervalo de confiança de Wilson para uma proporção. Funciona bem com amostras pequenas."""
    if total == 0:
        return (0.0, 0.0)
    p = acertos / total
    centro = (p + z * z / (2 * total)) / (1 + z * z / total)
    margem = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def mcnemar_exato(b, c):
    """Teste de McNemar exato: b e c são os clipes em que apenas um dos dois métodos acertou."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def contar(pares):
    """pares: lista de (positivo real, alerta disparado)."""
    vp = sum(1 for real, previsto in pares if real and previsto)
    fn = sum(1 for real, previsto in pares if real and not previsto)
    vn = sum(1 for real, previsto in pares if not real and not previsto)
    fp = sum(1 for real, previsto in pares if not real and previsto)
    return vp, fp, vn, fn


def f1_score(vp, fp, fn):
    return 2 * vp / (2 * vp + fp + fn) if vp else 0.0


def metricas(pares, reamostragens=2000):
    vp, fp, vn, fn = contar(pares)
    total = vp + fp + vn + fn
    sorteio = random.Random(42)
    f1s = []
    if total:
        for _ in range(reamostragens):
            a = contar(sorteio.choices(pares, k=total))
            f1s.append(f1_score(a[0], a[1], a[3]))
        f1s.sort()
    return {
        "vp": vp, "fp": fp, "vn": vn, "fn": fn, "total": total,
        "acuracia": (vp + vn) / total if total else 0.0, "ic_acuracia": wilson(vp + vn, total),
        "precisao": vp / (vp + fp) if vp + fp else 0.0, "ic_precisao": wilson(vp, vp + fp),
        "sensibilidade": vp / (vp + fn) if vp + fn else 0.0, "ic_sensibilidade": wilson(vp, vp + fn),
        "especificidade": vn / (vn + fp) if vn + fp else 0.0, "ic_especificidade": wilson(vn, vn + fp),
        "f1": f1_score(vp, fp, fn),
        "ic_f1": (f1s[int(0.025 * len(f1s))], f1s[int(0.975 * len(f1s))]) if f1s else (0.0, 0.0),
    }


# Dados
def listar_clipes(pasta):
    clipes = []
    for pessoa in sorted(os.listdir(pasta)):
        dir_pessoa = os.path.join(pasta, pessoa)
        if pessoa.startswith("_") or not os.path.isdir(dir_pessoa):
            continue
        for cenario in sorted(os.listdir(dir_pessoa)):
            dir_cenario = os.path.join(dir_pessoa, cenario)
            if not os.path.isdir(dir_cenario):
                continue
            for arquivo in sorted(os.listdir(dir_cenario)):
                nome, extensao = os.path.splitext(arquivo)
                if extensao.lower() in EXTENSOES:
                    clipes.append(Clipe(os.path.join(dir_cenario, arquivo), pessoa, cenario, nome))
    return clipes


def ler_pessoas(pasta):
    caminho = os.path.join(pasta, "pessoas.csv")
    if not os.path.exists(caminho):
        return {}
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return {linha["pessoa"]: linha["fitzpatrick"] for linha in csv.DictReader(arquivo)}


def caminho_landmarks(pasta, modo, clipe):
    return os.path.join(pasta, "_landmarks", VERSAO_PONTOS, modo, clipe.pessoa, clipe.cenario, clipe.nome + ".json")


def pontos_em_dia(pasta, modo, clipe):
    """Os pontos guardados valem até o vídeo ser regravado. Vídeo apagado mantém os pontos que já existem."""
    destino = caminho_landmarks(pasta, modo, clipe)
    if not os.path.exists(destino):
        return False
    return not os.path.exists(clipe.caminho) or os.path.getmtime(destino) >= os.path.getmtime(clipe.caminho)


def extrair_landmarks(clipes, pasta, modo):
    """Passa cada vídeo pelo YOLO e pelo MediaPipe e guarda só os pontos, nunca a imagem."""
    pendentes = [c for c in clipes if not pontos_em_dia(pasta, modo, c)]
    if not pendentes:
        return
    import cv2
    from pipeline import DetectorMaos

    detector = DetectorMaos(**MODOS[modo])
    for i, clipe in enumerate(pendentes, 1):
        print(f"[{modo}] {i}/{len(pendentes)} {clipe.id}", flush=True)
        detector.reiniciar()
        cap = cv2.VideoCapture(clipe.caminho)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Só os quadros que uma CPU simples daria conta, porque o recorte depende do quadro anterior
        passo = max(1, round(fps / FPS_ANALISE))
        quadros = []
        indice = 0
        while True:
            sucesso, frame = cap.read()
            if not sucesso:
                break
            if indice % passo == 0:
                quadro = detector.processar(frame)
                quadros.append({
                    "t": round(indice / fps, 4),
                    "m": [{"p": [[round(v, 1) for v in ponto] for ponto in pontos], "l": lado, "c": round(conf, 3)} for pontos, lado, conf in quadro.maos],
                })
            indice += 1
        cap.release()
        destino = caminho_landmarks(pasta, modo, clipe)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(destino, "w", encoding="utf-8") as arquivo:
            json.dump({"fps": fps / passo, "quadros": quadros}, arquivo)


def carregar_landmarks(clipes, pasta, modo):
    dados = {}
    for clipe in clipes:
        with open(caminho_landmarks(pasta, modo, clipe), encoding="utf-8") as arquivo:
            dados[clipe] = json.load(arquivo)
    return dados


def reproduzir(dados, cfg, fabrica=MaquinaEstados, classificador=classificar_pose, pular=1):
    """Repassa os pontos guardados pela lógica do gesto, usando o tempo do próprio vídeo."""
    rastreador = RastreadorMaos(cfg, fabrica, classificador)
    t_alerta = None
    com_mao = 0
    quadros = dados["quadros"][::pular]
    for quadro in quadros:
        deteccoes = [(mao["p"], mao["l"]) for mao in quadro["m"]]
        com_mao += 1 if deteccoes else 0
        if rastreador.atualizar(deteccoes, quadro["t"]) and t_alerta is None:
            t_alerta = quadro["t"]
    return Leitura(t_alerta is not None, t_alerta, rastreador.estado_maximo, com_mao / max(len(quadros), 1))


def avaliar_clipes(clipes, dados, cfg, fabrica=MaquinaEstados, classificador=classificar_pose, pular=1):
    return {c: reproduzir(dados[c], cfg, fabrica, classificador, pular) for c in clipes}


def pares_de(leituras, filtro=lambda c: True):
    return [(c.positivo, l.alerta) for c, l in leituras.items() if filtro(c)]


def calibrar(clipes, dados, cfg):
    """Ajusta um parâmetro de cada vez, em duas passadas, pelo maior F1 (empate: maior sensibilidade)."""
    def nota(c):
        m = metricas(pares_de(avaliar_clipes(clipes, dados, c)), reamostragens=0)
        return (m["f1"], m["sensibilidade"])

    melhor, melhor_nota = cfg, nota(cfg)
    for _ in range(2):
        for parametro, valores in VARIACOES.items():
            for valor in valores:
                candidata = replace(melhor, **{parametro: valor})
                n = nota(candidata)
                if n > melhor_nota:
                    melhor, melhor_nota = candidata, n
    return melhor


def encolher(frame, escala):
    """Diminui a imagem dentro do mesmo quadro, como se a pessoa estivesse mais longe da câmera."""
    import cv2

    altura, largura = frame.shape[:2]
    menor = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    topo = (altura - menor.shape[0]) // 2
    esquerda = (largura - menor.shape[1]) // 2
    return cv2.copyMakeBorder(menor, topo, altura - menor.shape[0] - topo, esquerda, largura - menor.shape[1] - esquerda, cv2.BORDER_REPLICATE)


def faixa_palma(pixels):
    return next(nome for limite, nome in FAIXAS_PALMA if pixels >= limite)


def simular_distancia(clipes, dados, cfg):
    """Testa os clipes com a mão cada vez menor, procurando por movimento e no quadro inteiro. Agrupa pelo tamanho da palma em pixels."""
    import cv2
    from pipeline import DetectorMaos

    grupos = {}
    for modo in MODOS_DISTANCIA:
        detector = DetectorMaos(**MODOS[modo])
        grupos[modo] = defaultdict(list)
        for escala in ESCALAS_DISTANCIA:
            for clipe in clipes:
                palmas = [tamanho_palma(m["p"]) for q in dados[clipe]["quadros"] for m in q["m"]]
                if not palmas:
                    continue
                print(f"[distância] {modo} escala {escala} {clipe.id}", flush=True)
                detector.reiniciar()
                rastreador = RastreadorMaos(cfg)
                cap = cv2.VideoCapture(clipe.caminho)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                passo = max(1, round(fps / FPS_ANALISE))
                alerta = False
                indice = 0
                while True:
                    sucesso, frame = cap.read()
                    if not sucesso:
                        break
                    if indice % passo == 0:
                        quadro = detector.processar(encolher(frame, escala) if escala < 1 else frame)
                        if rastreador.atualizar([(p, l) for p, l, _ in quadro.maos], indice / fps):
                            alerta = True
                    indice += 1
                cap.release()
                grupos[modo][faixa_palma(statistics.median(palmas) * escala)].append((clipe.positivo, alerta))
        detector.fechar()
    return grupos


def motivo_falha(leitura):
    if leitura.frac_mao < 0.2:
        return "A mão quase não foi detectada no vídeo"
    if leitura.estado_maximo == REPOUSO:
        return "A mão foi vista, mas nenhuma pose do gesto foi reconhecida"
    if leitura.estado_maximo == MAO_ABERTA:
        return "Reconheceu a mão aberta, mas não o polegar dobrado"
    if leitura.estado_maximo == POLEGAR_RECOLHIDO:
        return "Reconheceu o polegar dobrado, mas não o punho fechado"
    return "Outro"


# Desempenho computacional
def ambiente():
    import psutil
    info = {
        "sistema": platform.platform(),
        "processador": platform.processor(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(),
        "memoria_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
    }
    if sys.platform == "win32":
        import winreg
        chave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["processador"] = winreg.QueryValueEx(chave, "ProcessorNameString")[0].strip()
    for pacote in ("opencv-python", "mediapipe", "ultralytics", "torch"):
        try:
            from importlib.metadata import version
            info[pacote] = version(pacote)
        except Exception:
            info[pacote] = "não instalado"
    return info


def medir_configuracao(parametros):
    """Roda em um processo separado para a memória de um teste não contaminar o outro."""
    import cv2
    import numpy as np
    import psutil
    from pipeline import DetectorMaos

    processo = psutil.Process()
    detector = DetectorMaos(localizador=parametros["localizador"], yolo_a_cada=parametros.get("yolo_a_cada", 3), rastrear=parametros.get("rastrear", True))
    rastreador = RastreadorMaos()
    intervalo = 1.0 / FPS_ANALISE
    ms_total, ms_local, ms_mp, ms_logica, memoria = [], [], [], [], []
    tempo_video = 0.0
    aquecimento = 10
    processo.cpu_percent(None)
    inicio_medida = time.perf_counter()
    for caminho in parametros["videos"]:
        cap = cv2.VideoCapture(caminho)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        passo = max(1, round(fps / FPS_ANALISE))
        detector.reiniciar()
        indice = 0
        while True:
            sucesso, frame = cap.read()
            if not sucesso:
                break
            indice += 1
            if (indice - 1) % passo:
                continue
            inicio = time.perf_counter()
            if parametros.get("altura"):
                proporcao = parametros["altura"] / frame.shape[0]
                frame = cv2.resize(frame, None, fx=proporcao, fy=proporcao)
            carga = parametros.get("carga", 1)
            if carga == 2:
                frame = np.hstack([frame, frame])
            elif carga == 4:
                frame = np.vstack([np.hstack([frame, frame]), np.hstack([frame, frame])])
            quadro = detector.processar(frame)
            antes_logica = time.perf_counter()
            tempo_video += intervalo
            rastreador.atualizar([(p, l) for p, l, _ in quadro.maos], tempo_video)
            fim = time.perf_counter()
            # Os primeiros quadros ficam de fora porque a primeira inferência é bem mais lenta
            aquecimento -= 1
            if aquecimento < 0:
                ms_local.append(quadro.ms_localizar)
                ms_mp.append(quadro.ms_mediapipe)
                ms_logica.append((fim - antes_logica) * 1000)
                ms_total.append((fim - inicio) * 1000)
                if len(ms_total) % 15 == 0:
                    memoria.append(processo.memory_info().rss / 1e6)
            # Respeita o ritmo de uma câmera ao vivo, senão o uso de CPU sairia sempre no máximo
            time.sleep(max(0.0, intervalo - (time.perf_counter() - inicio)))
        cap.release()
    cpu = processo.cpu_percent(None) / psutil.cpu_count()
    ms_total.sort()
    return {
        "quadros": len(ms_total),
        "duracao_s": round(time.perf_counter() - inicio_medida, 1),
        "ms_medio": round(statistics.mean(ms_total), 1),
        "ms_p95": round(ms_total[int(0.95 * len(ms_total))], 1),
        "ms_localizar": round(statistics.mean(ms_local), 2),
        "ms_mediapipe": round(statistics.mean(ms_mp), 2),
        "ms_logica": round(statistics.mean(ms_logica), 3),
        "fps_maximo": round(1000 / statistics.mean(ms_total), 1),
        "cpu_percentual": round(cpu, 1),
        "ram_mb": round(max(memoria), 0) if memoria else 0,
    }


def medir_desempenho(clipes, quantidade):
    # Mistura clipes com e sem o gesto, sempre no cenário de luz boa quando existir
    plena = [c for c in clipes if c.cenario == "plena"] or clipes
    amostra = sorted(plena, key=lambda c: (c.positivo, c.id))
    amostra = (amostra[::max(1, len(amostra) // quantidade)])[:quantidade]
    videos = [c.caminho for c in amostra]
    testes = [
        (NOME_PADRAO, {"localizador": "movimento"}),
        ("Movimento + MediaPipe, sem rastreamento", {"localizador": "movimento", "rastrear": False}),
        ("MediaPipe no quadro inteiro", {"localizador": None}),
        ("Padrão, imagem reduzida para 360p", {"localizador": "movimento", "altura": 360}),
        ("Padrão, mosaico com 2 cópias da cena", {"localizador": "movimento", "carga": 2}),
        ("Padrão, mosaico com 4 cópias da cena", {"localizador": "movimento", "carga": 4}),
    ]
    if "yolo" in modos_disponiveis():
        testes[1:1] = [
            (NOME_YOLO, {"localizador": "yolo", "yolo_a_cada": 3}),
            (NOME_YOLO_TODOS, {"localizador": "yolo", "yolo_a_cada": 1}),
        ]
    resultados = []
    for nome, parametros in testes:
        print(f"[desempenho] {nome}", flush=True)
        saida = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--medir", json.dumps({**parametros, "videos": videos})],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        linhas = [l for l in saida.stdout.splitlines() if l.startswith("{")]
        if not linhas:
            print(saida.stderr[-2000:])
            raise SystemExit(f"A medição de desempenho falhou em: {nome}")
        resultados.append((nome, json.loads(linhas[-1])))
    return resultados


# Relatório
def num(valor, casas=1):
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor):
    return f"{num(100 * valor)}%"


def pct_ic(valor, ic):
    return f"{num(100 * valor)}% ({num(100 * ic[0])} a {num(100 * ic[1])})"


def tabela(cabecalho, linhas):
    if not linhas:
        return "Nenhum.\n"
    saida = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    saida += ["| " + " | ".join(str(celula) for celula in linha) + " |" for linha in linhas]
    return "\n".join(saida) + "\n"


def linha_grupo(nome, m):
    return [nome, m["total"], m["vp"], m["fn"], m["vn"], m["fp"], pct_ic(m["acuracia"], m["ic_acuracia"]),
            pct_ic(m["sensibilidade"], m["ic_sensibilidade"]), pct_ic(m["especificidade"], m["ic_especificidade"])]


CAB_GRUPO = ["Grupo", "Clipes", "VP", "FN", "VN", "FP", "Acurácia (IC 95%)", "Sensibilidade (IC 95%)", "Especificidade (IC 95%)"]


def veredito_meta(valor, ic):
    if ic[0] > META:
        return f"acima de {META:.0%} mesmo no limite inferior do intervalo de confiança"
    if valor > META:
        return f"acima de {META:.0%} na estimativa, mas o intervalo de confiança ainda inclui {META:.0%}"
    return f"não ultrapassa {META:.0%}"


def figura_matriz(m, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valores = [[m["vp"], m["fn"]], [m["fp"], m["vn"]]]
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    maximo = max(max(l) for l in valores) or 1
    for i, linha in enumerate(valores):
        for j, valor in enumerate(linha):
            intensidade = 0.08 + 0.82 * valor / maximo
            cor = (1 - intensidade * (1 - 0.165), 1 - intensidade * (1 - 0.471), 1 - intensidade * (1 - 0.839))
            ax.add_patch(plt.Rectangle((j + 0.01, 1 - i + 0.01), 0.98, 0.98, color=cor))
            ax.text(j + 0.5, 1 - i + 0.5, str(valor), ha="center", va="center", fontsize=16, color="#ffffff" if intensidade > 0.6 else "#0b0b0b")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Alerta", "Sem alerta"])
    ax.set_yticks([1.5, 0.5], ["Com o sinal", "Sem o sinal"])
    ax.set_xlabel("Resposta do sistema", color="#52514e")
    ax.set_ylabel("O que havia no clipe", color="#52514e")
    ax.tick_params(length=0, colors="#52514e")
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_custo(desempenho, destino):
    """Compara o gasto do sistema com o da versão com YOLO. Um painel para cada medida, cada um com sua escala."""
    medidos = dict(desempenho)
    if NOME_YOLO not in medidos:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    itens = [("Movimento + MediaPipe", medidos[NOME_PADRAO], "#2a78d6"), ("YOLO + MediaPipe", medidos[NOME_YOLO], "#eb6834")]
    paineis = (("ms_medio", "Tempo por quadro", "ms", 1), ("cpu_percentual", "Uso de CPU", "%", 1), ("ram_mb", "Memória", "MB", 0))
    fig, eixos = plt.subplots(1, 3, figsize=(7.2, 1.7), dpi=200, sharey=True)
    for ax, (chave, titulo, unidade, casas) in zip(eixos, paineis):
        maximo = max(d[chave] for _, d, _ in itens) or 1
        for i, (_, d, cor) in enumerate(itens):
            y = len(itens) - 1 - i
            ax.barh(y, d[chave], height=0.55, color=cor)
            ax.text(d[chave] + 0.03 * maximo, y, f"{num(d[chave], casas)} {unidade}", va="center", fontsize=9, color="#0b0b0b")
        ax.set_xlim(0, maximo * 1.55)
        ax.set_title(titulo, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#0b0b0b")
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#c9c8c2")
    eixos[0].set_yticks(range(len(itens)), [nome for nome, _, _ in reversed(itens)])
    eixos[0].set_ylim(-0.6, len(itens) - 0.4)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return True


def figura_cenarios(linhas, destino):
    """linhas: lista de (cenário, métricas). Um painel para cada métrica, com o intervalo de confiança."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 0.5 * len(linhas) + 1.4), dpi=200, sharey=True)
    paineis = (("sensibilidade", "Sensibilidade"), ("especificidade", "Especificidade"))
    for ax, (chave, titulo) in zip(eixos, paineis):
        for i, (nome, m) in enumerate(linhas):
            y = len(linhas) - 1 - i
            baixo, alto = m["ic_" + chave]
            ax.plot([100 * baixo, 100 * alto], [y, y], color="#2a78d6", lw=2, solid_capstyle="round")
            ax.plot([100 * m[chave]], [y], "o", color="#2a78d6", ms=7, mec="#fcfcfb", mew=1.5)
            ax.text(103, y, f"{100 * m[chave]:.0f}%", va="center", fontsize=8, color="#0b0b0b")
        ax.axvline(100 * META, color="#8a8983", lw=1, ls=(0, (3, 3)))
        ax.text(100 * META, len(linhas) - 0.45, "meta", ha="center", va="bottom", fontsize=8, color="#52514e")
        ax.set_title(titulo, fontsize=10, color="#0b0b0b", loc="left", pad=14)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.tick_params(axis="x", labelsize=8, colors="#52514e", length=0)
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#0b0b0b")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for lado in ax.spines.values():
            lado.set_visible(False)
    eixos[0].set_yticks(range(len(linhas)), [nome for nome, _ in reversed(linhas)])
    eixos[0].set_ylim(-0.6, len(linhas) - 0.4)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def montar_relatorio(clipes, teste, calibracao, cfg, pessoas, dados, leituras, desempenho, distancia, info, sintetico):
    r = []
    add = r.append
    add("# Relatório de avaliação\n")
    if sintetico:
        add("> **ATENÇÃO: DADOS DE TESTE DE SOFTWARE.** Este relatório foi gerado com dados sintéticos só para conferir o programa. Nenhum número daqui pode ir para o TCC.\n")
    geral = metricas(pares_de(leituras))
    positivos = [c for c in teste if c.positivo]

    # Etapa 1
    add("## 1. Preparação e conferência dos dados\n")
    n_pessoas = len({pessoa_real(c) for c in clipes})
    add(f"O dataset tem {len(clipes)} clipes de {n_pessoas} {'pessoa' if n_pessoas == 1 else 'pessoas'}. "
        f"A avaliação usa {len(teste)} clipes ({len(positivos)} com o sinal e {len(teste) - len(positivos)} sem o sinal), "
        f"analisados a até {FPS_ANALISE} quadros por segundo.\n")
    cenarios = sorted({c.cenario for c in clipes})
    contagem = Counter((c.pessoa, c.cenario) for c in clipes)
    add(tabela(["Pessoa", "Fitzpatrick"] + cenarios + ["Total"], [
        [p, pessoas.get(p, "não informado")] + [contagem[(p, s)] for s in cenarios] + [sum(contagem[(p, s)] for s in cenarios)]
        for p in sorted({c.pessoa for c in clipes})
    ]))
    if calibracao:
        add(f"\nCalibração feita com {len(calibracao)} clipes de {', '.join(sorted({c.pessoa for c in calibracao}))}. Esses clipes ficam fora do teste.\n")
    else:
        add("\nSem separação entre calibração e teste: usa a configuração padrão, cujos limiares foram escolhidos com os clipes do P01. "
            "Se o P01 está neste teste, o resultado é uma prévia otimista e não serve como medição final.\n")
    if len({pessoa_real(c) for c in teste}) == 1:
        add("\nO teste tem uma pessoa só: o resultado vale para essa pessoa e não pode ser generalizado para outras mãos e tons de pele.\n")
    add("\nConfiguração usada:\n\n```json\n" + json.dumps(asdict(cfg), indent=2, ensure_ascii=False) + "\n```\n")

    problemas = []
    for c in clipes:
        d = dados["movimento"][c]
        duracao = len(d["quadros"]) / d["fps"] if d["fps"] else 0
        frac = sum(1 for q in d["quadros"] if q["m"]) / max(len(d["quadros"]), 1)
        if duracao < 2:
            problemas.append((c.id, f"clipe muito curto ({duracao:.1f} s)"))
        elif d["fps"] < 10:
            problemas.append((c.id, f"taxa de quadros baixa ({d['fps']:.1f} FPS)"))
        elif c.positivo and frac < 0.2:
            problemas.append((c.id, f"mão detectada em só {100 * frac:.0f}% dos quadros"))
    add(f"\nClipes para conferir na mão: {len(problemas)}.\n")
    if problemas:
        add(tabela(["Clipe", "Motivo"], problemas))
    add("\nAmbiente de teste:\n\n```json\n" + json.dumps(info, indent=2, ensure_ascii=False) + "\n```\n")

    # Etapa 2
    add("## 2. Medição\n")
    add(f"Cada clipe conta uma vez: alerta disparado em algum momento é resposta positiva. O intervalo é de 95% (Wilson).\n")
    add(tabela(["Métrica", "Valor (IC 95%)"], [
        ["Acurácia", pct_ic(geral["acuracia"], geral["ic_acuracia"])],
        ["Precisão", pct_ic(geral["precisao"], geral["ic_precisao"])],
        ["Sensibilidade (recall)", pct_ic(geral["sensibilidade"], geral["ic_sensibilidade"])],
        ["Especificidade", pct_ic(geral["especificidade"], geral["ic_especificidade"])],
        ["F1-Score", pct_ic(geral["f1"], geral["ic_f1"])],
    ]))
    add(f"\nVP = {geral['vp']}, FP = {geral['fp']}, VN = {geral['vn']}, FN = {geral['fn']}.\n")
    add("\n![Matriz de confusão](matriz_confusao.png)\n")

    por_cenario = [(s, metricas(pares_de(leituras, lambda c, s=s: c.cenario == s))) for s in sorted({c.cenario for c in teste})]
    add("\nPor cenário:\n")
    add(tabela(CAB_GRUPO, [linha_grupo(s, m) for s, m in por_cenario]))
    add("\n![Sensibilidade e especificidade por cenário](metricas_por_cenario.png)\n")

    tons = sorted({pessoas.get(c.pessoa, "não informado") for c in teste})
    add("\nPor tom de pele (escala de Fitzpatrick):\n")
    add(tabela(CAB_GRUPO, [linha_grupo(t, metricas(pares_de(leituras, lambda c, t=t: pessoas.get(c.pessoa, "não informado") == t))) for t in tons]))
    add("\nPor pessoa:\n")
    add(tabela(CAB_GRUPO, [linha_grupo(p, metricas(pares_de(leituras, lambda c, p=p: c.pessoa == p))) for p in sorted({c.pessoa for c in teste})]))

    alertas = [l.t_alerta for c, l in leituras.items() if c.positivo and l.alerta]
    if alertas:
        add(f"\nNos clipes com o sinal, o alerta saiu em média {num(statistics.mean(alertas), 2)} s depois do início do clipe.\n")

    # Etapa 3
    add("\n## 3. Comparação\n")
    add("Mesmos clipes e mesmos pontos, trocando só a regra de decisão. O p vem do teste de McNemar exato contra o sistema proposto.\n")
    linhas = []
    for nome, (fabrica, classificador) in METODOS.items():
        outra = avaliar_clipes(teste, dados["movimento"], cfg, fabrica, classificador)
        m = metricas(pares_de(outra))
        b = sum(1 for c in teste if leituras[c].alerta == c.positivo and outra[c].alerta != c.positivo)
        c_ = sum(1 for c in teste if leituras[c].alerta != c.positivo and outra[c].alerta == c.positivo)
        linhas.append([nome, pct_ic(m["acuracia"], m["ic_acuracia"]), pct(m["sensibilidade"]), pct(m["especificidade"]), pct(m["f1"]),
                       "referência" if fabrica is MaquinaEstados else f"{num(mcnemar_exato(b, c_), 4)} (só o proposto acerta {b}, só o outro acerta {c_})"])
    add(tabela(["Método", "Acurácia (IC 95%)", "Sensibilidade", "Especificidade", "F1", "McNemar p"], linhas))

    localizadores = [("movimento (sistema)", leituras), ("quadro inteiro", avaliar_clipes(teste, dados["quadro_inteiro"], cfg))]
    if "yolo" in dados:
        localizadores.append(("YOLO", avaliar_clipes(teste, dados["yolo"], cfg)))
    add("\nOnde o MediaPipe olha, por cenário (mesma regra de decisão):\n")
    linhas = []
    for s in sorted({c.cenario for c in teste}) + ["Todos"]:
        filtro = (lambda c: True) if s == "Todos" else (lambda c, s=s: c.cenario == s)
        linha = [s]
        for _, outra in localizadores:
            m = metricas(pares_de(outra, filtro), reamostragens=0)
            linha += [pct(m["sensibilidade"]), pct(m["especificidade"])]
        linhas.append(linha)
    add(tabela(["Cenário"] + [f"{rotulo} {nome}" for nome, _ in localizadores for rotulo in ("Sens.", "Espec.")], linhas))
    if "yolo" in dados:
        com_yolo = localizadores[2][1]
        b = sum(1 for c in teste if leituras[c].alerta == c.positivo and com_yolo[c].alerta != c.positivo)
        c_ = sum(1 for c in teste if leituras[c].alerta != c.positivo and com_yolo[c].alerta == c.positivo)
        add(f"\nClipes em que só um dos dois acerta: movimento {b}, YOLO {c_}. McNemar p = {num(mcnemar_exato(b, c_), 4)}.\n")

    sem_rastreio = avaliar_clipes(teste, dados["sem_rastreio"], cfg)
    add("\nCom e sem o rastreamento do MediaPipe, por cenário (mesma regra de decisão):\n")
    linhas = []
    for s in sorted({c.cenario for c in teste}) + ["Todos"]:
        filtro = (lambda c: True) if s == "Todos" else (lambda c, s=s: c.cenario == s)
        a, b = metricas(pares_de(leituras, filtro), reamostragens=0), metricas(pares_de(sem_rastreio, filtro), reamostragens=0)
        linhas.append([s, pct(a["sensibilidade"]), pct(b["sensibilidade"]), pct(a["especificidade"]), pct(b["especificidade"])])
    add(tabela(["Cenário", "Sens. com rastreamento", "Sens. sem rastreamento", "Espec. com rastreamento", "Espec. sem rastreamento"], linhas))
    b = sum(1 for c in teste if leituras[c].alerta == c.positivo and sem_rastreio[c].alerta != c.positivo)
    c_ = sum(1 for c in teste if leituras[c].alerta != c.positivo and sem_rastreio[c].alerta == c.positivo)
    add(f"\nClipes em que só um dos dois acerta: com rastreamento {b}, sem rastreamento {c_}. McNemar p = {num(mcnemar_exato(b, c_), 4)}.\n")

    clipes_luz = [c for c in teste if c.cenario in CENARIOS_DE_LUZ]
    if clipes_luz:
        sem_correcao = avaliar_clipes(clipes_luz, dados["sem_correcao"], cfg)
        add("\nCom e sem a correção de iluminação, nos cenários de luz difícil:\n")
        linhas = []
        for s in sorted({c.cenario for c in clipes_luz}):
            filtro = lambda c, s=s: c.cenario == s
            a, b = metricas(pares_de({c: leituras[c] for c in clipes_luz}, filtro)), metricas(pares_de(sem_correcao, filtro))
            linhas.append([s, pct(a["sensibilidade"]), pct(b["sensibilidade"]), pct(a["especificidade"]), pct(b["especificidade"])])
        add(tabela(["Cenário", "Sens. com correção", "Sens. sem correção", "Espec. com correção", "Espec. sem correção"], linhas))

    add("\nValores publicados por trabalhos relacionados (dados e hardware diferentes, servem de referência, não de disputa direta):\n")
    add(tabela(["Trabalho", "Abordagem", "Resultado publicado", "Velocidade publicada"], [
        ["Azimi et al. (2023)", "CNN 3D (MobileNet) sobre o vídeo", "91,25% de acurácia", "16 FPS (i9 e RTX 2080 Ti), 6 FPS (Jetson Nano)"],
        ["Buccellato et al. (2025)", "MediaPipe, DeepSORT e Random Forest", "94% de acurácia", "0,067 s (Apple M3), 0,471 s (Jetson Orin Nano)"],
        ["Shaik, Reddy e Shaik (2025)", "MediaPipe, sequência aberta, punho, aberta", "84,2% ± 2,8% de acurácia", "6 a 10 FPS em notebook"],
    ]))

    # Etapa 4
    add("\n## 4. Estresse\n")
    add("### Mudança de configurações\n")
    add("Cada linha muda um único parâmetro a partir da configuração usada.\n")
    linhas = []
    for parametro, valores in VARIACOES.items():
        for valor in valores:
            m = metricas(pares_de(avaliar_clipes(teste, dados["movimento"], replace(cfg, **{parametro: valor}))), reamostragens=0)
            marca = " (usado)" if getattr(cfg, parametro) == valor else ""
            linhas.append([parametro, f"{str(valor).replace('.', ',')}{marca}", pct(m["acuracia"]), pct(m["sensibilidade"]), pct(m["especificidade"])])
    for pular in (1, 2, 3):
        m = metricas(pares_de(avaliar_clipes(teste, dados["movimento"], cfg, pular=pular)), reamostragens=0)
        marca = " (usado)" if pular == 1 else ""
        linhas.append(["quadros por segundo analisados", f"cerca de {FPS_ANALISE // pular}{marca}", pct(m["acuracia"]), pct(m["sensibilidade"]), pct(m["especificidade"])])
    add(tabela(["Parâmetro", "Valor", "Acurácia", "Sensibilidade", "Especificidade"], linhas))

    if distancia:
        add("\n### Distância simulada\n")
        add("A imagem de cada clipe é encolhida dentro do mesmo quadro, como se a pessoa estivesse mais longe. Agrupa pelo tamanho da palma em pixels. O quadro inteiro mostra o que acontece sem a detecção de movimento para indicar onde procurar a mão.\n")
        linhas = []
        for _, nome in FAIXAS_PALMA:
            if any(nome in grupo for grupo in distancia.values()):
                linha = [nome]
                for modo in MODOS_DISTANCIA:
                    vp, fp, vn, fn = contar(distancia[modo].get(nome, []))
                    linha += [f"{vp} de {vp + fn}", f"{fp} de {vn + fp}"]
                linhas.append(linha)
        add(tabela(["Palma (pixels)", "Sinais, movimento", "Alarmes, movimento", "Sinais, quadro inteiro", "Alarmes, quadro inteiro"], linhas))

    if desempenho:
        add("\n### Custo computacional, teste por partes e teste de escala\n")
        add(f"Medido sem placa de vídeo, no ritmo de {FPS_ANALISE} quadros por segundo. O FPS máximo é o limite calculado pelo tempo médio de cada quadro. A CPU é o percentual do processador inteiro.\n")
        add(tabela(["Configuração", "Tempo por quadro (ms)", "p95 (ms)", "Localizar (ms)", "MediaPipe (ms)", "Geometria e MEF (ms)", "FPS máximo", "CPU", "RAM (MB)"], [
            [n, num(d["ms_medio"]), num(d["ms_p95"]), num(d["ms_localizar"]), num(d["ms_mediapipe"]), num(d["ms_logica"], 3), num(d["fps_maximo"]), f"{num(d['cpu_percentual'])}%", int(d["ram_mb"])]
            for n, d in desempenho
        ]))
        if NOME_YOLO in dict(desempenho):
            add("\n![Gasto do sistema comparado com o YOLO](custo_localizador.png)\n")

    # Etapa 5
    add("\n## 5. Erros e limitações\n")
    falsos_negativos = [c for c in teste if c.positivo and not leituras[c].alerta]
    falsos_positivos = [c for c in teste if not c.positivo and leituras[c].alerta]
    add(f"Falsos negativos ({len(falsos_negativos)}), agrupados pelo ponto em que a sequência parou:\n")
    add(tabela(["Motivo", "Clipes"], Counter(motivo_falha(leituras[c]) for c in falsos_negativos).most_common()))
    add(f"\nFalsos positivos ({len(falsos_positivos)}), agrupados pela ação que a pessoa fazia:\n")
    add(tabela(["Ação", "Clipes"], Counter(c.acao for c in falsos_positivos).most_common()))
    if falsos_positivos:
        add("\nAntes de contar como erro, assista a esses clipes: é fácil fazer o sinal sem querer em um clipe de controle. "
            f"{', '.join(c.id for c in falsos_positivos)}.\n")
    add("\nLista completa em `por_clipe.csv`.\n")

    # Etapa 6
    add("\n## 6. Respostas às hipóteses (dados para a discussão)\n")
    add(f"- **H1, os pontos da mão bastam para reconhecer o gesto:** acurácia de {pct_ic(geral['acuracia'], geral['ic_acuracia'])}, {veredito_meta(geral['acuracia'], geral['ic_acuracia'])}. Sensibilidade de {pct_ic(geral['sensibilidade'], geral['ic_sensibilidade'])}.\n")
    if desempenho and NOME_YOLO in dict(desempenho):
        medidos = dict(desempenho)
        padrao, com_yolo = medidos[NOME_PADRAO], medidos[NOME_YOLO]
        add(f"- **H2, custo menor que o de uma CNN sobre a imagem inteira:** o sistema gastou {num(padrao['ms_medio'])} ms por quadro, {num(padrao['cpu_percentual'])}% de CPU e {int(padrao['ram_mb'])} MB, contra {num(com_yolo['ms_medio'])} ms, {num(com_yolo['cpu_percentual'])}% e {int(com_yolo['ram_mb'])} MB com o YOLOv8n. A geometria e a MEF levaram {num(padrao['ms_logica'], 3)} ms, e uma única passada do YOLOv8n pelo quadro levou {num(medidos[NOME_YOLO_TODOS]['ms_localizar'])} ms.\n")
    oclusao = metricas(pares_de(leituras, lambda c: c.cenario == "oclusao"))
    if oclusao["total"]:
        add(f"- **H3, oclusão parcial mantendo mais de {META:.0%}:** acurácia de {pct_ic(oclusao['acuracia'], oclusao['ic_acuracia'])} no cenário de oclusão, {veredito_meta(oclusao['acuracia'], oclusao['ic_acuracia'])}.\n")
    return "\n".join(r), geral, por_cenario


def main():
    parser = argparse.ArgumentParser(description="Avaliação do sistema sobre o dataset")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--calibrar", default="", help="Pessoas usadas só para ajustar os limiares, ex.: P01,P02")
    parser.add_argument("--sem-desempenho", action="store_true", help="Pula a medição de CPU, RAM e tempo")
    parser.add_argument("--distancia", action="store_true", help="Também simula a pessoa mais longe da câmera (demora)")
    parser.add_argument("--clipes-desempenho", type=int, default=10)
    parser.add_argument("--medir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.medir:
        print(json.dumps(medir_configuracao(json.loads(args.medir))))
        return

    clipes = listar_clipes(args.dataset)
    if not clipes:
        raise SystemExit(f"Nenhum vídeo em {args.dataset}/<pessoa>/<cenario>/<acao>_<nn>.mp4")

    # Confere antes de processar os vídeos, que é a parte demorada
    pessoas_calibracao = {p.strip() for p in args.calibrar.split(",") if p.strip()}
    calibracao = [c for c in clipes if c.pessoa in pessoas_calibracao]
    teste = [c for c in clipes if c.pessoa not in pessoas_calibracao]
    if not teste:
        raise SystemExit(
            f"O dataset só tem {', '.join(sorted({c.pessoa for c in clipes}))} e {args.calibrar} ficou só para calibrar, "
            "então não sobra ninguém para o teste. Grave outra pessoa ou uma segunda sessão (P01_s2), ou rode sem --calibrar para ver uma prévia."
        )

    pessoas = ler_pessoas(args.dataset)
    os.makedirs(args.saida, exist_ok=True)
    dados = {}
    for modo in modos_disponiveis():
        # A comparação sem a correção de luz só interessa nos cenários de luz difícil
        do_modo = [c for c in clipes if modo != "sem_correcao" or c.cenario in CENARIOS_DE_LUZ]
        extrair_landmarks(do_modo, args.dataset, modo)
        dados[modo] = carregar_landmarks(do_modo, args.dataset, modo)

    cfg = Configuracao()
    if calibracao:
        print("Calibrando...", flush=True)
        cfg = calibrar(calibracao, dados["movimento"], cfg)

    leituras = avaliar_clipes(teste, dados["movimento"], cfg)
    desempenho = [] if args.sem_desempenho else medir_desempenho(clipes, args.clipes_desempenho)
    distancia = simular_distancia(teste, dados["movimento"], cfg) if args.distancia else None
    sintetico = os.path.exists(os.path.join(args.dataset, "SINTETICO.txt"))

    relatorio, geral, por_cenario = montar_relatorio(clipes, teste, calibracao, cfg, pessoas, dados, leituras, desempenho, distancia, ambiente(), sintetico)
    with open(os.path.join(args.saida, "relatorio.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)
    with open(os.path.join(args.saida, "por_clipe.csv"), "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["clipe", "pessoa", "cenario", "acao", "com_sinal", "alerta", "t_alerta", "estado_maximo", "frac_quadros_com_mao"])
        for c, l in leituras.items():
            escritor.writerow([c.id, c.pessoa, c.cenario, c.acao, int(c.positivo), int(l.alerta), l.t_alerta if l.alerta else "", l.estado_maximo, round(l.frac_mao, 3)])
    figura_matriz(geral, os.path.join(args.saida, "matriz_confusao.png"))
    figura_cenarios(por_cenario, os.path.join(args.saida, "metricas_por_cenario.png"))
    if desempenho:
        figura_custo(desempenho, os.path.join(args.saida, "custo_localizador.png"))
    print(f"Pronto: {os.path.join(args.saida, 'relatorio.md')}")


if __name__ == "__main__":
    main()
