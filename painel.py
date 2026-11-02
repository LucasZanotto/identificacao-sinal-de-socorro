"""Tela ao vivo: desenha sobre o vídeo e ao lado dele o que o sistema está pensando a cada quadro."""
from collections import deque

import cv2
import numpy as np

from gesto import (
    DEDOS, GESTO_CONCLUIDO, MAO_ABERTA, POLEGAR_RECOLHIDO, REPOUSO,
    aberturas_dedos, classificar_pose, razao_polegar, rotular_pose, tamanho_palma,
)

ALTURA = 720
LARGURA_LATERAL = 400
MARGEM = 16

# Cores em BGR
BRANCO = (235, 235, 235)
CINZA = (150, 150, 150)
ESCURO = (70, 66, 62)
FUNDO = (38, 34, 30)
VERDE = (100, 210, 90)
VERMELHO = (70, 70, 235)
AMARELO = (0, 210, 235)
LARANJA = (0, 165, 255)
AZUL = (255, 130, 40)

NOMES_ESTADO = {REPOUSO: "Repouso", MAO_ABERTA: "Mao aberta", POLEGAR_RECOLHIDO: "Polegar recolhido", GESTO_CONCLUIDO: "Gesto concluido"}
CAIXAS_ESTADO = [(REPOUSO, "Repouso"), (MAO_ABERTA, "Aberta"), (POLEGAR_RECOLHIDO, "Polegar"), (GESTO_CONCLUIDO, "Punho")]
ESPERANDO = {
    REPOUSO: "Mostre a mao com os dedos esticados",
    MAO_ABERTA: "Agora dobre o polegar sobre a palma",
    POLEGAR_RECOLHIDO: "Agora feche os dedos sobre o polegar",
}
NOMES_POSE = {"aberta": "MAO ABERTA", "polegar": "POLEGAR DOBRADO", "punho": "PUNHO", None: "INDEFINIDA"}
NOMES_DEDOS = ("Indicador", "Medio", "Anelar", "Minimo")
CORES_ORIGEM = {"mao": VERDE, "movimento": LARANJA, "pessoa": AZUL, "inteiro": CINZA}
NOMES_ORIGEM = {"mao": "mao acompanhada", "movimento": "regiao que se mexeu", "pessoa": "pessoa (YOLO)", "inteiro": "quadro inteiro"}
LIGACOES_PALMA = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
LIGACOES_POLEGAR = [(1, 2), (2, 3), (3, 4)]


def texto(img, conteudo, x, y, cor=BRANCO, escala=0.45, grossura=1):
    cv2.putText(img, conteudo, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def titulo(img, conteudo, y):
    texto(img, conteudo, MARGEM, y, CINZA, 0.42)
    cv2.line(img, (MARGEM, y + 6), (LARGURA_LATERAL - MARGEM, y + 6), ESCURO, 1)


def estado_dedo(abertura, cfg):
    if abertura >= cfg.dedo_estendido:
        return "esticado", VERDE
    if abertura <= cfg.dedo_dobrado:
        return "dobrado", VERMELHO
    return "em duvida", AMARELO


def medidor(img, x, y, largura, valor, minimo, maximo, marcas, cor):
    """Barra pequena com a posição do valor e riscos nos limites que decidem a pose."""
    cv2.rectangle(img, (x, y), (x + largura, y + 8), ESCURO, -1)
    frac = float(np.clip((valor - minimo) / (maximo - minimo), 0, 1))
    cv2.rectangle(img, (x, y), (x + int(frac * largura), y + 8), cor, -1)
    for marca in marcas:
        mx = x + int((marca - minimo) / (maximo - minimo) * largura)
        cv2.line(img, (mx, y - 2), (mx, y + 10), BRANCO, 1)


class Painel:
    def __init__(self, cfg, tempo_inicial=0.0):
        self.cfg = cfg
        self.tempo_inicial = tempo_inicial
        self.eventos = deque(maxlen=6)
        self.estados = {}
        self.mostrar_mascara = True

    def _anotar(self, tempo, conteudo):
        linha = f"{tempo - self.tempo_inicial:6.1f}s  {conteudo}"
        self.eventos.appendleft(linha)
        print(linha, flush=True)

    def registrar(self, rastreador, concluidas, tempo):
        """Compara com o quadro anterior e anota no painel e no terminal o que mudou em cada mão."""
        vistas = set()
        for mao in rastreador.maos:
            vistas.add(mao.id)
            antes = self.estados.get(mao.id)
            if mao in concluidas:
                self._anotar(tempo, f"Mao #{mao.id}: GESTO CONCLUIDO, alerta disparado")
            elif antes is None:
                self._anotar(tempo, f"Mao #{mao.id} apareceu")
            elif antes != mao.estado:
                motivo = ""
                if mao.estado == REPOUSO:
                    motivo = " (fechou sem o polegar)" if mao.mef.candidato == "punho" else " (tempo esgotado)"
                self._anotar(tempo, f"Mao #{mao.id}: {NOMES_ESTADO[antes]} -> {NOMES_ESTADO[mao.estado]}{motivo}")
            self.estados[mao.id] = mao.estado
        for mao_id in list(self.estados):
            if mao_id not in vistas:
                self._anotar(tempo, f"Mao #{mao_id} saiu de vista")
                del self.estados[mao_id]

    def desenhar(self, frame, quadro, rastreador, alerta, mascara, medidas, tempo):
        video = self._desenhar_video(frame.copy(), quadro, rastreador)
        escala = ALTURA / video.shape[0]
        video = cv2.resize(video, (int(video.shape[1] * escala), ALTURA))
        alerta.desenhar(video)
        lateral = self._desenhar_lateral(quadro, rastreador, alerta, mascara, medidas, tempo)
        return np.hstack([video, lateral])

    def _desenhar_video(self, frame, quadro, rastreador):
        fator = frame.shape[0] / ALTURA
        grossura = max(1, round(1.5 * fator))
        for regiao in quadro.regioes:
            x1, y1, x2, y2 = regiao.caixa
            cor = CORES_ORIGEM[regiao.origem]
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, grossura)
            cv2.putText(frame, NOMES_ORIGEM[regiao.origem], (x1 + 4, y1 + int(18 * fator)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fator, cor, grossura, cv2.LINE_AA)
        for mao in rastreador.maos:
            self._desenhar_mao(frame, mao, fator, grossura)
        return frame

    def _desenhar_mao(self, frame, mao, fator, grossura):
        pontos = [(int(p[0]), int(p[1])) for p in mao.pontos]
        for a, b in LIGACOES_PALMA:
            cv2.line(frame, pontos[a], pontos[b], CINZA, grossura, cv2.LINE_AA)
        # Cada dedo na cor do que o sistema entendeu: verde esticado, vermelho dobrado, amarelo em dúvida
        for (base, meio, ponta), abertura in zip(DEDOS, aberturas_dedos(mao.pontos)):
            _, cor = estado_dedo(abertura, self.cfg)
            for a, b in zip(range(base, ponta), range(base + 1, ponta + 1)):
                cv2.line(frame, pontos[a], pontos[b], cor, grossura + 1, cv2.LINE_AA)
        pose = classificar_pose(mao.pontos, mao.lado, self.cfg)
        cor_polegar = CINZA if pose.quatro_dobrados else (AZUL if pose.polegar_recolhido else BRANCO)
        for a, b in LIGACOES_POLEGAR:
            cv2.line(frame, pontos[a], pontos[b], cor_polegar, grossura + 1, cv2.LINE_AA)
        # O ponto acompanha o tamanho da mão, senão uma mão distante some embaixo dos pontos
        raio = int(np.clip(tamanho_palma(mao.pontos) * 0.04, 1, 3 * fator))
        for p in pontos:
            cv2.circle(frame, p, raio, (255, 255, 255), -1, cv2.LINE_AA)
        x, y = pontos[0]
        cv2.putText(frame, f"Mao #{mao.id} | Est: {mao.estado}", (x - int(30 * fator), y + int(24 * fator)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fator, AMARELO, grossura, cv2.LINE_AA)

    def _desenhar_lateral(self, quadro, rastreador, alerta, mascara, medidas, tempo):
        img = np.full((ALTURA, LARGURA_LATERAL, 3), FUNDO, dtype=np.uint8)
        # Quando há mais de uma mão, o painel acompanha a mais adiantada na sequência
        mao = max(rastreador.maos, key=lambda m: (m.estado, tamanho_palma(m.pontos)), default=None)
        y = self._secao_sequencia(img, 24, mao, alerta, tempo)
        y = self._secao_leitura(img, y, mao, tempo)
        y = self._secao_regioes(img, y, quadro, mascara)
        y = self._secao_eventos(img, y)
        self._secao_desempenho(img, y, medidas)
        return img

    def _secao_sequencia(self, img, y, mao, alerta, tempo):
        titulo(img, "SEQUENCIA DO GESTO", y)
        y += 16
        estado = mao.estado if mao is not None else REPOUSO
        largura = (LARGURA_LATERAL - 2 * MARGEM - 3 * 4) // 4
        for i, (numero, nome) in enumerate(CAIXAS_ESTADO):
            x = MARGEM + i * (largura + 4)
            if numero == GESTO_CONCLUIDO:
                cor = VERMELHO if alerta.ativo else ESCURO
            else:
                cor = VERDE if numero == estado and mao is not None else ESCURO
            cv2.rectangle(img, (x, y), (x + largura, y + 30), cor, -1)
            tinta = (20, 20, 20) if cor == VERDE else BRANCO
            texto(img, str(numero), x + 5, y + 13, tinta, 0.38)
            texto(img, nome, x + 5, y + 26, tinta, 0.38)
        y += 50
        if alerta.ativo:
            texto(img, "ALERTA DISPARADO!", MARGEM, y, VERMELHO, 0.55, 2)
        elif mao is None:
            texto(img, "Procurando uma mao na imagem...", MARGEM, y, CINZA, 0.48)
        else:
            texto(img, ESPERANDO[estado], MARGEM, y, AMARELO, 0.48)
        y += 18
        if mao is not None and estado > REPOUSO:
            limite = self.cfg.timeout_estado
            sobra = max(0.0, limite - (tempo - mao.mef.tempo_estado))
            medidor(img, MARGEM, y - 8, 250, sobra, 0, limite, [], LARANJA)
            texto(img, f"{sobra:.1f} s para a proxima etapa", MARGEM + 258, y, CINZA, 0.36)
        return y + 18

    def _secao_leitura(self, img, y, mao, tempo):
        titulo(img, f"LEITURA DA MAO #{mao.id}" if mao is not None else "LEITURA DA MAO", y)
        y += 22
        if mao is None:
            texto(img, "Nenhuma mao encontrada neste momento.", MARGEM, y, CINZA)
            return y + 118
        pose = classificar_pose(mao.pontos, mao.lado, self.cfg)
        for nome, abertura in zip(NOMES_DEDOS, aberturas_dedos(mao.pontos)):
            situacao, cor = estado_dedo(abertura, self.cfg)
            texto(img, nome, MARGEM, y, BRANCO, 0.42)
            medidor(img, MARGEM + 76, y - 8, 130, abertura, 0.6, 2.2, [self.cfg.dedo_dobrado, self.cfg.dedo_estendido], cor)
            texto(img, f"{abertura:.2f}", MARGEM + 214, y, BRANCO, 0.4)
            texto(img, situacao, MARGEM + 290, y, cor, 0.4)
            y += 18
        razao = razao_polegar(mao.pontos)
        if pose.quatro_dobrados:
            situacao, cor = "escondido", CINZA
        else:
            situacao, cor = ("recolhido", AZUL) if pose.polegar_recolhido else ("para fora", BRANCO)
        texto(img, "Polegar", MARGEM, y, BRANCO, 0.42)
        medidor(img, MARGEM + 76, y - 8, 130, razao, 0, 2.0, [self.cfg.limiar_polegar], cor)
        texto(img, f"{razao:.2f}", MARGEM + 214, y, BRANCO, 0.4)
        texto(img, situacao, MARGEM + 290, y, cor, 0.4)
        y += 18
        # Só é um impedimento quando a configuração exige a palma. Fora isso é informação
        vista_ruim = self.cfg.exigir_palma and not pose.palma_para_camera
        texto(img, "Vista da mao: " + ("palma" if pose.palma_para_camera else "costas"), MARGEM, y, VERMELHO if vista_ruim else BRANCO, 0.42)
        y += 20
        rotulo = rotular_pose(pose, self.cfg)
        texto(img, f"Pose: {NOMES_POSE[rotulo]}", MARGEM, y, BRANCO, 0.5, 2)
        if rotulo is not None:
            texto(img, "confirmada" if mao.mef.pose_firme(tempo) else "confirmando", MARGEM + 270, y, CINZA, 0.4)
        return y + 22

    def _secao_regioes(self, img, y, quadro, mascara):
        titulo(img, "ONDE ESTA OLHANDO", y)
        topo = y + 14
        y += 26
        if not quadro.regioes:
            texto(img, "Nada se mexeu, so esperando", MARGEM, y, CINZA, 0.42)
            y += 18
        for regiao in quadro.regioes:
            texto(img, NOMES_ORIGEM[regiao.origem], MARGEM, y, CORES_ORIGEM[regiao.origem], 0.42)
            y += 18
        if quadro.clareado:
            texto(img, "Cena escura: quadro clareado", MARGEM, y, AMARELO, 0.42)
        if mascara is not None and self.mostrar_mascara:
            # Branco é o que mudou em relação ao fundo da cena
            pequena = cv2.cvtColor(cv2.resize(mascara, (160, 90), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            x = LARGURA_LATERAL - MARGEM - 160
            img[topo:topo + 90, x:x + 160] = pequena
            cv2.rectangle(img, (x, topo), (x + 160, topo + 90), ESCURO, 1)
        return topo + 112

    def _secao_eventos(self, img, y):
        titulo(img, "ULTIMOS EVENTOS", y)
        y += 22
        for linha in list(self.eventos)[:5]:
            cor = VERMELHO if "CONCLUIDO" in linha else BRANCO
            texto(img, linha[:58], MARGEM, y, cor, 0.38)
            y += 17
        return max(y, 0) + (5 - min(len(self.eventos), 5)) * 17 + 8

    def _secao_desempenho(self, img, y, medidas):
        titulo(img, "DESEMPENHO", y)
        y += 22
        texto(img, f"{medidas['fps']:.1f} FPS", MARGEM, y, BRANCO, 0.5, 2)
        texto(img, f"{medidas['ms_total']:.0f} ms por quadro", MARGEM + 110, y, BRANCO, 0.42)
        y += 20
        texto(img, f"Localizar {medidas['ms_localizar']:.1f} ms   MediaPipe {medidas['ms_mediapipe']:.1f} ms   Gesto {medidas['ms_gesto']:.2f} ms", MARGEM, y, CINZA, 0.37)
        y += 18
        texto(img, "q sai | p pausa | m mostra o movimento", MARGEM, y, CINZA, 0.37)
