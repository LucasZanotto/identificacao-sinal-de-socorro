"""Mecanismo de alerta: faixa na tela, som e registro em arquivo. Nenhuma imagem é salva."""
import csv
import os
import sys
import threading
import time
from datetime import datetime

import cv2


class Alerta:
    def __init__(self, caminho_log="alertas.csv", duracao_banner=4.0, com_som=True):
        self.caminho_log = caminho_log
        self.duracao_banner = duracao_banner
        self.com_som = com_som
        self.fim_banner = 0.0

    @property
    def ativo(self):
        return time.monotonic() < self.fim_banner

    def disparar(self, id_mao, pontos):
        self.fim_banner = time.monotonic() + self.duracao_banner
        self._registrar(id_mao, pontos)
        if self.com_som:
            threading.Thread(target=self._tocar_som, daemon=True).start()
        print(f"[ALERTA] Sinal de socorro detectado (mão #{id_mao})")

    def _registrar(self, id_mao, pontos):
        novo = not os.path.exists(self.caminho_log)
        with open(self.caminho_log, "a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            if novo:
                escritor.writerow(["data_hora", "mao", "pulso_x", "pulso_y"])
            escritor.writerow([datetime.now().isoformat(timespec="seconds"), id_mao, int(pontos[0][0]), int(pontos[0][1])])

    @staticmethod
    def _tocar_som():
        if sys.platform == "win32":
            import winsound
            for _ in range(3):
                winsound.Beep(1200, 250)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def desenhar(self, frame):
        if self.ativo:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 255), -1)
            texto = "!!! ALERTA: SINAL DE SOCORRO DETECTADO !!!"
            # Reduz o texto quando o vídeo é estreito, como o de um celular em pé
            (largura, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            escala = min(1.1, 1.1 * (frame.shape[1] - 40) / largura)
            cv2.putText(frame, texto, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 3 if escala > 0.7 else 2, cv2.LINE_AA)
