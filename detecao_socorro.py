import argparse
import os
import threading
import time

import cv2

from alerta import Alerta
from gesto import RastreadorMaos
from painel import Painel
from pipeline import DetectorMaos

JANELA = "Sistema de Monitoramento (Movimento + MediaPipe)"
FALHAS_PARA_RECONECTAR = 60
ESPERA_RECONEXAO = 2.0  # Segundos


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Monitoramento: detecção de movimento + MediaPipe + máquina de estados")
    parser.add_argument("--fonte", default="0", help="Índice da câmera, arquivo de vídeo ou endereço, como rtsp://... (padrão: 0)")
    parser.add_argument("--localizador", default="movimento", choices=["movimento", "yolo", "inteiro"],
                        help="Como escolher a região da imagem. O padrão é o sistema; yolo e inteiro servem para comparar")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela de vídeo, só alerta e registra")
    parser.add_argument("--sem-som", action="store_true")
    return parser.parse_args()


class VideoArquivo:
    def __init__(self, caminho):
        self.cap = cv2.VideoCapture(caminho)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def aberto(self):
        return self.cap.isOpened()

    def ler(self):
        return self.cap.read()

    def liberar(self):
        self.cap.release()


class CameraAoVivo:
    """Lê a câmera em outra thread e guarda só o quadro mais recente.

    Sem isso, em câmera IP os quadros que o sistema não consegue processar a tempo ficam na fila
    e o vídeo vai atrasando cada vez mais. Se a conexão cair, tenta conectar de novo.
    """

    def __init__(self, fonte):
        self.fonte = fonte
        self.trava = threading.Lock()
        self.novo = threading.Event()
        self.quadro = None
        self.ativa = True
        self.cap = self._abrir()
        self.thread = threading.Thread(target=self._ler_sempre, daemon=True)
        self.thread.start()

    def _abrir(self):
        if isinstance(self.fonte, str) and self.fonte.startswith("rtsp"):
            # TCP perde menos imagem do que UDP em rede com falhas
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.fonte)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(self.fonte, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _ler_sempre(self):
        falhas = 0
        while self.ativa:
            sucesso, frame = self.cap.read()
            if sucesso:
                falhas = 0
                with self.trava:
                    self.quadro = frame
                self.novo.set()
                continue
            falhas += 1
            time.sleep(0.03)
            if falhas >= FALHAS_PARA_RECONECTAR:
                print("Câmera sem imagem, tentando conectar de novo...", flush=True)
                self.cap.release()
                time.sleep(ESPERA_RECONEXAO)
                self.cap = self._abrir()
                falhas = 0

    def aberto(self):
        return self.cap.isOpened()

    def ler(self, espera=10.0):
        if not self.novo.wait(espera):
            return False, None
        with self.trava:
            self.novo.clear()
            return True, self.quadro

    def liberar(self):
        self.ativa = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def main():
    args = ler_argumentos()
    webcam = args.fonte.isdigit()
    ao_vivo = webcam or "://" in args.fonte

    # Abre a câmera antes de carregar os modelos: se o endereço estiver errado, avisa na hora
    fonte = CameraAoVivo(int(args.fonte) if webcam else args.fonte) if ao_vivo else VideoArquivo(args.fonte)
    if not fonte.aberto():
        fonte.liberar()
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {args.fonte}")

    print("Carregando modelos...")
    detector = DetectorMaos(localizador=None if args.localizador == "inteiro" else args.localizador)
    try:
        monitorar(args, fonte, detector, webcam, ao_vivo)
    finally:
        fonte.liberar()
        detector.fechar()
        cv2.destroyAllWindows()


def monitorar(args, fonte, detector, webcam, ao_vivo):
    rastreador = RastreadorMaos()
    alerta = Alerta(com_som=not args.sem_som)
    painel = Painel(rastreador.cfg, time.monotonic() if ao_vivo else 0.0)
    if not args.sem_janela:
        cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    print("Iniciando monitoramento. Pressione 'q' na janela para sair.")

    indice = 0
    fps = 0.0
    pausado = False
    while True:
        inicio = time.perf_counter()
        sucesso, frame = fonte.ler()
        if not sucesso:
            print("Fim do vídeo." if not ao_vivo else "A câmera parou de enviar imagem.")
            break

        if ao_vivo:
            tempo = time.monotonic()
        else:
            # Em arquivo o tempo vem do próprio vídeo, assim o resultado não depende da velocidade do PC
            tempo = indice / fonte.fps
        if webcam:
            frame = cv2.flip(frame, 1)
        indice += 1

        quadro = detector.processar(frame)
        antes_gesto = time.perf_counter()
        concluidas = rastreador.atualizar([(pontos, lado) for pontos, lado, _ in quadro.maos], tempo)
        ms_gesto = (time.perf_counter() - antes_gesto) * 1000
        for mao in concluidas:
            alerta.disparar(mao.id, mao.pontos)
        painel.registrar(rastreador, concluidas, tempo)

        ms_total = (time.perf_counter() - inicio) * 1000
        fps = 0.9 * fps + 0.1 * 1000 / max(ms_total, 1e-3)
        if args.sem_janela:
            continue

        mascara = detector.movimento.mascara if detector.movimento is not None else None
        medidas = {"fps": fps, "ms_total": ms_total, "ms_localizar": quadro.ms_localizar, "ms_mediapipe": quadro.ms_mediapipe, "ms_gesto": ms_gesto}
        cv2.imshow(JANELA, painel.desenhar(frame, quadro, rastreador, alerta, mascara, medidas, tempo))

        # Em arquivo espera o tempo do vídeo, para dar para acompanhar a tela
        restante = 1000 / fonte.fps - (time.perf_counter() - inicio) * 1000 if not ao_vivo else 1
        tecla = cv2.waitKey(max(1, int(restante))) & 0xFF
        while tecla == ord("p") or pausado:
            if tecla == ord("p"):
                pausado = not pausado
            if not pausado:
                break
            tecla = cv2.waitKey(50) & 0xFF
            if tecla == ord("q"):
                break
        if tecla == ord("m"):
            painel.mostrar_mascara = not painel.mostrar_mascara
        if tecla == ord("q"):
            break


if __name__ == "__main__":
    main()
