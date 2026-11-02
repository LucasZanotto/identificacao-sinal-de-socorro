"""Gravação guiada do dataset. O nome de cada arquivo já diz o que a pessoa fez, então não
é preciso rotular nada à mão.

Estrutura gerada:  dataset/<pessoa>/<cenario>/<acao>_<nn>.mp4  e  dataset/pessoas.csv
Vídeos gravados no celular também servem, basta colocar nas mesmas pastas com os mesmos nomes.
"""
import argparse
import csv
import os
import time

import cv2

ACAO_POSITIVA = "socorro"
DURACAO_CLIPE = 6.0  # Segundos
CONTAGEM = 3  # Segundos de preparação antes de gravar

CONTROLES_LUZ = [("acenar", 1), ("apontar", 1), ("cocar_cabeca", 1), ("abrir_fechar", 1), ("fechar_colado", 1)]

# (cenário, orientação, [(ação, repetições)])
ROTEIRO = [
    ("plena", "Luz boa, a cerca de 1,5 m da camera", [(ACAO_POSITIVA, 4)] + CONTROLES_LUZ),
    ("baixa", "Pouca luz (apague a luz principal), a 1,5 m", [(ACAO_POSITIVA, 4)] + CONTROLES_LUZ),
    ("contraluz", "Janela ou luz forte atras da pessoa, a 1,5 m", [(ACAO_POSITIVA, 4)] + CONTROLES_LUZ),
    ("oclusao", "Luz boa, com copo ou manga cobrindo parte da mao", [(ACAO_POSITIVA, 3), ("segurar_copo", 1), ("ajeitar_manga", 1)]),
    ("brusco", "Luz boa, mao em movimento rapido", [(ACAO_POSITIVA, 3), ("mexer_rapido", 1), ("abrir_fechar", 1)]),
    ("longe", "Luz boa, a 3 ou 4 m da camera", [(ACAO_POSITIVA, 3), ("acenar", 1), ("abrir_fechar", 1)]),
    ("virada", "Luz boa, a 1,5 m. Palma virada uns 45 graus para o lado, sem olhar para a camera", [(ACAO_POSITIVA, 3), ("acenar", 1), ("abrir_fechar", 1)]),
    ("costas", "Luz boa, a 1,5 m. Costas da mao para a camera", [(ACAO_POSITIVA, 3), ("abrir_fechar", 2)]),
    ("manga_longe", "Luz boa, a 3 ou 4 m, com a manga cobrindo parte da palma", [(ACAO_POSITIVA, 3), ("ajeitar_manga", 1), ("abrir_fechar", 1)]),
    ("camera_alta", "Luz boa, camera no alto (uns 2,5 m) inclinada para baixo, a 2 ou 3 m", [(ACAO_POSITIVA, 3), ("acenar", 1), ("abrir_fechar", 1)]),
]

INSTRUCOES = {
    ACAO_POSITIVA: "Faca o SINAL: mao aberta, dobre o polegar, feche os dedos",
    "acenar": "Acene com a mao aberta",
    "apontar": "Aponte para alguma coisa com o indicador",
    "cocar_cabeca": "Coce a cabeca ou arrume o cabelo",
    "abrir_fechar": "Abra e feche a mao normalmente, com o polegar por fora",
    "fechar_colado": "Mao reta com o polegar encostado no indicador, feche normalmente",
    "segurar_copo": "Segure e movimente um copo",
    "ajeitar_manga": "Ajeite a manga da roupa",
    "mexer_rapido": "Mexa a mao aberta rapidamente, sem fazer o sinal",
}


def listar_clipes():
    clipes = []
    for cenario, orientacao, acoes in ROTEIRO:
        for acao, repeticoes in acoes:
            for n in range(1, repeticoes + 1):
                clipes.append((cenario, orientacao, acao, n))
    return clipes


def registrar_pessoa(dataset, pessoa, fitzpatrick):
    os.makedirs(dataset, exist_ok=True)
    caminho = os.path.join(dataset, "pessoas.csv")
    pessoas = {}
    if os.path.exists(caminho):
        with open(caminho, newline="", encoding="utf-8") as arquivo:
            pessoas = {linha["pessoa"]: linha["fitzpatrick"] for linha in csv.DictReader(arquivo)}
    pessoas[pessoa] = fitzpatrick
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["pessoa", "fitzpatrick"])
        escritor.writerows(sorted(pessoas.items()))


def medir_fps(cap, quadros=20):
    inicio = time.perf_counter()
    for _ in range(quadros):
        cap.read()
    # O limite evita um valor absurdo se a fonte não for uma câmera (o gravador de vídeo trava com ele)
    return min(quadros / max(time.perf_counter() - inicio, 1e-6), 60.0)


def gravar_clipe(cap, caminho, duracao, fps, mostrar=None):
    """Grava o clipe sem nenhuma marcação por cima da imagem."""
    sucesso, frame = cap.read()
    if not sucesso:
        return 0
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    altura, largura = frame.shape[:2]
    escritor = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
    total = 0
    inicio = time.perf_counter()
    while sucesso and (time.perf_counter() - inicio) < duracao:
        escritor.write(frame)
        total += 1
        if mostrar:
            mostrar(frame, f"GRAVANDO {duracao - (time.perf_counter() - inicio):.0f}s")
        sucesso, frame = cap.read()
    escritor.release()
    return total


def escrever(frame, linhas, cor=(255, 255, 255)):
    tela = frame.copy()
    cv2.rectangle(tela, (0, 0), (tela.shape[1], 40 + 34 * len(linhas)), (0, 0, 0), -1)
    for i, linha in enumerate(linhas):
        cv2.putText(tela, linha, (15, 35 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2)
    return tela


def main():
    parser = argparse.ArgumentParser(description="Gravação guiada do dataset")
    parser.add_argument("--pessoa", required=True, help="Código da pessoa, por exemplo P01 (não use o nome). Segunda sessão da mesma pessoa: P01_s2")
    parser.add_argument("--fitzpatrick", required=True, choices=["I", "II", "III", "IV", "V", "VI"])
    parser.add_argument("--fonte", default="0", help="Índice da câmera ou URL do celular (padrão: 0)")
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a câmera: {args.fonte}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = medir_fps(cap)
    print(f"Câmera a {fps:.1f} quadros por segundo")

    registrar_pessoa(args.dataset, args.pessoa, args.fitzpatrick)
    janela = "Gravacao guiada"

    def mostrar(frame, texto):
        cv2.imshow(janela, escrever(frame, [texto], (0, 0, 255)))
        cv2.waitKey(1)

    clipes = listar_clipes()
    indice = 0
    while indice < len(clipes):
        cenario, orientacao, acao, n = clipes[indice]
        caminho = os.path.join(args.dataset, args.pessoa, cenario, f"{acao}_{n:02d}.mp4")
        # Permite parar e continuar depois de onde parou
        if os.path.exists(caminho):
            indice += 1
            continue

        tecla = -1
        while tecla not in (ord(" "), ord("q"), ord("r")):
            sucesso, frame = cap.read()
            if not sucesso:
                raise SystemExit("A câmera parou de enviar imagem")
            feitos = sum(os.path.exists(os.path.join(args.dataset, args.pessoa, c, f"{a}_{m:02d}.mp4")) for c, _, a, m in clipes)
            cv2.imshow(janela, escrever(frame, [
                f"{args.pessoa}  clipe {feitos + 1} de {len(clipes)}  cenario: {cenario}",
                orientacao,
                f"Acao: {INSTRUCOES[acao]}",
                "ESPACO grava | R refaz o anterior | Q sai",
            ]))
            tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        if tecla == ord("r"):
            if indice > 0:
                indice -= 1
                anterior = clipes[indice]
                refazer = os.path.join(args.dataset, args.pessoa, anterior[0], f"{anterior[2]}_{anterior[3]:02d}.mp4")
                if os.path.exists(refazer):
                    os.remove(refazer)
            continue

        fim_contagem = time.perf_counter() + CONTAGEM
        while time.perf_counter() < fim_contagem:
            sucesso, frame = cap.read()
            if sucesso:
                mostrar(frame, f"Prepare-se: {fim_contagem - time.perf_counter():.0f}s  ({INSTRUCOES[acao]})")

        total = gravar_clipe(cap, caminho, DURACAO_CLIPE, fps, mostrar)
        print(f"{caminho}: {total} quadros")
        indice += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Pronto: {args.pessoa} com {sum(os.path.exists(os.path.join(args.dataset, args.pessoa, c, f'{a}_{m:02d}.mp4')) for c, _, a, m in clipes)} de {len(clipes)} clipes.")


if __name__ == "__main__":
    main()
