# Identificação contínua do Sinal de Pedido de Socorro

Sistema de visão computacional que acompanha o vídeo de uma câmera comum e avisa quando alguém faz o
**Sinal Universal de Pedido de Socorro** ([*Signal for Help*](https://en.wikipedia.org/wiki/Signal_for_Help)),
o gesto criado pela Canadian Women's Foundation para pedir ajuda sem falar nada.

Roda só na CPU, sem placa de vídeo e sem internet. Nenhuma imagem é gravada durante o monitoramento.

TCC de Arthur Romanatto Moro e Lucas Abati Zanotto, com orientação de Herculano Haymussi De Biasi (Unoesc, 2026).

| Durante o gesto | Gesto reconhecido |
|---|---|
| ![Painel do sistema com o polegar dobrado e os quatro dedos esticados](docs/painel_polegar.jpg) | ![Painel do sistema com o alerta disparado depois que a mão fechou](docs/painel_alerta.jpg) |

<sub>Imagens do sistema rodando sobre o vídeo [Signal for Help (demonstrative example)](https://commons.wikimedia.org/wiki/File:Signal_for_Help(_demonstrative_example_).ogg), de GemoWiki, licença CC BY 4.0, com o painel acrescentado.</sub>

## O gesto

1. Mão aberta, com os dedos esticados.
2. O polegar dobra sobre a palma.
3. Os outros quatro dedos fecham por cima do polegar.

O que importa é a ordem. Quem fecha a mão normalmente dobra os dedos com o polegar ainda para fora, e nesse caso o sistema não dispara.

## Baixar e instalar

Você vai precisar de:

- [Python 3.12](https://www.python.org/downloads/release/python-31210/), a versão usada nos testes. As dependências pedem no mínimo a 3.11. No Windows, marque "Add python.exe to PATH" na instalação.
- Git, ou baixe o projeto pelo botão **Code > Download ZIP** aqui do GitHub.
- Uma webcam, um celular ou um vídeo gravado. Placa de vídeo não é necessária.

Baixe o projeto e crie um ambiente só para ele:

```bash
git clone https://github.com/LucasZanotto/identificacao-sinal-de-socorro.git
cd identificacao-sinal-de-socorro
python -m venv .venv
```

Ative o ambiente com o comando do seu terminal:

```bash
# Windows, PowerShell
.venv\Scripts\Activate.ps1

# Windows, Prompt de Comando
.venv\Scripts\activate.bat

# Linux e macOS
source .venv/bin/activate
```

Instale as bibliotecas:

```bash
pip install -r requirements.txt
```

O modelo do MediaPipe (`hand_landmarker.task`) já vem no repositório. O sistema só foi testado no Windows 11.

## Rodar

```bash
python detecao_socorro.py
```

Isso abre a câmera padrão e uma janela com o vídeo e o painel. Sobre o vídeo aparece só a mão, com o quadrado verde. Faça o gesto olhando o painel. Quando ele é reconhecido,
aparece uma faixa vermelha, toca um som e fica um registro em `alertas.csv`, com a data, a hora e a posição da mão.

Teclas: `q` sai, `p` pausa e `m` mostra ou esconde a imagem do movimento.

### Ver tudo o que o sistema examina

```bash
python detecao_socorro_profundo.py
```

É o mesmo programa, com as mesmas opções e o mesmo resultado. A diferença é o que aparece na tela: ele desenha também os quadrados laranja das regiões que se mexeram e o quadro inteiro quando ele é examinado, e lista essas regiões no painel. Serve para entender por que uma mão foi ou não encontrada.

### Outras fontes de vídeo

| Fonte | Comando |
|---|---|
| Outra câmera do computador | `python detecao_socorro.py --fonte 1` |
| Vídeo gravado | `python detecao_socorro.py --fonte video.mp4` |
| Celular com [DroidCam](https://www.droidcam.app/) | `python detecao_socorro.py --fonte http://IP_DO_CELULAR:4747/video` |
| Câmera de segurança IP | `python detecao_socorro.py --fonte rtsp://usuario:senha@IP_DA_CAMERA:554/stream1` |

Com celular e câmera IP, o sistema lê a imagem em paralelo e usa sempre o quadro mais recente, então o vídeo não acumula
atraso, e tenta conectar de novo se a conexão cair. `--sem-janela` e `--sem-som` desligam a tela e o som.
`--localizador inteiro` e `--localizador yolo` trocam a detecção de movimento por outra forma de procurar a mão, só para comparar.

### Ver funcionando sem câmera

O primeiro comando baixa um vídeo público do gesto para `testes/dados` e confere o sistema nele. O segundo mostra o vídeo com o painel. O alerta deve aparecer duas vezes, perto de 1,3 s e de 4,2 s.

```bash
python -m pytest testes/test_video_real.py
python detecao_socorro.py --fonte testes/dados/signal_for_help_wikimedia.ogg
```

### O que o painel mostra

- **No vídeo:** o quadrado verde da mão acompanhada e os pontos da mão. Dedo verde está esticado, vermelho está dobrado e amarelo está em dúvida. O polegar fica azul quando está recolhido.
- **Sequência do gesto:** em que etapa está, o que ele espera agora e quanto tempo falta para desistir.
- **Leitura da mão:** a medida de cada dedo e do polegar, com os limites que decidem a pose, e a pose que ele entendeu.
- **Onde está olhando:** se há mão acompanhada, com a imagem do que se mexeu. A versão profunda lista também as regiões em movimento.
- **Últimos eventos:** cada mudança de etapa, com o motivo quando a sequência volta ao início. O terminal mostra o mesmo.
- **Desempenho:** FPS e o tempo de cada parte.

## Como funciona

Cada quadro passa pelas etapas abaixo e depois é descartado.

```mermaid
flowchart TD
    subgraph V["Onde olhar e extrair os pontos"]
        direction LR
        A["Quadro da câmera"] --> B["Correção de luz"]
        B -- "procura mãos novas" --> C["Detecção de movimento"]
        C --> E["MediaPipe, modo imagem:<br>palma e 21 pontos"]
        B -- "mão já encontrada" --> D["Recorte que segue a mão"]
        D --> F["MediaPipe, modo de vídeo:<br>só os 21 pontos"]
        E -- "abre um recorte" --> D
    end
    subgraph P["Decidir se é o gesto"]
        direction LR
        G["Geometria:<br>duas medidas e a pose"] --> H["Máquina de estados:<br>ordem do gesto"]
        H -- "sequência completa" --> I["Alerta"]
    end
    V -- "21 pontos de cada mão" --> P
```

**1. Correção de luz.** Se o brilho médio do quadro fica abaixo de 90, em uma escala de 0 a 255, a imagem é clareada com correção gama. Quadro claro passa direto.

**2. Onde procurar.** O detector de palma do MediaPipe reduz a imagem para 192x192 pixels, e nessa redução uma mão a 3 ou 4 metros praticamente some. Para contornar isso, o MOG2 do OpenCV compara cada quadro com um modelo do fundo da cena e aponta o que se mexeu. Recortar essa região funciona como um zoom. Essa etapa não usa rede neural. Quando nada se mexe, o MediaPipe olha o quadro inteiro a cada 5 quadros, o que cobre uma mão parada perto da câmera.

**3. Pontos da mão.** O MediaPipe Hand Landmarker devolve 21 pontos de cada mão encontrada: pulso, articulações e pontas dos dedos.

**4. Acompanhar a mão.** Cada mão encontrada ganha um recorte só dela, que a segue nos quadros seguintes, e o modo de vídeo do MediaPipe passa a acompanhá-la usando a posição anterior. O detector de palma deixa de ser necessário, e é ele que costuma perder a mão virada, fechada ou coberta. O sistema acompanha até 2 mãos por vez.

**5. Geometria.** Os pontos viram duas medidas. Como são razões entre distâncias, o valor não muda com a distância da câmera.

| Medida | Cálculo | Leitura |
|---|---|---|
| Abertura de cada dedo | distância da ponta ao pulso ÷ distância da base do dedo ao pulso | 1,5 ou mais é esticado, 1,4 ou menos é dobrado. Fica perto de 1,8 com o dedo esticado e de 0,9 com o dedo fechado |
| Polegar | distância da ponta do polegar à base do dedo mínimo ÷ largura da palma | Menos de 1,1 é recolhido |

Com isso cada quadro recebe uma pose: **aberta** (quatro dedos esticados e polegar para fora), **polegar** (quatro dedos esticados e polegar recolhido), **punho** (quatro dedos dobrados) ou nenhuma. A palma não precisa estar de frente, então as costas da mão também contam. No punho o polegar não é conferido, porque ele fica escondido e o modelo só estima a posição.

**6. Máquina de estados.** A pose de um quadro não diz nada sozinha. O que identifica o sinal é a sequência, conferida por uma máquina de estados para cada mão.

```mermaid
stateDiagram-v2
    state "0 Repouso" as repouso
    state "1 Mão aberta" as aberta
    state "2 Polegar recolhido" as polegar
    state "3 Gesto concluído" as fim
    repouso --> aberta: mão aberta
    repouso --> polegar: mão levantada com o polegar já dobrado
    aberta --> polegar: polegar dobra
    polegar --> aberta: polegar abre por 0,3 s
    polegar --> fim: dedos fecham
    aberta --> repouso: punho sem passar pelo polegar
    fim --> repouso: alerta disparado
```

Três tempos protegem essa sequência:

| Parâmetro | Valor | Para que serve |
|---|---|---|
| `tempo_confirmacao` | 0,13 s | A pose precisa durar isso para valer, o que ignora leituras erradas em quadros borrados. É contado em tempo, então funciona igual em qualquer FPS |
| `tempo_cancelar` | 0,3 s | A mão aberta precisa durar isso para desfazer o polegar recolhido, porque a imagem borra enquanto os dedos fecham rápido |
| `timeout_estado` | 3 s | Sem ver a pose da etapa atual por esse tempo, a sequência volta ao início |

**7. Alerta.** Faixa vermelha na tela, som e uma linha em `alertas.csv`. O aviso fica só no computador de monitoramento, porque um alarme no próprio ambiente filmado poderia denunciar a vítima a quem está perto dela. Ele é um pedido de verificação para uma pessoa, não a confirmação de uma ocorrência.

### Sobre redes neurais

O trabalho não treina nenhuma rede neural, e a decisão do gesto é só geometria. Os dois modelos do MediaPipe, o detector de palma e o de pontos, são redes convolucionais pequenas usadas prontas, apenas para extrair os pontos. O YOLOv8n não faz parte do sistema: ele foi usado em um protótipo anterior e hoje serve só para comparar o gasto na avaliação.

## Gravar os vídeos de teste

Cada sessão grava 62 clipes de 6 segundos (33 com o sinal e 29 de controle), em cerca de 12 minutos. O programa diz o que fazer em cada clipe e já salva com o nome certo.

```bash
python gravar_dataset.py --pessoa P01 --fitzpatrick III --dataset C:\tcc\dataset
```

- `--pessoa` é um código, nunca o nome. Neste trabalho, P01 e P01_s2 são as duas sessões de um dos autores e P02 é o outro autor. O sufixo `_s2` faz o relatório entender que é a mesma pessoa.
- `--fitzpatrick` é o tom de pele na escala de Fitzpatrick (I a VI). É obrigatório no comando e só serve para agrupar o relatório.
- Com o celular como câmera, acrescente `--fonte http://IP_DO_CELULAR:4747/video`.
- Grave fora de pastas sincronizadas, como OneDrive e Google Drive, para os vídeos não subirem para a nuvem.

`ESPAÇO` grava, `R` refaz o clipe anterior e `Q` sai. Se parar no meio, o mesmo comando continua de onde parou, e para refazer um clipe basta apagar o arquivo dele. Os cenários e as ações estão na lista `ROTEIRO` de `gravar_dataset.py`.

Vídeos gravados direto no celular também servem, nas mesmas pastas e com os mesmos nomes:

```
dataset/pessoas.csv                       (colunas: pessoa,fitzpatrick)
dataset/P01/plena/socorro_01.mp4
dataset/P01/plena/acenar_01.mp4
```

## Avaliar

```bash
python avaliar.py --dataset C:\tcc\dataset --calibrar P01 --distancia
```

- `--calibrar P01` usa os clipes dessa pessoa só para ajustar os limites, e eles ficam fora do teste. Sem essa opção, o relatório é só uma prévia.
- `--distancia` encolhe a imagem como se a pessoa estivesse mais longe, para descobrir até que tamanho de mão o sistema funciona. Essa parte demora, porque processa os vídeos de novo.
- Para a avaliação comparar o gasto com o YOLO, rode antes `pip install -r requirements-comparacao.txt`. Sem isso ela funciona e pula essa comparação.

Os pontos extraídos ficam em `dataset/_landmarks`, então rodar de novo é rápido. A saída fica em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | Todas as tabelas: métricas com intervalo de confiança, cenários, comparações, estresse e erros |
| `por_clipe.csv` | O resultado de cada clipe |
| `custo_localizador.png` | Gasto da detecção de movimento e do YOLO |
| `matriz_confusao.png`, `metricas_por_cenario.png` | As outras figuras |

Antes de contar um alarme como erro, assista ao clipe. É fácil fazer o sinal sem querer em um clipe de controle.

## Testes automáticos

```bash
python -m pytest testes
```

Os testes da lógica usam uma mão sintética e não precisam de câmera. `test_video_real.py` baixa o vídeo público do gesto e roda o sistema inteiro nele. Esses testes conferem o programa, mas não medem a precisão em pessoas reais. Isso é papel do `avaliar.py`.

## Arquivos do projeto

| Arquivo | O que faz |
|---|---|
| `detecao_socorro.py` | Programa principal: lê o vídeo e liga todas as partes |
| `detecao_socorro_profundo.py` | O mesmo programa, mostrando também as regiões que se mexeram |
| `pipeline.py` | Correção de luz, detecção de movimento, MediaPipe e os recortes que seguem cada mão |
| `gesto.py` | Medidas geométricas, poses e máquina de estados. Não depende de câmera nem de modelo |
| `painel.py` | A tela que mostra o que o sistema está pensando |
| `alerta.py` | Faixa, som e registro em `alertas.csv` |
| `gravar_dataset.py` | Gravação guiada dos clipes |
| `avaliar.py` | Avaliação completa e relatório |
| `testes/` | Testes automáticos |
| `hand_landmarker.task` | Modelo pronto do MediaPipe |
| `yolov8n.pt` | Modelo pronto do YOLOv8n, só para a comparação de gasto |
| `docs/resultados.md` | Todos os números da medição final, com as tabelas que não couberam no texto do TCC |

## Resultados

> **Medição final com duas pessoas**, os próprios autores, com o mesmo tom de pele e que já conheciam o gesto. Os 58 clipes da primeira sessão de um dos autores serviram só para calibrar. O teste usa 124 clipes gravados depois de o código e os limites estarem fechados, 62 de cada autor. Os resultados não valem para outras pessoas nem para outros tons de pele.

Esta seção é um resumo. Todas as tabelas, inclusive os resultados por cenário, a análise de cada erro, a mudança de cada parâmetro e o resultado de cada clipe, estão em [docs/resultados.md](docs/resultados.md).

O sistema reconheceu **53 de 66 sinais** e disparou em **2 de 58 clipes de controle**:

| Métrica | Valor | Intervalo de confiança de 95% |
|---|---|---|
| Acurácia | 87,9% | 81,0 a 92,5 |
| Precisão | 96,4% | 87,7 a 99,0 |
| Sensibilidade | 80,3% | 69,2 a 88,1 |
| Especificidade | 96,6% | 88,3 a 99,0 |
| F1-Score | 87,6% | 80,7 a 93,3 |

A diferença entre as duas pessoas foi grande. Em uma gravação o sistema reconheceu os 33 sinais (98,4% de acurácia). Na outra reconheceu 20 de 33 (77,4%), porque a mão aparecia bem menor na imagem, a cena de pouca luz ficou quase escura e o movimento brusco foi mais amplo. Os quadros de todos os erros foram conferidos, e em todos os sinais perdidos o gesto foi feito.

**Comparação com regras mais simples, nos mesmos clipes:**

| Regra de decisão | Acurácia | Sensibilidade | Especificidade | McNemar p |
|---|---|---|---|---|
| Sistema proposto | 87,9% | 80,3% | 96,6% | referência |
| Protótipo anterior | 87,9% | 84,8% | 91,4% | 1,0000 |
| Abre e fecha, sem olhar o polegar | 71,0% | 84,8% | 55,2% | menor que 0,0001 |
| Só punho, sem sequência | 61,3% | 89,4% | 29,3% | menor que 0,0001 |

As regras simples pegam mais sinais, mas disparam à toa em 26 e 41 dos 58 controles.

**Onde o MediaPipe olha:** o YOLO reconheceu 5 sinais a mais que a detecção de movimento, todos com a mão pequena (McNemar p = 0,0625). No quadro inteiro, a sensibilidade caiu para 77,3%.

**Rastreamento:** sem ele, a sensibilidade cai para 69,7% (McNemar p = 0,0391).

**Correção de luz:** na cena quase escura, sem ela a sensibilidade cai de 87,5% para 62,5%.

**Gasto**, sem placa de vídeo, em um Ryzen 5 4600G com 16 GB, com o vídeo chegando a 15 quadros por segundo. A CPU é o percentual do processador inteiro:

| Configuração | Tempo por quadro | FPS máximo | CPU | Memória |
|---|---|---|---|---|
| **Movimento + MediaPipe (o sistema)** | **22,5 ms** | **44,5** | **3,9%** | **309 MB** |
| O mesmo, sem o rastreamento | 30,9 ms | 32,4 | 4,9% | 305 MB |
| MediaPipe no quadro inteiro | 17,2 ms | 58,2 | 3,5% | 303 MB |
| YOLO a cada 3 quadros + MediaPipe | 41,5 ms | 24,1 | 9,8% | 639 MB |
| YOLO em todo quadro + MediaPipe | 83,4 ms | 12,0 | 19,8% | 636 MB |

Medidas geométricas e máquina de estados levam 0,049 ms por quadro.

**Distância simulada:** com a palma a partir de 30 pixels, o sistema reconheceu 106 de 117 sinais. Entre 20 e 30 pixels caiu para 52 de 70, e abaixo de 15 pixels quase não reconhece. Na prática, a câmera precisa mostrar a palma com pelo menos uns 30 pixels.

## Limitações

- **Duas pessoas só.** Os testes usam os dois autores, que conhecem o gesto e têm o mesmo tom de pele. Os resultados não valem para outras pessoas nem para outros tons de pele.
- **Mão pequena na imagem:** abaixo de uns 30 pixels de palma, o sistema começa a perder sinais.
- **Pouca luz e movimento rápido:** a cena quase escura e o braço em movimento amplo derrubaram o reconhecimento.
- **Poucos clipes por cenário.** Cada cenário tem de 10 a 18 clipes, então os intervalos de confiança são largos.
- **Polegar escondido:** uma mão que fecha com o polegar escondido, como de costas para a câmera, pode ser confundida com o sinal, porque no punho o polegar não é conferido.
- **Câmera de segurança ainda não foi testada.** Os testes usaram câmeras comuns, em um computador mais potente do que um dispositivo embarcado.
- **Clipes curtos:** a medição não diz quantos alarmes falsos apareceriam em horas de monitoramento.
- **Mão totalmente de lado:** os dedos se escondem uns atrás dos outros, então o gesto tende a falhar.
- **Câmera fixa:** a detecção de movimento pressupõe que a câmera não se mexe.
- **Só este gesto.** O sistema também não sabe se a pessoa está fazendo o gesto de brincadeira, por isso o alerta precisa de verificação humana.

## Problemas comuns

| O que aparece | O que fazer |
|---|---|
| `python` não é reconhecido | Instale o Python marcando "Add python.exe to PATH", ou use `py -3.12` no lugar de `python` |
| Erro de instalação em `matplotlib` ou outra biblioteca | Confira a versão com `python --version` e use a 3.12 |
| `Activate.ps1` bloqueado por "execução de scripts" | Rode `Set-ExecutionPolicy -Scope Process RemoteSigned`, que só vale para aquela janela, ou use o Prompt de Comando |
| `Não foi possível abrir a fonte de vídeo` | Outro programa pode estar usando a câmera. Feche-o ou tente `--fonte 1`. No celular, confira o IP e se os dois aparelhos estão na mesma rede |
| `No module named 'ultralytics'` | Só acontece com `--localizador yolo`. Rode `pip install -r requirements-comparacao.txt` |
| A mão não é encontrada de longe | Deixe a câmera parada e mexa a mão, porque é o movimento que indica onde procurar. Abaixo de uns 10 pixels de palma o reconhecimento cai bastante |
| O gesto não dispara | Olhe a **Leitura da mão** no painel: ela mostra qual dedo ficou fora do limite e em que etapa a sequência parou |

## Créditos

- [Signal for Help](https://en.wikipedia.org/wiki/Signal_for_Help): gesto criado pela Canadian Women's Foundation.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), do Google, licença Apache 2.0.
- [OpenCV](https://opencv.org/), licença Apache 2.0.
- [YOLOv8n](https://github.com/ultralytics/ultralytics), da Ultralytics, licença AGPL-3.0, usado só para comparar o gasto.
- Vídeo de teste de [GemoWiki](https://commons.wikimedia.org/wiki/File:Signal_for_Help(_demonstrative_example_).ogg), licença CC BY 4.0.

Projeto acadêmico. O código ainda não tem uma licença definida.
