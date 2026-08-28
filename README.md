# FalaEdinho

Aplicação **desktop** para transcrever áudio e vídeo com **WhisperX**: legendas, texto, diarização de locutores e setup automático de FFmpeg, PyTorch (CUDA ou CPU) e WhisperX.

A interface é CustomTkinter (arrastar e soltar arquivos). O pipeline pesado roda numa thread à parte — a janela não congela.

---

## Conceitos

| Conceito | O que é |
|---|---|
| **Fila** | Um ou mais `.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a` processados em lote. |
| **Modelo Whisper** | Tamanho da rede (`tiny` … `large-v3`). Maior = melhor qualidade, mais VRAM e tempo. |
| **Mistura de faixas** | Junta streams de áudio (ex.: jogo + microfone no ShadowPlay) antes de transcrever. |
| **Diarização** | Marca quem falou (`SPEAKER_00`, …) via Pyannote. Exige token Hugging Face. |
| **Cache-first** | Modelos já baixados em `~/.cache/huggingface` são usados **sem** falar com a Hub. Só baixa se faltar arquivo. |

A pasta de cada arquivo ganha os formatos marcados (`.srt`, `.vtt`, `.txt`, `.json`, `.tsv`) e abre no explorador ao terminar.

---

## Arquitetura em runtime

```
Você  --GUI (CustomTkinter)-->  FalaEdinho
                                  │
                                  ├─ Setup (1ª vez): FFmpeg + PyTorch + WhisperX
                                  │
                                  ├─ Thread de trabalho
                                  │     WhisperX: VAD → transcrição → alinhamento → diarização → export
                                  │
                                  └─ Disco
                                        %LOCALAPPDATA%\FalaEdinho\config.json
                                        cache Hugging Face  (~/.cache/huggingface)
                                        pasta do arquivo de mídia  (saídas)
```

- A GUI **nunca** chama WhisperX no thread principal. Logs, progresso e erros vão por `queue.Queue`.
- GPU NVIDIA: o setup instala o wheel CUDA do PyTorch compatível com o driver. Sem GPU: PyTorch CPU.
- Diarização gated: token Read + aceite do modelo `pyannote/speaker-diarization-community-1` na mesma conta.

---

## Pré-requisitos

Antes de `python main.py`:

- **Python 3.10+** (3.10–3.12 são os mais estáveis para WhisperX; 3.13/3.14 podem exigir o modo de compatibilidade do setup)
- **pip** e, na primeira instalação do WhisperX, **Git** no PATH (fallback PyPI se o Git faltar)
- Windows 10/11 (alvo principal). Linux também; macOS precisa de FFmpeg via Homebrew
- GPU NVIDIA **opcional**, com driver recente, para CUDA

PyTorch e WhisperX **não** vão no `requirements.txt`: o Setup Automático instala na primeira execução.

---

## Getting started (local)

Tudo abaixo assume a raiz do repositório.

### 1. Clonar e entrar na pasta

```powershell
git clone https://github.com/<seu-usuario>/FalaEdinho.git
cd FalaEdinho
```

### 2. Criar o ambiente virtual

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Linux/macOS: `python3 -m venv .venv` e `source .venv/bin/activate`.

### 3. Subir o app

```powershell
python main.py
```

O que acontece no startup:

1. Verifica Python, FFmpeg, PyTorch e WhisperX
2. Se faltar algo, abre o **Setup Automático** (download do FFmpeg, wheel CUDA/CPU, WhisperX)
3. Se o ambiente já estiver pronto, abre a janela principal

Para **não** disparar o setup sozinho (só a tela, com o botão manual):

```powershell
$env:FALAEDINHO_NO_AUTOSETUP = "1"
python main.py
```

### 4. Primeiro uso ponta a ponta

1. Arraste um vídeo/áudio ou use **Selecionar arquivos…**
2. Escolha modelo, idioma (ou detectar automaticamente) e formatos de saída
3. Se a gravação tem duas faixas (sistema + microfone), deixe **Misturar faixas de áudio** ligado
4. Para identificar locutores: cole um token Hugging Face (Read) e marque **Identificar locutores**
5. Clique em **Iniciar Transcrição**
6. Ao terminar, a pasta do arquivo abre com `.srt` / `.txt` / etc.

Token (Read): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
Aceite o modelo na **mesma** conta: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### 5. Parar

Feche a janela. O ambiente virtual e os modelos no cache do Hugging Face permanecem no disco.

---

## Configuração

Tudo que o app persiste (token, último modelo, mistura, duração da última transcrição) fica **fora** do repositório:

| Onde | O que |
|---|---|
| `%LOCALAPPDATA%\FalaEdinho\config.json` (Windows) | Preferências e token HF |
| `%LOCALAPPDATA%\FalaEdinho\ffmpeg\` | FFmpeg baixado pelo setup, se não estava no PATH |
| `~/.cache/huggingface\` | Pesos Whisper, wav2vec2, Pyannote |

Se você já usava a pasta antiga `PyWhisperX-GUI`, o `config.json` é copiado uma vez para `FalaEdinho`.

Não commite `config.json` nem tokens. Não há `.env` obrigatório.

---

## Estrutura do repositório

```
main.py                 # Entrada; abre setup ou a janela principal
requirements.txt        # Só GUI + requests (FFmpeg/PyTorch/WhisperX pelo setup)
core/
  installer.py          # First-run: FFmpeg, PyTorch CUDA/CPU, WhisperX
  hardware.py           # nvidia-smi / WMI + métricas ao vivo CPU/GPU/RAM
  ffmpeg.py             # PATH local e download do binário
  audio.py              # ffprobe + mistura de faixas → PCM 16 kHz mono
  whisper_runner.py     # Pipeline em thread (transcrever, alinhar, diarizar, exportar)
  hf_offline.py         # Cache local primeiro; Hub só se faltar modelo
gui/
  app.py                # Janela principal
  setup_window.py       # Setup automático
  theme.py              # Dark theme + janela com Drag & Drop
  widgets.py            # Select readonly, dicas, inteiros positivos
utils/
  config.py             # config.json do usuário
  paths.py              # AppData, FFmpeg local, subprocess sem console
  events.py             # Envelope da fila GUI ↔ worker
docs/Especificação A.md
```

O WhisperX **não** é vendored: depois do setup, o `import whisperx` usa o pacote do ambiente.

---

## Troubleshooting

| Sintoma | O que checar |
|---|---|
| `ModuleNotFoundError: customtkinter` | `pip install -r requirements.txt` no venv ativo |
| Setup pede Git | Instale Git e reabra o terminal, ou deixe o fallback PyPI rodar |
| FFmpeg ausente no Windows | O setup baixa o essentials; se falhar por permissão, rode o terminal elevado |
| `torch.cuda.is_available() = False` com RTX | Driver NVIDIA; o setup tenta `cu128` → … → CPU. Veja o log do setup |
| CUDA Out of Memory | Modelo menor, `int8`, ou deixe o app reduzir o lote sozinho |
| Diarização 403 / gated | Token da **mesma** conta que aceitou o Pyannote; permissão Read |
| Diarização cinza na GUI | Cole o token HF primeiro; locutores só habilitam com token |
| Transcrição “muda” no ShadowPlay | Ligue misturar faixas (duas primeiras = sistema + microfone) |
| `DLL load failed` / Controle de Aplicativo | O Windows (Smart App Control) bloqueou o PyAV. O FalaEdinho usa FFmpeg no lugar — feche o app e abra de novo. Se outra DLL (SciPy etc.) ainda for bloqueada: Configurações → Segurança do Windows → Controle de aplicativos e do navegador → Smart App Control → **Desativado** (exige reinício). |
| HTTP HEAD no log / timeout sem rede | Modelos precisam estar no cache; o app tenta offline primeiro |
| Python 3.14 e WhisperX recusa instalar | O setup usa `--ignore-requires-python`; o mais estável continua sendo 3.12 |

Log da janela (console inferior) e a faixa de status (decorrido / restante / última transcrição) mostram o estágio atual.
