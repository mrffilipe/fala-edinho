# Especificação Técnica: FalaEdinho (interface desktop WhisperX e setup automático)

## 1. Visão Geral do Projeto

Desenvolver o **FalaEdinho**, uma aplicação desktop em Python que automatize o setup completo do **WhisperX** (identificação de hardware, instalação do PyTorch correto com suporte CUDA/CPU e dependências do WhisperX/FFmpeg) e forneça uma interface gráfica amigável para transcrição, diarização de locutores e geração de legendas.

---

## 2. Requisitos de Ambiente e Stack Tecnológica

* **Linguagem:** Python 3.10+
* **Interface Gráfica (GUI):** `CustomTkinter` (para design moderno e suporte nativo a Drag & Drop via `tkinterdnd2`) ou `PyQt6` / `PySide6`.
* **Gerenciamento de Processos:** `subprocess`, `sys`, `os`.
* **Detecção de Hardware:** `torch` (pós-instalação), `pynvml` ou parsing via `nvidia-smi` / `wmic`.
* **Manipulação de Áudio/Legendas:** `WhisperX` (pipeline oficial), `FFmpeg`.

---

## 3. Módulos do Sistema

### Módulo A: Setup Automation & Hardware Detection (First-Run Engine)

1. **Verificação de Dependências do Sistema:**
* Checar se o **FFmpeg** está instalado e acessível no `PATH` do sistema. Se não estiver, fazer o download do binário do FFmpeg e adicionar ao contexto da aplicação.


2. **Identificação de Hardware:**
* Detectar se existe GPU NVIDIA compatível no sistema.
* Identificar a versão dos drivers CUDA suportados.


3. **Gerenciador de Instalação (Bootstrap):**
* Se a GPU NVIDIA for detectada: instalar a versão do PyTorch compilada com suporte a CUDA (ex: `cu118` ou `cu121`).
* Se nenhuma GPU for encontrada: instalar PyTorch em modo CPU.
* Executar a instalação automática do pacote oficial WhisperX via repositório Git:
`pip install git+[https://github.com/m-bain/whisperx.git](https://github.com/m-bain/whisperx.git)`


4. **Download de Modelos e Token HuggingFace:**
* Solicitar e armazenar o Token de Acesso do HuggingFace (necessário para os modelos de diarização da `pyannote.audio`).



---

### Módulo B: Interface Gráfica (GUI) & UX

#### 1. Painel de Entrada de Arquivos

* **Zona de Drop / Seleção:** Área visual para arrastar e soltar (Drag & Drop) arquivos de vídeo/áudio (`.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a`) ou botão para abrir o Explorador de Arquivos do sistema (`filedialog`).
* **Lista de Arquivos:** Exibição da fila de processamento (suporte a múltiplos arquivos em lote).

#### 2. Painel de Parâmetros do WhisperX

Interface para ajuste dos argumentos do WhisperX:

* **Modelo Whisper:** ComboBox (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`).
* **Idioma:** Dropdown de seleção de idioma (ou opção `Auto-detect`).
* **Batch Size:** Slider ou Number Input (Padrão GPU: `16`, Padrão CPU: `4`).
* **Compute Type:** Radio Buttons (`float16` para GPUs modernas, `int8` para CPUs ou GPUs mais antigas, `float32`).
* **Diarização (Identificação de Locutores):**
* Checkbox: *Ativar Diarização (Pyannote)*.
* Inputs condicionais (se ativado): `min_speakers` e `max_speakers` (opcionais).
* Input de Texto: `HuggingFace Access Token` (com persistência salva em arquivo de configuração local).


* **Formato de Saída:** Checkboxes para exportação (`.srt`, `.vtt`, `.txt`, `.json`, `.tsv`).

#### 3. Painel de Status e Logs

* **Barra de Progresso:** Feedback visual do estágio atual (*Carregando modelo*, *Transcrevendo*, *Alinhando timestamps*, *Diarizando*, *Exportando*).
* **Terminal/Log Window:** Área de texto estilizada mostrando a saída padrão do console em tempo real.

---

## 4. Arquitetura da Aplicação e Multithreading

* **Separação de Threads:** A interface gráfica NUNCA deve congelar. O pipeline do WhisperX deve rodar em uma thread separada (`threading.Thread` ou `QThread`).
* **Comunicação de Eventos:** Utilizar filas (`queue.Queue` ou sinais do PyQt) para enviar logs, percentual de progresso e erros da thread do WhisperX para a GUI.

---

## 5. Fluxo de Execução (User Journey)

1. **Inicialização:** A aplicação inicia e verifica se o ambiente WhisperX está pronto. Se não, exibe uma tela de *Setup Automático* e realiza as instalações necessárias.
2. **Configuração:** O usuário insere o arquivo (arrastando ou selecionando), define o modelo e ativa a diarização (colocando o token HF se necessário).
3. **Execução:** O usuário clica em **"Iniciar Transcrição"**.
4. **Resultado:** Ao finalizar, o sistema abre automaticamente a pasta onde os arquivos exportados (`.srt`, `.txt`, etc.) foram salvos.

---

## 6. Critérios de Aceite (Entregáveis do Agente)

1. Script principal `main.py` executável sem erros.
2. Módulo de instalação `installer.py` que trata erros de permissão e dependências ausentes (como Git ou FFmpeg).
3. Arquivo `requirements.txt` básico contendo as dependências da GUI e ferramentas de suporte.
4. Código limpo, comentado e com tratamento explícito de exceções para falhas de memória VRAM (Out of Memory) ajustando automaticamente o `batch_size` se necessário.