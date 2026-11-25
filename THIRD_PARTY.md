# What hearth installs, and its licences

hearth and its chat window are Apache-2.0. Their releases contain only their own code:
everything below is downloaded from its publisher, onto the user's computer, by
the installer or when it is chosen on the setup page. (A summary to find your way
by, not legal advice; each part's own licence text is what counts.)

| Part | Licence | When |
|---|---|---|
| [uv](https://github.com/astral-sh/uv), [Python](https://github.com/astral-sh/python-build-standalone) | Apache-2.0 / MIT; PSF | always |
| [Ollama](https://github.com/ollama/ollama) and the libraries in its archive (llama.cpp, NVIDIA's CUDA runtime, AMD's ROCm) | MIT; the archive's own notices | always |
| Python packages in `uv.lock` | MIT, BSD, Apache-2.0, MPL-2.0, PSF — and the next two rows | always |
| `phonemizer-fork`, and espeak-ng inside `espeakng-loader` (used by the natural voices) | **GPL-3.0** | always installed, used only for the voices |
| FFmpeg inside PyAV's wheel (used by speech recognition) | LGPL/GPL, per its build | always installed, used only for speech recognition |
| Chat models in `catalog.json` (Qwen) | Apache-2.0 | when chosen |
| nomic-embed-text | Apache-2.0 | with the first model |
| Kokoro-82M voices | Apache-2.0 | when chosen |
| Whisper small (CTranslate2) | MIT | when chosen |
| [Piper](https://github.com/rhasspy/piper) and its voices | MIT; **each voice has its own** (en_US-amy: see its model card) | only by `hearth voice` |

A model named by hand ("Another model") has its publisher's licence: some forbid
commercial use.
