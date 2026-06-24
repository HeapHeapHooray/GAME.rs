# GAME.rs

A high-performance, native Rust reimplementation of the inference pipeline for **[GAME (Generative Adaptive MIDI Extractor)](https://github.com/openvpi/GAME)**. Made with Gemini 🚀

This implementation replaces the original PyTorch-based inference script with a fast, self-contained CLI tool utilizing ONNX Runtime and the Symphonia audio decoding library.

---

## Key Features & Advantages

* **Instant Startup**: No Python interpreter initialization or heavy PyTorch library loading.
* **Robust Audio Stream Parsing**: Decodes audio packet-by-packet dynamically using the native Rust library `symphonia`. It completely avoids failures on streaming/incomplete FLAC/WAV files that lack a finalized `total_samples` header.
* **Lightweight Environment**: Eliminates the need for a 1GB+ Python environment containing PyTorch, libsndfile, and GPU runtime libraries.
* **Identical Transcription Accuracy**: Matches the Python engine's transcribed note pitches, onsets, offsets, and delta times exactly.

---

## Prerequisites

Before building or running the project, make sure you have:
1. The Rust toolchain (Cargo/rustc 1.70+) installed.
2. The exported ONNX model checkpoints of the model (e.g. `GAME-1.0-large-onnx`). If you haven't exported them yet, you can run the original Python project's `deploy.py` script to generate them.

---

## Compilation

Build the project in release mode for maximum performance:

```bash
cargo build --release
```

The compiled binary will be located at `target/release/game_rs`.

---

## CLI Usage

Run transcription on an audio file using the compiled binary:

```bash
./target/release/game_rs <input-audio-file> --model-dir <onnx-model-dir> [options]
```

### Example
```bash
./target/release/game_rs "/path/to/audio.flac" \
  --model-dir "/path/to/model_onnx" \
  --output "output.mid"
```

### CLI Parameters & Flags

* `<input>`: **(Required)** Path to the input audio file (supports FLAC, WAV, MP3, etc.).
* `-m, --model-dir <path>`: Path to the directory containing the ONNX models (`config.json`, `encoder.onnx`, `segmenter.onnx`, `bd2dur.onnx`, `estimator.onnx`). *Default: `checkpoints/GAME-1.0-large-onnx`*.
* `-o, --output <path>`: Destination path for the output MIDI file. *Default: `<input-name>.mid`*.
* `--tempo <bpm>`: Tempo for the exported MIDI. *Default: `120.0`*.
* `--seg-threshold <threshold>`: Boundary decoding threshold for the segmentation model. *Default: `0.2`*.
* `--seg-radius <radius>`: Local boundary search radius. *Default: `2`*.
* `--est-threshold <threshold>`: Presence detecting threshold for the pitch estimation model. *Default: `0.2`*.
* `-l, --language <lang>`: Optional language parameter guiding the transcription (`en`, `ja`, `zh`, `yue`).
* `--nsteps <steps>`: Number of D3PM boundary-refining loop steps. *Default: `8`*.
* `--t0 <t0>`: Starting T value for the D3PM loop. *Default: `0.0`*.

---

## Technical Pipeline

The inference pipeline corresponds directly to the original model structure:
1. **Audio Decoding**: The audio is loaded via `symphonia` and resampled linearly (using linear interpolation) to the model sample rate (usually `44100` Hz) and mixed to mono.
2. **Encoder**: The resampled audio waveform is run through `encoder.onnx` to generate segmenter features (`x_seg`), estimator features (`x_est`), and the temporal mask (`mask_t`).
3. **D3PM Segmentation Loop**: A refining loop runs `segmenter.onnx` for `nsteps` starting from `t0` to generate a boundary map.
4. **Boundary to Duration Conversion**: The predicted boundaries are run through `bd2dur.onnx` to compute note durations (`durations`) and note masks (`mask_n`).
5. **Pitch Estimation**: The estimator model (`estimator.onnx`) uses `x_est` and boundaries to predict presence probabilities and pitch scores.
6. **MIDI Exporter**: voicings are filtered and overlapping notes are cleaned up before outputting a standard single-track metrical MIDI file using the `midly` crate.
