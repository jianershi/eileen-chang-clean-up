# Eileen Chang audio cleanup

Clean the archival recording of **Eileen Chang, “Chinese Translation: A Vehicle of Cultural Influence”** using [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) and FFmpeg.

Outputs preserve the complete input recording. The full archival input is approximately 105 minutes, including the subsequent seminar:

- `outputs/NVIDIA_NemotronLabs_REUSE_full_recording.flac` — lossless, 44.1 kHz mono master.
- `outputs/NVIDIA_NemotronLabs_REUSE_full_recording.m4a` — smaller AAC listening copy at 192 kb/s.

Audio files are ignored by Git and can be shared separately as release attachments. Checksums and validation results for both restorations are in [docs/release-assets.json](docs/release-assets.json).

## Google Colab

Colab runs Jupyter `.ipynb` notebooks with an optional GPU.

1. [Open in Colab](https://colab.research.google.com/github/jianershi/eileen-chang-clean-up/blob/main/REUSE_full_recording_GPU.ipynb) and select a **T4 GPU** runtime.
2. Choose **Runtime → Run all**. The notebook downloads the code from this repository.
3. Upload your complete original audio when prompted, then download the new FLAC and M4A results.

Colab's free GPU availability varies. Its default `/content/` storage is temporary; preserve the entire work directory to resume after a VM reset.

## Command line

Requires Python 3.10+ and FFmpeg/ffprobe. Enhancement needs a Linux x86_64 NVIDIA CUDA GPU; CPU export and verification also work on macOS. Python 3.13 matches the original T4 run.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/setup_gpu.py --install-torch
python -m eileen_cleanup restore /path/to/recording.ext
python -m eileen_cleanup verify outputs/NVIDIA_NemotronLabs_REUSE_full_recording.flac
python -m eileen_cleanup verify outputs/NVIDIA_NemotronLabs_REUSE_full_recording.m4a
```

Use any complete source encoding supported by FFmpeg. Rerun the same command to resume; keep the source and configuration unchanged. `--overwrite` replaces existing outputs deliberately. If GPU memory runs out, use `--core-seconds 10` and a new `--work-dir`.

After all GPU chunks finish, export on a CPU with `finalize`. To encode only an M4A from an existing restored FLAC, use `m4a`:

```sh
python -m eileen_cleanup finalize --work-dir work/nemotron-restoration
python -m eileen_cleanup m4a outputs/NVIDIA_NemotronLabs_REUSE_full_recording.flac
```

## Processing

The workflow uses a 60 Hz high-pass filter, float32 NemotronLabs RE-USE inference with 20-second cores and 2-second context, one-second crossfade margins, and measured loudness normalization targeting -20 LUFS. The model revision is pinned in `src/eileen_cleanup/profile.py`; model files download separately. Checkpoints support validated resumption.

The new model is pinned to revision `31f82f2efad1095034492f236e8dac9239ba713b`. The full 105-minute Colab run completed on a Tesla T4 with Torch 2.10.0+cu130; both exports were independently verified after download. Its separate cache and output names preserve the earlier restoration. Enhancement may introduce artifacts, so keep the original for comparison.

## Transcription comparison

We compared the raw source and four restorations against the published main speech cited below: original recording **02:31.27–39:35.07**, with **4,261 normalized reference words**. This evaluation covers the main speech only, excluding the introduction, Q&A and subsequent seminar.

**Word Error Rate (WER)** is the number of substitutions, deletions and insertions divided by the reference word count. Lower is better.

| Audio | Model version (revision) | Word errors | Word Error Rate (WER) | Substitutions | Deletions | Insertions |
|---|---|---:|---:|---:|---:|---:|
| Raw source (no enhancement) | — | 856 | 20.09% | 517 | 205 | 134 |
| Original NVIDIA RE-USE | [`0ee3101`](https://huggingface.co/nvidia/RE-USE/tree/0ee310117ef0ae1e652654c821cf140bdcdc8415) | 814 | 19.10% | 542 | 151 | 121 |
| **NVIDIA NemotronLabs RE-USE** | [`31f82f2`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b) | **788** | **18.49%** | 501 | 153 | 134 |
| UniPASE | [`f0b4d4c`](https://huggingface.co/Xiaobin-Rong/unipase/tree/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0) | 824 | 19.34% | 535 | 160 | 129 |
| Cisco StuPASE | [`539963f`](https://huggingface.co/cisco-ai/stupase/tree/539963f425ae201ac8c379b6480e3b182a4840ad) | 893 | 20.96% | 616 | 151 | 126 |

Versions identify the exact model repository revisions used, rather than release numbers. UniPASE used `enable_plc=True`; StuPASE used 8 steps, CFG 0.5 and sway -1.

All five used Whisper large-v3 with the same 82 time windows, CPU int8, beam size 5, temperature 0 and English recognition, with no VAD, text prompts or previous-text conditioning. The reference was used only for scoring.

The raw source was decoded and resampled only; the restored outputs include their existing preparation and loudness processing. These scores compare the resulting audio files, rather than isolated model architectures. They measure transcription errors against an edited publication, which may differ from the delivered speech; they are not hallucination rates or human listening scores.

## License and credits

Project code: [MIT](LICENSE). NVIDIA's model and implementation use [OpenMDW 1.1](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/LICENSE), which permits commercial use. The archival audio retains its separate terms below.

**Audio copyright: Schlesinger Library of Harvard. Noncommercial reuse with attribution is permitted.** Include this README when sharing the enhanced audio. Published text, PDFs and subtitles are not included.

[Original audio](https://mps.lib.harvard.edu/sds/audio/503996508) · [Catalog record](https://hollisarchives.lib.harvard.edu/catalog/sch01475_sch01475c00589) · [Library citation guide](https://guides.library.harvard.edu/schlesinger/citation_guide)

> Radcliffe College Archives sound recordings collection, 1951–2008. Bunting Institute Seminars. Reyher; Bas-Comain. undatedd. RG XX, Series 8–9, RA.T-106/nd., Audiotape 01. Schlesinger Library, Radcliffe Institute, Harvard University, Cambridge, Mass. https://hollisarchives.lib.harvard.edu/catalog/sch01475_sch01475c00589. Accessed October 23, 2026.

Thanks to Schlesinger Library and its archivists for preserving and providing access to the recording, and to journalist **Chai Jing / 柴静** for reporting on its rediscovery: [substack](https://chaijing.substack.com/p/767), [YouTube note](https://www.youtube.com/post/UgkxVZFF41EiWCfXv78iSDDW0gkHSGkU02ue), [channel](https://www.youtube.com/@chaijing2023).

Enhancement research: **Szu-Wei Fu et al.** “Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement” (2026). [Paper](https://arxiv.org/abs/2603.02641) · [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).

Related scholarship: **EILEEN CHANG and CHRISTOPHER LEE.** “Chinese Translation: A Vehicle of Cultural Influence.” *PMLA* 130, no. 2 (2015): 488–98. [JSTOR](https://www.jstor.org/stable/44015733). Referenced for research; its text is not distributed here.

## Development

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m build
python scripts/package_source.py
```
