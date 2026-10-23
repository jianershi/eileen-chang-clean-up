# Eileen Chang audio cleanup

Clean the archival recording of **Eileen Chang, “Chinese Translation: A Vehicle of Cultural Influence”** using [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE) and FFmpeg.

Outputs preserve the complete input recording. The current full restoration is approximately 105 minutes, including the subsequent seminar:

- `outputs/NVIDIA_REUSE_full_recording.flac` — lossless, 44.1 kHz mono master.
- `outputs/NVIDIA_REUSE_full_recording.m4a` — smaller AAC listening copy at 192 kb/s.

Audio files are ignored by Git and can be shared separately as release attachments. Output checksums are in [docs/release-assets.json](docs/release-assets.json).

## Google Colab

Colab runs Jupyter `.ipynb` notebooks with an optional GPU.

1. Upload [REUSE_full_recording_GPU.ipynb](REUSE_full_recording_GPU.ipynb) to Colab and select a **T4 GPU** runtime.
2. Run `python scripts/package_source.py` locally and upload the resulting source ZIP when prompted. Alternatively, set the notebook's `REPOSITORY_URL` after publishing this repository.
3. Run the cells in order, upload your complete original audio, and download the FLAC and M4A results.

Colab's free GPU availability varies. Its default `/content/` storage is temporary; preserve the entire work directory to resume after a VM reset.

## Command line

Requires Python 3.10+ and FFmpeg/ffprobe. Enhancement needs a Linux x86_64 NVIDIA CUDA GPU; CPU export and verification also work on macOS. Python 3.13 matches the original T4 run.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/setup_gpu.py --install-torch
python -m eileen_cleanup restore /path/to/recording.ext
python -m eileen_cleanup verify outputs/NVIDIA_REUSE_full_recording.flac
python -m eileen_cleanup verify outputs/NVIDIA_REUSE_full_recording.m4a
```

Use any complete source encoding supported by FFmpeg. Rerun the same command to resume; keep the source and configuration unchanged. `--overwrite` replaces existing outputs deliberately. If GPU memory runs out, use `--core-seconds 10` and a new `--work-dir`.

After all GPU chunks finish, export on a CPU with `finalize`. To encode only an M4A from an existing restored FLAC, use `m4a`:

```sh
python -m eileen_cleanup finalize --work-dir work/restoration
python -m eileen_cleanup m4a outputs/NVIDIA_REUSE_full_recording.flac
```

## Processing

The workflow uses a 60 Hz high-pass filter, float32 RE-USE inference with 20-second cores and 2-second context, one-second crossfade margins, and measured loudness normalization targeting -20 LUFS. The model revision is pinned in `src/eileen_cleanup/profile.py`; model files download separately. Checkpoints support validated resumption.

The original T4 run completed successfully. The refactor uses float32 checkpoints instead of intermediate PCM16 and has CPU tests; its GPU entry point has not been rerun. Enhancement may introduce artifacts, so keep the original for comparison.

## License and credits

Project code: [MIT](LICENSE). NVIDIA's model and implementation retain the [NVIDIA One-Way Noncommercial License](https://github.com/NVlabs/HMAR/blob/main/LICENSE), restricted to noncommercial research and education.

**Audio copyright: Schlesinger Library of Harvard. Noncommercial reuse with attribution is permitted.** Include this README when sharing the enhanced audio. Published text, PDFs and subtitles are not included.

[Original audio](https://mps.lib.harvard.edu/sds/audio/503996508) · [Catalog record](https://hollisarchives.lib.harvard.edu/catalog/sch01475_sch01475c00589) · [Library citation guide](https://guides.library.harvard.edu/schlesinger/citation_guide)

> Radcliffe College Archives sound recordings collection, 1951–2008. Bunting Institute Seminars. Reyher; Bas-Comain. undatedd. RG XX, Series 8–9, RA.T-106/nd., Audiotape 01. Schlesinger Library, Radcliffe Institute, Harvard University, Cambridge, Mass. https://hollisarchives.lib.harvard.edu/catalog/sch01475_sch01475c00589. Accessed October 23, 2026.

Thanks to Schlesinger Library and its archivists for preserving and providing access to the recording, and to journalist **Chai Jing / 柴静** for reporting on its rediscovery: [substack](https://chaijing.substack.com/p/767), [YouTube note](https://www.youtube.com/post/UgkxVZFF41EiWCfXv78iSDDW0gkHSGkU02ue), [channel](https://www.youtube.com/@chaijing2023).

Enhancement research: **Szu-Wei Fu et al.** “Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement” (2026). [Paper](https://arxiv.org/abs/2603.02641) · [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE).

Related scholarship: **EILEEN CHANG and CHRISTOPHER LEE.** “Chinese Translation: A Vehicle of Cultural Influence.” *PMLA* 130, no. 2 (2015): 488–98. [JSTOR](https://www.jstor.org/stable/44015733). Referenced for research; its text is not distributed here.

## Development

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m build
python scripts/package_source.py
```
