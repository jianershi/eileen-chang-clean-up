# SPDX-License-Identifier: MIT
"""CPU regressions for sample alignment, resumability and actual audio export."""
import ast
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from eileen_cleanup.chunks import checkpoint_paths, join_chunks, plan_chunks
from eileen_cleanup.common import probe
from eileen_cleanup.export import export_m4a
from eileen_cleanup.profile import MODEL_REPOSITORY, SOURCE_NOTICE
from eileen_cleanup.restore import bind_run, enhance_chunks, finalize_audio, normalize_master, restore_audio
from eileen_cleanup.verify import verify_audio

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, total, rate=1000):
        audio = np.random.default_rng(3).normal(0, .01, total).astype("float32")
        source = self.root / "source.wav"
        sf.write(source, audio, rate, subtype="FLOAT")
        return audio, source

    def test_identity_join_including_short_final_core(self):
        for total in (1, 400, 20_000, 20_001, 40_500, 63_109):
            with self.subTest(samples=total):
                audio, source = self.prepare(total)
                chunks = plan_chunks(total, 1000)
                directory = self.root / str(total)
                rows = enhance_chunks(source, chunks, directory, 1000, lambda x: x)
                output = self.root / "joined.wav"
                self.assertEqual(join_chunks(rows, directory, output, total, 1000), total)
                joined, rate = sf.read(output, dtype="float32")
                self.assertEqual(rate, 1000)
                np.testing.assert_allclose(joined, audio, rtol=1e-5, atol=1e-7)

    def test_reuse_and_corrupt_checkpoint_recovery(self):
        _, source = self.prepare(40_100)
        chunks = plan_chunks(40_100, 1000)
        calls = []
        enhancer = lambda x: (calls.append(len(x)) or x)
        rows = enhance_chunks(source, chunks, self.root, 1000, enhancer)
        self.assertEqual(len(calls), len(chunks))
        enhance_chunks(source, chunks, self.root, 1000, None)
        path, _ = checkpoint_paths(self.root, 1)
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Missing or invalid"):
            enhance_chunks(source, chunks, self.root, 1000, None)
        enhance_chunks(source, chunks, self.root, 1000, enhancer)
        self.assertEqual(len(calls), len(chunks) + 1)
        self.assertEqual(len(rows), len(chunks))

    def test_cache_refuses_changed_input_or_configuration(self):
        config = {"source_sha256": "example-a", "core_seconds": 20, "model": MODEL_REPOSITORY}
        bind_run(self.root, config)
        bind_run(self.root, config)
        for changed in ({**config, "source_sha256": "example-b"}, {**config, "core_seconds": 10},
                        {**config, "model": "nvidia/RE-USE"}):
            with self.assertRaisesRegex(ValueError, "different source or configuration"):
                bind_run(self.root, changed)

    def test_finalize_refuses_previous_model_checkpoints(self):
        (self.root / "manifest.json").write_text(json.dumps({"model": "nvidia/RE-USE"}))
        with self.assertRaisesRegex(ValueError, "different model"):
            finalize_audio(self.root, self.root / "outputs")
        self.assertFalse((self.root / "outputs").exists())

    def test_orphan_cache_is_not_attached_to_another_input(self):
        (self.root / "prepared.wav").write_bytes(b"unfinished")
        with self.assertRaisesRegex(ValueError, "without a run identity"):
            bind_run(self.root, {})

    def test_nonfinite_prediction_is_not_checkpointed(self):
        _, source = self.prepare(1000)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            enhance_chunks(source, plan_chunks(1000, 1000), self.root, 1000,
                           lambda x: np.full_like(x, np.nan))
        self.assertFalse(checkpoint_paths(self.root, 0)[0].exists())

    def test_join_rejects_gaps(self):
        _, source = self.prepare(40_100)
        rows = enhance_chunks(source, plan_chunks(40_100, 1000), self.root, 1000, lambda x: x)
        rows[1]["start"] += 1
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            join_chunks(rows, self.root, self.root / "bad.wav", 40_100, 1000)

    def test_invalid_window_configuration(self):
        for args in ({"total": 0}, {"overlap_seconds": 0}, {"context_seconds": .5}, {"core_seconds": 2}):
            options = {"total": 1000, "rate": 1000, **args}
            with self.assertRaises(ValueError):
                plan_chunks(**options)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AudioExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rate = 44_100
        t = np.arange(self.rate * 8 + 137, dtype=np.float64) / self.rate
        self.audio = ((.04 + .02 * np.sin(2 * np.pi * .7 * t)) *
                      (np.sin(2 * np.pi * 437 * t) + .3 * np.sin(2 * np.pi * 813 * t))).astype("float32")
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, self.rate, subtype="FLOAT")

    def tearDown(self):
        self.temporary.cleanup()

    def test_normalization_preserves_samples_and_exports_aligned_aac(self):
        master = self.root / "master.flac"
        normalize_master(self.source, master)
        checks = verify_audio(master)
        self.assertEqual(checks["samples"], len(self.audio))
        self.assertLess(checks["peak"], .72)
        with contextlib.redirect_stdout(io.StringIO()):
            report = export_m4a(master, self.root / "copy.m4a")
        self.assertAlmostEqual(report["duration_seconds"], len(self.audio) / self.rate, delta=.05)
        self.assertEqual(probe(self.root / "copy.m4a")["format"]["tags"]["comment"], SOURCE_NOTICE)
        decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(self.root / "copy.m4a"),
                                            "-f", "f32le", "-"])
        compressed = np.frombuffer(decoded, dtype="<f4")
        lossless, _ = sf.read(master, dtype="float32")
        self.assertGreater(np.corrcoef(lossless[2000:-2000], compressed[2000:len(lossless)-2000])[0, 1], .99)
        with self.assertRaises(FileExistsError):
            export_m4a(master, self.root / "copy.m4a")

    def test_silent_master_is_rejected(self):
        sf.write(self.source, np.zeros(self.rate * 4), self.rate, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "silent"):
            normalize_master(self.source, self.root / "silent.flac")

    def test_complete_orchestration_and_cached_resume_without_gpu(self):
        calls = []

        class IdentityBackend:
            device_name, torch_version = "test CPU identity", "test"

            def __init__(self, directory, rate):
                pass

            def __call__(self, audio):
                calls.append(len(audio))
                return audio

        with patch("eileen_cleanup.model.download_model", return_value=(self.root, "test-weights")), \
                patch("eileen_cleanup.model.ReuseEnhancer", IdentityBackend), \
                contextlib.redirect_stdout(io.StringIO()):
            report = restore_audio(self.source, self.root / "outputs", self.root / "work",
                                   core_seconds=3, context_seconds=1, overlap_seconds=.5)
        self.assertEqual(report["verification"]["samples"], len(self.audio))
        self.assertTrue(calls)
        with patch("eileen_cleanup.model.download_model", side_effect=AssertionError("GPU should not run")), \
                contextlib.redirect_stdout(io.StringIO()):
            resumed = restore_audio(self.source, self.root / "outputs", self.root / "work",
                                    core_seconds=3, context_seconds=1, overlap_seconds=.5, overwrite=True)
        self.assertEqual(resumed["model_weights_sha256"], "test-weights")
        self.assertEqual(resumed["environment"]["gpu"], "test CPU identity")


class SourceTests(unittest.TestCase):
    def test_wheel_selection_rejects_abi_python_torch_and_cuda_mismatch(self):
        setup = script("setup_gpu")
        good = "mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        names = [good, good.replace("TRUE", "FALSE"), good.replace("cp313", "cp312"),
                 good.replace("cu13", "cu12"), good.replace("torch2.10", "torch2.9")]
        assets = [{"name": name, "browser_download_url": "https://github.com/state-spaces/mamba/" + name}
                  for name in names]
        selected = setup.matching_wheel(assets, "mamba_ssm", "cp313", "2.10", 13, True, "x86_64")
        self.assertEqual(selected["name"], good)
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            setup.matching_wheel(assets[1:], "mamba_ssm", "cp313", "2.10", 13, True, "x86_64")

    def test_notebook_is_clean_audio_only_and_valid_python(self):
        notebook = json.loads((ROOT / "REUSE_full_recording_GPU.ipynb").read_text())
        self.assertNotIn("widgets", notebook["metadata"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                source = "".join(cell["source"])
                ast.parse(source)
                self.assertNotIn("whisper", source.lower())
                self.assertNotIn("mp4", source.lower())
                self.assertNotIn("EXPECTED_SHA256", source)

    def test_source_archive_excludes_research_media_and_backups(self):
        packaging = script("package_source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "LICENSE", "source.pdf", "source.mp3", "transcript.srt",
                         "work/original-notebook/old.ipynb", "outputs/recording.flac",
                         "src/eileen_cleanup/main.py", "src/eileen_cleanup/weights.safetensors",
                         "docs/text.pdf", "docs/release-assets.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            archive = packaging.package(root, root / "source.zip")
            with ZipFile(archive) as zipped:
                names = zipped.namelist()
            self.assertEqual(len(names), 4)
            self.assertTrue(all(Path(name).suffix not in (".mp3", ".flac", ".pdf", ".srt", ".safetensors") for name in names))

    def test_unwrap_model_config_handles_flat_and_nested_laughter_configs(self):
        from eileen_cleanup.model import unwrap_model_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = {"model_cfg": {"d_state": 16}}
            (root / "config.json").write_text(json.dumps(flat))
            self.assertEqual(unwrap_model_config(root), flat)
            (root / "config.json").write_text(json.dumps({"cfg": flat}))
            self.assertEqual(unwrap_model_config(root), flat)

    def test_cli_parses_ensemble_and_backend_flags(self):
        from eileen_cleanup.cli import parser

        args = parser().parse_args(["ensemble", "input.wav", "--backend", "mlx", "--shift-seconds", "10"])
        self.assertEqual(args.command, "ensemble")
        self.assertEqual(args.backend, "mlx")
        self.assertEqual(args.shift_seconds, 10.0)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe")
    and importlib.util.find_spec("scipy") and importlib.util.find_spec("soxr"),
    "FFmpeg, scipy, and soxr required",
)
class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rate = 44_100
        t = np.arange(self.rate * 8 + 137, dtype=np.float64) / self.rate
        self.audio = ((.04 + .02 * np.sin(2 * np.pi * .7 * t)) *
                      (np.sin(2 * np.pi * 437 * t) + .3 * np.sin(2 * np.pi * 1213 * t))).astype("float32")
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, self.rate, subtype="FLOAT")

    def tearDown(self):
        self.temporary.cleanup()

    def test_shifted_pass_identity_preserves_samples_and_values(self):
        from eileen_cleanup.ensemble import run_resumable_pass

        joined, info = run_resumable_pass(
            self.source, self.root / "pass_shifted", self.rate,
            "test/repo", "0" * 40,
            core_seconds=3.0, context_seconds=1.0, overlap_seconds=0.5, shift_seconds=1.5,
            enhancer_factory=lambda repo, rev, sr: (lambda x: x),
        )
        self.assertFalse(info["reused"])
        out, rate = sf.read(joined, dtype="float32")
        self.assertEqual(rate, self.rate)
        self.assertEqual(len(out), len(self.audio))
        np.testing.assert_allclose(out, self.audio, rtol=1e-4, atol=1e-5)

    def test_gentle_wiener_and_linear_master_preserve_length(self):
        from eileen_cleanup.ensemble import gentle_wiener_blockwise, normalize_linear_master

        cleaned = gentle_wiener_blockwise(self.audio, sample_rate=self.rate, block_seconds=4.0, overlap_seconds=0.5)
        self.assertEqual(len(cleaned), len(self.audio))
        self.assertTrue(np.isfinite(cleaned).all())

        master = self.root / "linear.flac"
        norm = normalize_linear_master(self.source, master)
        checks = verify_audio(master)
        self.assertEqual(checks["samples"], len(self.audio))
        self.assertLess(checks["peak"], .72)
        self.assertTrue(np.isfinite(norm["linear_gain_db"]))

    def test_ensemble_end_to_end_resume_and_pass1_reuse(self):
        from eileen_cleanup.ensemble import restore_ensemble

        calls = []

        def factory(repo, rev, sr):
            class Identity:
                device_name = "test CPU identity"
                torch_version = "test"

                def __call__(self, audio):
                    calls.append((repo, sr, len(audio)))
                    return audio
            return Identity()

        with contextlib.redirect_stdout(io.StringIO()):
            report = restore_ensemble(
                self.source, self.root / "out", self.root / "work",
                core_seconds=3.0, context_seconds=1.0, overlap_seconds=0.5, shift_seconds=1.5,
                enhancer_factory=factory,
            )
        self.assertEqual(report["verification"]["samples"], len(self.audio))
        self.assertLess(report["verification"]["peak"], .72)
        self.assertTrue(calls)
        self.assertTrue((self.root / "out" / "REUSE_ensemble_full_recording.flac").exists())
        self.assertTrue((self.root / "out" / "REUSE_ensemble_full_recording.m4a").exists())

        # Resuming into the same work directory should run zero enhancement calls.
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = restore_ensemble(
                self.source, self.root / "out", self.root / "work",
                core_seconds=3.0, context_seconds=1.0, overlap_seconds=0.5, shift_seconds=1.5,
                overwrite=True,
                enhancer_factory=lambda *a: (_ for _ in ()).throw(AssertionError("Cached passes should not rerun")),
            )
        self.assertEqual(resumed["verification"]["samples"], len(self.audio))

        # Reusing pass1_joined into a fresh work directory should skip Pass 1 (44.1 kHz).
        pass23_calls = []

        def factory_no_44k(repo, rev, sr):
            self.assertEqual(sr, 16_000)
            return lambda x: (pass23_calls.append(sr) or x)

        with contextlib.redirect_stdout(io.StringIO()):
            reused_p1 = restore_ensemble(
                self.source, self.root / "out2", self.root / "work2",
                core_seconds=3.0, context_seconds=1.0, overlap_seconds=0.5, shift_seconds=1.5,
                pass1_joined=self.root / "work" / "pass1_44k_laughter" / "joined.wav",
                enhancer_factory=factory_no_44k,
            )
        self.assertEqual(reused_p1["verification"]["samples"], len(self.audio))
        self.assertTrue(pass23_calls)


if __name__ == "__main__":
    unittest.main()
