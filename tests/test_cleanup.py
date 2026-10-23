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
from eileen_cleanup.profile import SOURCE_NOTICE
from eileen_cleanup.restore import bind_run, enhance_chunks, normalize_master, restore_audio
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
        config = {"source_sha256": "example-a", "core_seconds": 20}
        bind_run(self.root, config)
        bind_run(self.root, config)
        for changed in ({**config, "source_sha256": "example-b"}, {**config, "core_seconds": 10}):
            with self.assertRaisesRegex(ValueError, "different source or configuration"):
                bind_run(self.root, changed)

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


if __name__ == "__main__":
    unittest.main()
