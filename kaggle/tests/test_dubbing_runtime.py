import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dubbing_runtime import (
    ProviderConfig,
    RunContext,
    build_atempo_filter,
    fit_audio_to_slot,
    resolve_single_dataset_video,
    sha256_file,
)


class DubbingRuntimeTests(unittest.TestCase):
    def test_sha256_and_exact_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.mp4"
            path.write_bytes(b"abc123")
            digest = sha256_file(path)
            self.assertEqual(
                digest,
                "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090",
            )

    def test_dataset_requires_exactly_one_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_bytes(b"a")
            (root / "b.mp4").write_bytes(b"b")
            with self.assertRaises(RuntimeError):
                resolve_single_dataset_video(root)

    def test_run_context_persists_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"input-video")

            ctx = RunContext.create(
                source,
                work_root=root / "runs",
                config={"profile": "gpu-production"},
            )
            self.assertTrue((ctx.run_dir / "run_manifest.json").is_file())

            artifact = ctx.stage_dir("audio_extraction") / "audio.wav"
            artifact.write_bytes(b"audio")
            with ctx.stage("audio_extraction", message="Audio extraction"):
                ctx.record_outputs("audio_extraction", [artifact])

            self.assertTrue(ctx.is_complete("audio_extraction"))
            ctx.verify_input_unchanged()

            resumed = RunContext.resume(ctx.run_dir)
            self.assertEqual(resumed.manifest.run_id, ctx.manifest.run_id)
            self.assertEqual(
                resumed.manifest.input_sha256,
                ctx.manifest.input_sha256,
            )
            payload = json.loads(
                (resumed.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["stages"]["audio_extraction"]["status"],
                "passed",
            )

    def test_provider_config_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASR_PROVIDER": " IndicConformer ",
                "TRANSLATE_PROVIDER": "indictrans2",
                "TTS_PROVIDER": "local",
                "LIPSYNC_PROVIDER": "musetalk",
            },
            clear=False,
        ):
            cfg = ProviderConfig.from_env()

        self.assertEqual(cfg.asr, "indicconformer")
        self.assertEqual(cfg.translation, "indictrans2")
        self.assertEqual(cfg.lipsync, "musetalk")

    def test_fit_audio_to_slot(self) -> None:
        result = fit_audio_to_slot(5.0, 4.0)
        self.assertAlmostEqual(result["applied_atempo_factor"], 1.25)
        self.assertTrue(result["within_limits"])

        clamped = fit_audio_to_slot(8.0, 2.0)
        self.assertTrue(clamped["clamped"])
        self.assertAlmostEqual(clamped["applied_atempo_factor"], 1.6)
        self.assertFalse(clamped["within_limits"])

    def test_build_atempo_filter(self) -> None:
        self.assertEqual(build_atempo_filter(1.25), "atempo=1.25000000")
        self.assertEqual(
            build_atempo_filter(4.0),
            "atempo=2.00000000,atempo=2.00000000",
        )
        self.assertEqual(
            build_atempo_filter(0.25),
            "atempo=0.50000000,atempo=0.50000000",
        )


if __name__ == "__main__":
    unittest.main()
