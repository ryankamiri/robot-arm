"""Shared config/checkpoint continuation and local W&B integration checks."""

from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from training.shared.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from training.shared.config import load_config, save_config
from training.shared.metrics import log_metrics, start_run


class SharedTrainingTests(unittest.TestCase):
    def test_yaml_round_trip_preserves_nested_resolved_config(self):
        config = {
            "optimizer": {"learning_rate": 0.0003, "name": "AdamW"},
            "history": [0, 1, 2],
            "eval_at_end": True,
            "scheduler": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_config(config, Path(directory) / "run/config.yaml")
            self.assertEqual(load_config(path), config)

    def test_yaml_rejects_nonmapping_and_nonportable_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for contents in ("", "[1, 2]", "1: value", "value: .nan", "date: 2026-10-15"):
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_checkpoint_restores_next_update_and_caller_state(self):
        # Tiny linear fixtures exercise serialization, not a BC/SAC architecture.
        torch.manual_seed(12)
        D_in, D_out = 3, 2
        models = {name: nn.Linear(D_in, D_out) for name in ("actor", "critic")}
        optimizers = {name: AdamW(model.parameters(), lr=0.01) for name, model in models.items()}

        def update(models, optimizers) -> None:
            B = 2
            inputs: Float[Tensor, "B D_in"] = torch.randn(B, D_in)
            for name, model in models.items():
                optimizers[name].zero_grad()
                loss: Float[Tensor, ""] = model(inputs).square().mean()
                loss.backward()
                optimizers[name].step()

        update(models, optimizers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(
                path, models, optimizers,
                {"history_length": 3},
                {"optimizer_step": 1, "epoch": 0},
                {"sampler_order": [2, 0, 1], "sampler_offset": 2, "wandb_run_id": "test-id"},
            )
            update(models, optimizers)
            expected = {
                name: {key: value.clone() for key, value in model.state_dict().items()}
                for name, model in models.items()
            }
            restored = {name: nn.Linear(D_in, D_out) for name in models}
            # An intentionally different LR verifies optimizer settings restore.
            restored_optimizers = {
                name: AdamW(model.parameters(), lr=0.5) for name, model in restored.items()
            }
            state = load_checkpoint(path, restored, restored_optimizers)
            self.assertEqual(state.progress, {"optimizer_step": 1, "epoch": 0})
            self.assertEqual(state.config, {"history_length": 3})
            self.assertEqual(state.extra_state["sampler_offset"], 2)
            self.assertEqual(state.extra_state["wandb_run_id"], "test-id")
            update(restored, restored_optimizers)
            for name, model in restored.items():
                self.assertEqual(restored_optimizers[name].param_groups[0]["lr"], 0.01)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_checkpoint(path, {"wrong_name": restored["actor"]}, restored_optimizers)

    def test_rng_round_trip_includes_python_numpy_torch_and_available_gpus(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        state = capture_rng_state()
        python_expected = random.random()
        numpy_expected = np.random.random()
        V = 4
        cpu_expected: Float[Tensor, "V"] = torch.rand(V)
        mps_expected: Float[Tensor, "V"] | None = (
            torch.rand(V, device="mps").cpu() if torch.backends.mps.is_available() else None
        )
        cuda_expected: Float[Tensor, "V"] | None = (
            torch.rand(V, device="cuda").cpu() if torch.cuda.is_available() else None
        )
        restore_rng_state(state)
        self.assertEqual(random.random(), python_expected)
        self.assertEqual(np.random.random(), numpy_expected)
        torch.testing.assert_close(torch.rand(V), cpu_expected, rtol=0, atol=0)
        if mps_expected is not None:
            torch.testing.assert_close(torch.rand(V, device="mps").cpu(), mps_expected, rtol=0, atol=0)
        if cuda_expected is not None:
            torch.testing.assert_close(torch.rand(V, device="cuda").cpu(), cuda_expected, rtol=0, atol=0)

    def test_failed_checkpoint_write_preserves_previous_file(self):
        D_in, D_out = 3, 2
        model = nn.Linear(D_in, D_out)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, {"actor": model}, {}, {}, {"optimizer_step": 1})
            original = path.read_bytes()
            with patch("training.shared.checkpoints.torch.save", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"actor": model}, {}, {}, {"optimizer_step": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_metrics_keep_optimizer_counter_separate_from_wandb_records(self):
        run = Mock()
        log_metrics(run, {"train/nll": 0.5}, 10)
        log_metrics(run, {"val/nll": 0.6}, 10)
        self.assertEqual(run.log.call_args_list[0].args[0], {"optimizer_step": 10, "train/nll": 0.5})
        self.assertEqual(run.log.call_args_list[1].args[0], {"optimizer_step": 10, "val/nll": 0.6})
        with self.assertRaises(ValueError):
            log_metrics(run, {"optimizer_step": 99}, 10)

    def test_real_wandb_run_logs_locally_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            run = start_run("robot-arm-shared-tests", {"history_length": 3}, "offline", directory)
            try:
                self.assertTrue(run.settings._offline)
                self.assertEqual(run.config["history_length"], 3)
                log_metrics(run, {"train/nll": 0.5}, 1)
                log_metrics(run, {"val/nll": 0.6}, 1)
                self.assertEqual(run.step, 2)
            finally:
                run.finish()


if __name__ == "__main__":
    unittest.main()
