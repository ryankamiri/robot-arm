"""Contract, gradient, density, and checkpoint checks for the actor."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution

from models.architecture import GaussianParameters
from models.dimensions import A, C_RGB, J
from models.policy import DapPolicy
from models.distributions import sample_tanh_gaussian


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.B, self.L = 2, 2
        self.H_in, self.W_in = 24, 32
        self.model = DapPolicy(self.L)
        self.images = torch.randn(self.B, self.L, C_RGB, self.H_in, self.W_in)
        self.positions = torch.rand(self.B, self.L, J) * 2 - 1

    def test_shapes_and_history_configuration(self):
        parameters = self.model(self.images, self.positions)
        self.assertEqual(parameters.mean.shape, (self.B, A))
        self.assertEqual(parameters.log_std.shape, (self.B, A))
        sample = self.model.sample(self.images, self.positions)
        self.assertEqual(sample.actions.shape, (self.B, A))
        self.assertEqual(sample.log_prob.shape, (self.B,))
        self.assertTrue(torch.isfinite(sample.log_prob).all())
        self.assertTrue((sample.actions.abs() <= 1).all())

        # A different total history and odd image dimensions need no new code.
        B, L, H_in, W_in = 1, 3, 17, 23
        other = DapPolicy(history_length=L)
        output = other(
            torch.randn(B, L, C_RGB, H_in, W_in), torch.zeros(B, L, J)
        )
        self.assertEqual(output.mean.shape, (B, A))

    def test_sampling_is_stochastic_and_deterministic_action_is_repeatable(self):
        first = self.model.sample(self.images, self.positions).actions
        second = self.model.sample(self.images, self.positions).actions
        self.assertFalse(torch.equal(first, second))
        deterministic = self.model.deterministic_action(self.images, self.positions)
        torch.testing.assert_close(
            deterministic, self.model.deterministic_action(self.images, self.positions)
        )
        torch.testing.assert_close(
            deterministic, self.model(self.images, self.positions).mean.tanh()
        )

    def test_reparameterized_gradients_reach_both_branches_and_heads(self):
        sample = self.model.sample(self.images, self.positions)
        # A surrogate actor loss checks differentiability, not a selected RL loss.
        loss = sample.log_prob.mean() + sample.actions.square().mean()
        loss.backward()
        parameters = dict(self.model.named_parameters())
        for name in (
            "network.image_encoder.stages.0.conv1.weight",
            "network.position_encoder.0.weight",
            "network.fusion.0.weight",
            "network.mean_head.weight",
            "network.log_std_head.weight",
        ):
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_sample_log_probability_matches_pytorch_transformed_distribution(self):
        B = 5
        parameters = GaussianParameters(
            mean=torch.zeros(B, A, dtype=torch.float64),
            log_std=torch.full((B, A), -1.0, dtype=torch.float64),
        )
        sample = sample_tanh_gaussian(parameters)
        reference = Independent(
            TransformedDistribution(
                Normal(parameters.mean, parameters.log_std.exp()),
                [TanhTransform()],
            ),
            1,
        )
        torch.testing.assert_close(sample.log_prob, reference.log_prob(sample.actions))

    def test_log_probability_and_gradients_stay_finite_at_tanh_saturation(self):
        B = 1
        mean = torch.tensor([[80.0, -80.0] * (A // 2)], requires_grad=True)
        log_std = torch.zeros(B, A, requires_grad=True)
        sample = sample_tanh_gaussian(GaussianParameters(mean, log_std))
        self.assertTrue((sample.actions.abs() <= 1).all())
        self.assertTrue(torch.isfinite(sample.log_prob).all())
        sample.log_prob.sum().backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(log_std.grad).all())

    def test_scale_bounds_do_not_squash_latent_mean(self):
        with torch.no_grad():
            self.model.network.log_std_head.weight.zero_()
            self.model.network.log_std_head.bias.copy_(
                torch.tensor([-20.0, -10.0, -5.0, 0.0, 2.0, 10.0])
            )
            self.model.network.mean_head.weight.zero_()
            self.model.network.mean_head.bias.fill_(3.0)
        output = self.model(self.images, self.positions)
        torch.testing.assert_close(
            output.log_std,
            torch.tensor([[-5.0, -5.0, -5.0, 0.0, 2.0, 2.0]]).expand(self.B, A),
        )
        self.assertTrue((output.mean == 3).all())
        self.assertTrue(
            (self.model.deterministic_action(self.images, self.positions).abs() <= 1).all()
        )

    def test_local_hub_checkpoint_preserves_weights_config_and_predictions(self):
        model = DapPolicy(self.L, -4.0, 1.0).eval()
        with torch.inference_mode():
            expected = model(self.images, self.positions)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(config["history_length"], self.L)
            self.assertEqual(config["log_std_min"], -4.0)
            self.assertEqual(config["log_std_max"], 1.0)
            self.assertTrue((Path(directory) / "model.safetensors").is_file())
            restored = DapPolicy.from_pretrained(directory, strict=True)
        with torch.inference_mode():
            actual = restored(self.images, self.positions)
        torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
        torch.testing.assert_close(actual.log_std, expected.log_std, rtol=0, atol=0)
        self.assertFalse(restored.training)

    def test_explicit_history_and_input_contract(self):
        with self.assertRaises(TypeError):
            DapPolicy()
        with self.assertRaises(ValueError):
            DapPolicy(history_length=0)
        with self.assertRaises(ValueError):
            DapPolicy(history_length=2, log_std_min=3, log_std_max=2)
        with self.assertRaises(ValueError):
            self.model(self.images[:, :1], self.positions[:, :1])
        with self.assertRaises(ValueError):
            self.model(self.images, self.positions[:, :, : J - 1])
        with self.assertRaises(TypeError):
            self.model(self.images.to(torch.uint8), self.positions)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_apple_gpu_forward_sampling_and_backward(self):
        model = self.model.to("mps")
        sample = model.sample(self.images.to("mps"), self.positions.to("mps"))
        self.assertEqual(sample.actions.device.type, "mps")
        self.assertTrue(torch.isfinite(sample.log_prob).all().item())
        self.assertTrue((sample.actions.abs() <= 1).all().item())
        (sample.log_prob.mean() + sample.actions.square().mean()).backward()
        self.assertTrue(torch.isfinite(model.network.mean_head.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
