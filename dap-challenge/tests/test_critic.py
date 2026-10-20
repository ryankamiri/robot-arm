"""Verify BC transfer, critic independence, action gradients and checkpoint compatibility."""

from io import BytesIO
import unittest

import torch

from models.architecture import ObservationEncoder
from models.critic import DapCritic, TwinCritics
from models.dimensions import A, C_RGB, I_INPUTS
from models.policy import DapPolicy


class CriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.B, self.L, self.H_in, self.W_in = 2, 3, 17, 23
        self.actor = DapPolicy(self.L).eval()
        self.images = torch.rand(self.B, self.L, I_INPUTS, C_RGB, self.H_in, self.W_in)
        self.positions = torch.rand(self.B, self.L, A) * 2 - 1
        self.actions = torch.rand(self.B, A) * 2 - 1
        self.critics = TwinCritics.from_policy(self.actor)

    def test_transfer_copies_bc_encoders_but_keeps_all_three_models_independent(self):
        for encoder_name in ('image_encoder', 'position_encoder'):
            source = getattr(self.actor.network, encoder_name)
            for critic in (self.critics.q1, self.critics.q2):
                copied = getattr(critic, encoder_name)
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(copied.state_dict()[key], value, rtol=0, atol=0)

        models = (self.actor, self.critics.q1, self.critics.q2)
        storage = [{parameter.data_ptr() for parameter in model.parameters()} for model in models]
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertTrue(storage[first].isdisjoint(storage[second]))
        for layer in ('fusion.0.weight', 'fusion.2.weight', 'q_head.weight'):
            first = self.critics.q1.state_dict()[layer]
            second = self.critics.q2.state_dict()[layer]
            self.assertFalse(torch.equal(first, second))
        for critic in (self.critics.q1, self.critics.q2):
            self.assertFalse(any('mean_head' in name or 'log_std_head' in name for name in critic.state_dict()))
            actual = ObservationEncoder.forward(critic, self.images, self.positions)
            expected = ObservationEncoder.forward(self.actor.network, self.images, self.positions)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_twin_outputs_are_action_conditioned_vectors_including_batch_one(self):
        values = self.critics(self.images, self.positions, self.actions)
        self.assertEqual(values.q1.shape, (self.B,))
        self.assertEqual(values.q2.shape, (self.B,))
        self.assertFalse(torch.equal(values.q1, values.q2))
        changed = self.critics(self.images, self.positions, -self.actions)
        self.assertFalse(torch.equal(values.q1, changed.q1))
        self.assertFalse(torch.equal(values.q2, changed.q2))
        single = self.critics(self.images[:1], self.positions[:1], self.actions[:1])
        self.assertEqual(single.q1.shape, (1,))
        self.assertEqual(single.q2.shape, (1,))

    def test_scalar_output_is_unbounded_and_not_squashed(self):
        critic = self.critics.q1
        with torch.no_grad():
            critic.q_head.weight.zero_()
            critic.q_head.bias.fill_(-3.)
        torch.testing.assert_close(critic(self.images, self.positions, self.actions), torch.full((self.B,), -3.))

    def test_gradients_reach_actions_positions_both_views_and_critic_encoders(self):
        images = self.images.clone().requires_grad_()
        positions = self.positions.clone().requires_grad_()
        actions = self.actions.clone().requires_grad_()
        values = self.critics(images, positions, actions)
        (values.q1.sum() + values.q2.sum()).backward()
        for gradient in (images.grad, positions.grad, actions.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        for camera in range(I_INPUTS):
            self.assertGreater(images.grad[:, :, camera].abs().sum().item(), 0)
        for critic in (self.critics.q1, self.critics.q2):
            parameters = dict(critic.named_parameters())
            for name in ('image_encoder.stages.0.conv1.weight', 'position_encoder.0.weight', 'fusion.0.weight', 'q_head.weight'):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.actor.parameters()))

    def test_training_one_critic_changes_neither_actor_nor_other_critic(self):
        actor_before = {key: value.clone() for key, value in self.actor.state_dict().items()}
        second_before = {key: value.clone() for key, value in self.critics.q2.state_dict().items()}
        first_before = self.critics.q1.image_encoder.stages[0].conv1.weight.detach().clone()
        optimizer = torch.optim.Adam(self.critics.q1.parameters(), lr=1e-3)
        predictions = self.critics.q1(self.images, self.positions, self.actions)
        (predictions - torch.tensor([.2, .9])).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(first_before, self.critics.q1.image_encoder.stages[0].conv1.weight))
        for model, expected in ((self.actor, actor_before), (self.critics.q2, second_before)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_frozen_bc_actor_stays_frozen_but_critics_are_trainable_and_match_dtype(self):
        self.actor.double().requires_grad_(False)
        critics = TwinCritics.from_policy(self.actor)
        self.assertFalse(self.actor.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.actor.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in critics.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in critics.parameters()))
        values = critics(self.images.double(), self.positions.double(), self.actions.double())
        self.assertEqual(values.q1.dtype, torch.float64)

    def test_single_critic_factory_and_serialization_round_trip(self):
        critic = DapCritic.from_policy(self.actor)
        self.assertEqual(critic(self.images, self.positions, self.actions).shape, (self.B,))
        expected = self.critics(self.images, self.positions, self.actions)
        buffer = BytesIO()
        torch.save(self.critics.state_dict(), buffer)
        buffer.seek(0)
        restored = TwinCritics(self.L)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        actual = restored(self.images, self.positions, self.actions)
        torch.testing.assert_close(actual.q1, expected.q1, rtol=0, atol=0)
        torch.testing.assert_close(actual.q2, expected.q2, rtol=0, atol=0)

    def test_invalid_history_observations_and_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            TwinCritics(0)
        with self.assertRaises(TypeError):
            DapCritic(True)
        with self.assertRaises(ValueError):
            DapCritic(1).initialize_from_policy(self.actor)
        with self.assertRaises(ValueError):
            self.critics(self.images[:, :, :1], self.positions, self.actions)
        for actions in (self.actions[:, :A-1], self.actions.double(), torch.full_like(self.actions, float('nan')),
                        torch.full_like(self.actions, 1.1)):
            with self.assertRaises(ValueError):
                self.critics(self.images, self.positions, actions)
        with self.assertRaises(TypeError):
            self.critics(self.images, self.positions, self.actions.long())

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple MPS is unavailable')
    def test_apple_gpu_transfer_forward_and_action_gradients(self):
        actor = self.actor.to('mps').requires_grad_(False)
        critics = TwinCritics.from_policy(actor)
        actions = self.actions.to('mps').requires_grad_()
        values = critics(self.images.to('mps'), self.positions.to('mps'), actions)
        self.assertEqual(values.q1.device.type, 'mps')
        (values.q1.square().mean() + values.q2.square().mean()).backward()
        self.assertTrue(torch.isfinite(actions.grad).all().item())
        self.assertGreater(actions.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
