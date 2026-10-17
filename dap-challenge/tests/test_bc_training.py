"""Check BC likelihood, weighted validation, checkpoint selection and exact CPU resume."""

import copy
import json
import math
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
from jaxtyping import Float, UInt8
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from torch.utils.data import DataLoader

from models.architecture import GaussianParameters
from models.dimensions import A
from models.policy import DapPolicy
from training.bc.loss import expert_nll
from training.bc.train import BatchMetrics, batch_metrics, evaluate, train
from training.shared.config import load_config
from training.shared.normalization import JOINT_NAMES, denormalize_controller


class BCTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(Path(__file__).parents[1] / 'configs/bc.yaml')
        self.config['dataset']['root'] = str(self.root / 'data')
        self.config['preprocessing'] = {'image_width': 16, 'image_height': 12}
        self.config['training']['max_epochs'] = 2
        self.config['training']['batch_size'] = 2
        self.config['training']['evaluation']['every_steps'] = 2
        self.run = Mock(id='bc-test-run')

    def episodes(self):
        for number in range(3):
            path = self.root / 'data' / f'episode_{number}'
            path.mkdir(parents=True)
            with av.open(str(path / 'video.mp4'), 'w') as container:
                stream = container.add_stream('mpeg4', rate=30)
                stream.width, stream.height, stream.pix_fmt = 16, 12, 'yuv420p'
                for index in range(3):
                    pixels: UInt8[np.ndarray, 'H_src W_src C_in'] = np.full((12, 16, 3), index * 60, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            shutil.copy2(path / 'video.mp4', path / 'mac_video.mp4')
            actions = denormalize_controller(torch.tensor([[-1.] * A, [0.] * A, [1.] * A]), JOINT_NAMES)
            positions = denormalize_controller(torch.zeros(3, A), JOINT_NAMES)
            pq.write_table(pa.table({
                'time_index': [0, 1, 2], 'frame_index': [0, 1, 2],
                'mac_frame_index': [0, 1, 2], 'mac_camera_timestamp_s': [0., 0.03, 0.06],
                'camera_timestamp_s': [0., 0.03, 0.06], 'timestamp_s': [0.001, 0.031, 0.061],
                'action_timestamp_s': [0.002, 0.032, 0.062],
                'positions': positions.tolist(), 'actions': actions.tolist(),
            }), path / 'steps.parquet')
            (path / 'metadata.json').write_text(json.dumps({
                'schema_version': 2, 'episode_id': path.name, 'joint_names': list(JOINT_NAMES),
                'normalized': False, 'controller_convention': 'lerobot_range',
                'calibration': {'source': 'synthetic test'}, 'status': 'accepted',
                'camera': {'rotation_clockwise': 0}, 'mac_camera': {'rotation_clockwise': 0},
            }))

    def test_nll_matches_pytorch_transformed_distribution_and_has_gradients(self):
        mean: Float[Tensor, 'B A'] = torch.full((2, A), 0.2, requires_grad=True)
        log_std: Float[Tensor, 'B A'] = torch.full((2, A), -0.3, requires_grad=True)
        targets: Float[Tensor, 'B A'] = torch.linspace(-0.8, 0.8, 2 * A).reshape(2, A)
        distribution = Independent(TransformedDistribution(Normal(mean, log_std.exp()), [TanhTransform()]), 1)
        losses = expert_nll(GaussianParameters(mean, log_std), targets, 1e-4)
        self.assertEqual(losses.shape, (2,))
        torch.testing.assert_close(losses, -distribution.log_prob(targets))
        losses.mean().backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(log_std.grad).all())
        self.assertGreater(mean.grad.abs().sum().item(), 0)
        self.assertGreater(log_std.grad.abs().sum().item(), 0)

    def test_endpoints_remain_unmodified_and_invalid_labels_are_rejected(self):
        targets = torch.tensor([[-1., 1., 0., -1., 1., 0.]])
        original = targets.clone()
        parameters = GaussianParameters(torch.zeros(1, A), torch.zeros(1, A))
        self.assertTrue(torch.isfinite(expert_nll(parameters, targets, 1e-4)).all())
        torch.testing.assert_close(targets, original)
        for epsilon in (0., 1., float('nan'), 1e-30):
            with self.assertRaises(ValueError):
                expert_nll(parameters, targets, epsilon)
        for value in (1.1, float('nan')):
            with self.assertRaises(ValueError):
                expert_nll(parameters, torch.full((1, A), value), 1e-4)

    def test_action_mae_uses_deterministic_bounded_actions_not_pre_tanh_mean(self):
        policy = Mock(return_value=GaussianParameters(torch.full((2, A), math.atanh(0.5)), torch.zeros(2, A)))
        batch = {'images': torch.zeros(2, 3, 2, 3, 6, 8), 'positions': torch.zeros(2, 3, A),
                 'expert_action': torch.tensor([[0.] * A, [1.] * A])}
        values = batch_metrics(policy, batch, torch.device('cpu'), 1e-4)
        torch.testing.assert_close(values.action_mae, torch.full((2,), 0.5))
        policy.assert_called_once()

    def test_validation_weights_samples_including_partial_batch_and_restores_mode(self):
        policy = DapPolicy(3)
        loader = DataLoader([0, 1, 2], batch_size=2)
        with patch('training.bc.train.batch_metrics', side_effect=[
            BatchMetrics(torch.tensor([1., 2.]), torch.tensor([0., 0.])),
            BatchMetrics(torch.tensor([9.]), torch.tensor([0.9])),
        ]):
            metrics = evaluate(policy, loader, torch.device('cpu'), 1e-4)
            self.assertEqual(metrics['val/nll'], 4.)
            self.assertAlmostEqual(metrics['val/action_mae'], 0.3, places=6)
        self.assertTrue(policy.training)
        with patch('training.bc.train.batch_metrics', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                evaluate(policy, loader, torch.device('cpu'), 1e-4)
        self.assertTrue(policy.training)

    def test_latest_and_best_keep_distinct_steps_and_rollout_ids(self):
        self.episodes()
        self.config['training']['evaluation']['every_steps'] = 1
        output = self.root / 'run'
        with patch('training.bc.train.start_run', return_value=self.run):
            with patch('training.bc.train.evaluate', side_effect=[{'val/nll': value, 'val/action_mae': 0.1} for value in [2., 1., 3., 4., 5., 6.]]):
                train(self.config, output, 'cpu', 'test', 'disabled')
        latest = torch.load(output / 'latest.pt', weights_only=True)
        best = torch.load(output / 'best.pt', weights_only=True)
        self.assertEqual(latest['progress']['optimizer_step'], 6)
        self.assertEqual(best['progress']['optimizer_step'], 2)
        self.assertEqual(latest['extra_state']['best_validation_nll'], 1.)
        split = latest['extra_state']['split']
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(len(split['validation']), 1)
        self.assertEqual(sorted(path.name for path in output.glob('*.pt')), ['best.pt', 'latest.pt'])
        logged = [call.args[0] for call in self.run.log.call_args_list]
        self.assertTrue(all('train/action_mae' in row for row in logged))
        self.assertTrue(all('val/action_mae' in row for row in logged))
        self.run.finish.assert_called_once()
        with self.assertRaises(FileExistsError):
            train(self.config, output, 'cpu', 'test', 'disabled')

    def test_interrupted_mid_epoch_resumes_to_identical_weights_and_optimizer(self):
        self.episodes()
        with patch('training.bc.train.start_run', return_value=self.run):
            train(self.config, self.root / 'full', 'cpu', 'test', 'disabled')
            with patch('training.bc.train.log_metrics', side_effect=RuntimeError('interruption')):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    train(self.config, self.root / 'resumed', 'cpu', 'test', 'disabled')
            train(self.config, self.root / 'resumed', 'cpu', 'test', 'disabled', self.root / 'resumed/latest.pt')
        full = torch.load(self.root / 'full/latest.pt', weights_only=True)
        resumed = torch.load(self.root / 'resumed/latest.pt', weights_only=True)
        self.assertEqual(full['progress'], resumed['progress'])
        self.assertEqual(full['extra_state']['best_validation_nll'], resumed['extra_state']['best_validation_nll'])
        for key, value in full['models']['actor'].items():
            torch.testing.assert_close(value, resumed['models']['actor'][key], rtol=0, atol=0)
        for key, state in full['optimizers']['actor']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, resumed['optimizers']['actor']['state'][key][name], rtol=0, atol=0)

    def test_final_validation_runs_when_interval_exceeds_training_length(self):
        self.episodes()
        self.config['training']['evaluation']['every_steps'] = 100
        with patch('training.bc.train.start_run', return_value=self.run):
            with patch('training.bc.train.evaluate', return_value={'val/nll': 2., 'val/action_mae': 0.1}) as validation:
                train(self.config, self.root / 'final', 'cpu', 'test', 'disabled')
        validation.assert_called_once()
        self.assertTrue((self.root / 'final/best.pt').is_file())

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple MPS unavailable')
    def test_full_training_loop_on_mps_with_disabled_wandb(self):
        self.episodes()
        self.config['training']['max_epochs'] = 1
        output = train(self.config, self.root / 'mps', 'mps', 'robot-arm-tests', 'disabled')
        checkpoint = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['progress']['optimizer_step'], 3)
        self.assertTrue(math.isfinite(checkpoint['extra_state']['best_validation_nll']))
        self.assertTrue((output / 'best.pt').is_file())

    def test_resume_rejects_changed_config(self):
        self.episodes()
        with patch('training.bc.train.start_run', return_value=self.run):
            train(self.config, self.root / 'run', 'cpu', 'test', 'disabled')
        changed = copy.deepcopy(self.config)
        changed['training']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'same config'):
            train(changed, self.root / 'run', 'cpu', 'test', 'disabled', self.root / 'run/latest.pt')


if __name__ == '__main__':
    unittest.main()
