"""Synthetic real-file checks for alignment, causal windows and BC batches."""

import json
import shutil
from pathlib import Path
import tempfile
import unittest

import av
from datasets import Dataset as HFDataset
from jaxtyping import UInt8
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from models.dimensions import A, C_RGB
from shared.normalization import JOINT_NAMES, denormalize_controller
from models.policy import DapPolicy
from training.bc.dataset import BCDataset, create_dataloaders
from shared.config import load_config
from training.shared.episodes import discover_episodes, load_episode
from shared.observations import preprocess_image


class BCDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write_episode(self, name: str, T: int = 4, frame_indices=None) -> Path:
        """Write tiny MP4/Parquet fixtures, not real robot demonstrations."""
        path = self.root / name
        path.mkdir()
        H_src, W_src = 6, 8
        with av.open(str(path / "video.mp4"), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = W_src, H_src, "yuv420p"
            for index in range(T):
                pixels: UInt8[np.ndarray, "H_src W_src C_in"] = np.full(
                    (H_src, W_src, C_RGB), 30 + index * 50, dtype=np.uint8
                )
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        shutil.copy2(path / 'video.mp4', path / 'mac_video.mp4')
        if frame_indices is None:
            frame_indices = list(range(T))
        table = {
            "time_index": list(range(T)),
            "frame_index": frame_indices,
            "mac_frame_index": list(range(T)),
            "mac_camera_timestamp_s": [index / 10 for index in range(T)],
            "timestamp_s": [index / 10 + 0.01 for index in range(T)],
            "camera_timestamp_s": [index / 10 for index in frame_indices],
            "action_timestamp_s": [index / 10 + 0.02 for index in range(T)],
            "positions": [[index / 10] * A for index in range(T)],
            "actions": [[-1 + 2 * index / max(T - 1, 1)] * A for index in range(T)],
        }
        for column_name in ("positions", "actions"):
            table[column_name] = denormalize_controller(torch.tensor(table[column_name]), JOINT_NAMES).tolist()
        pq.write_table(pa.table(table), path / "steps.parquet")
        (path / "metadata.json").write_text(json.dumps({
            "schema_version": 2,
            "episode_id": name,
            "joint_names": list(JOINT_NAMES),
            "normalized": False,
            "controller_convention": "lerobot_range",
            "calibration": {"source": "synthetic test fixture"},
            "status": "accepted",
            "camera": {"rotation_clockwise": 0},
            "mac_camera": {"rotation_clockwise": 0},
        }))
        return path

    def rewrite_columns(self, path: Path, **changes) -> None:
        """Modify a fixture table to exercise invalid or altered recordings."""
        columns = pq.read_table(path / "steps.parquet").to_pydict()
        columns.update(changes)
        pq.write_table(pa.table(columns), path / "steps.parquet")

    def test_preprocessing_preserves_rgb_and_uses_selected_rectangular_size(self):
        H_src, W_src, H_in, W_in = 6, 8, 168, 224
        pixels: UInt8[Tensor, "H_src W_src C_in"] = torch.zeros(
            H_src, W_src, C_RGB, dtype=torch.uint8
        )
        pixels[:, :, 0] = 255
        image = preprocess_image(pixels, W_in, H_in)
        self.assertEqual(image.shape, (C_RGB, H_in, W_in))
        self.assertEqual(image.dtype, torch.float32)
        torch.testing.assert_close(image[0], torch.ones(H_in, W_in))
        torch.testing.assert_close(image[1:], torch.zeros(C_RGB - 1, H_in, W_in))

    def test_preprocessing_undoes_saved_clockwise_rotation_before_resize(self):
        original = torch.arange(6 * 8 * C_RGB, dtype=torch.uint8).reshape(6, 8, C_RGB)
        recorded = torch.rot90(original, k=-1, dims=(0, 1))
        torch.testing.assert_close(preprocess_image(recorded, 8, 6, 90), preprocess_image(original, 8, 6))
        torch.testing.assert_close(preprocess_image(original, 8, 6, 0), preprocess_image(original, 8, 6))

    def test_requires_both_videos_and_explicit_rotation(self):
        path = self.write_episode('missing_mac')
        (path / 'mac_video.mp4').unlink()
        with self.assertRaisesRegex(ValueError, 'missing required camera video'):
            load_episode(path, 8, 6)
        path = self.write_episode('missing_rotation')
        metadata = json.loads((path / 'metadata.json').read_text())
        del metadata['camera']['rotation_clockwise']
        (path / 'metadata.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'rotation_clockwise'):
            load_episode(path, 8, 6)

    def test_rejects_invalid_secondary_indices_and_capture_times(self):
        mutations = (
            {'mac_frame_index': [0, 1, 2, 99]},
            {'mac_camera_timestamp_s': [0., 0.1, 0.2, 0.9]},
            {'mac_frame_index': [0, 0, 2, 3], 'mac_camera_timestamp_s': [0., 0.1, 0.2, 0.3]},
        )
        for index, mutation in enumerate(mutations):
            path = self.write_episode(f'bad_mac_{index}')
            self.rewrite_columns(path, **mutation)
            with self.assertRaises(ValueError):
                load_episode(path, 8, 6)

    def test_loads_real_video_and_hf_table_without_modifying_endpoint_labels(self):
        path = self.write_episode("episode_0")
        H_in, W_in = 6, 8
        episode = load_episode(path, W_in, H_in)
        self.assertIsInstance(episode.table, HFDataset)
        self.assertEqual(episode.frames[0].shape, (len(episode), C_RGB, H_in, W_in))
        self.assertTrue(all((view >= 0).all() for view in episode.frames))
        self.assertTrue(all((view <= 1).all() for view in episode.frames))
        self.assertTrue((episode.actions[0] == -1).all())
        self.assertTrue((episode.actions[-1] == 1).all())

    def test_padding_is_causal_and_target_stays_at_current_time(self):
        episode = load_episode(self.write_episode("episode_0"), 8, 6)
        L = 3
        dataset = BCDataset([episode], L, .01, .1)
        for time_index, indices in enumerate(([0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3])):
            sample = dataset[time_index]
            torch.testing.assert_close(sample["images"], torch.stack([view[indices] for view in episode.frames], dim=1))
            torch.testing.assert_close(sample["positions"], episode.positions[indices])
            torch.testing.assert_close(sample["expert_action"], episode.actions[time_index])

    def test_windows_never_cross_episode_boundaries(self):
        first = load_episode(self.write_episode("episode_0"), 8, 6)
        second = load_episode(self.write_episode("episode_1"), 8, 6)
        dataset = BCDataset([first, second], 3, .01, .1)
        sample = dataset[len(first)]
        self.assertEqual(sample["episode_id"], second.episode_id)
        self.assertEqual(sample["time_index"], 0)
        torch.testing.assert_close(sample["positions"], second.positions[[0, 0, 0]])
        torch.testing.assert_close(sample["images"], torch.stack([view[[0, 0, 0]] for view in second.frames], dim=1))

    def test_activity_is_episode_local_and_keeps_every_target(self):
        episodes = []
        for index in range(2):
            path = self.write_episode(f'constant_{index}')
            self.rewrite_columns(path, positions=[[0.] * 5 + [50.]] * 4,
                                 actions=[[float(index*20)] * 5 + [50.]] * 4)
            episodes.append(load_episode(path,8,6))
        dataset = BCDataset(episodes,3,.01,.1)
        self.assertEqual(len(dataset),8)
        self.assertFalse(any(dataset[index]['non_waiting'] for index in range(len(dataset))))
        for index,episode in enumerate(episodes):
            for time_index in range(len(episode)):
                sample = dataset[index*4+time_index]
                torch.testing.assert_close(sample['expert_action'],episode.actions[time_index])

    def test_repeated_and_skipped_video_frames_use_explicit_indices(self):
        path = self.write_episode("episode_0", T=3, frame_indices=[0, 0, 2])
        episode = load_episode(path, 8, 6)
        sample = BCDataset([episode], 3, .01, .1)[2]
        torch.testing.assert_close(sample["images"], torch.stack((episode.frames[0][[0, 0, 2]], episode.frames[1][[0, 1, 2]]), dim=1))
        self.assertGreater(sample["images"][2].mean().item(), sample["images"][1].mean().item())
        torch.testing.assert_close(sample["positions"], episode.positions)
        torch.testing.assert_close(sample["expert_action"], episode.actions[2])

    def test_rejects_future_capture_invalid_frames_and_raw_units(self):
        mutations = (
            {"camera_timestamp_s": [0.5, 0.1, 0.2, 0.3]},
            {"frame_index": [0, 1, 2, 99]},
            {"positions": [[200.0] * A] * 4},
            {"action_timestamp_s": [0.0, 0.12, 0.22, 0.32]},
            {"time_index": [0, 1, 3, 4]},
            {"frame_index": [0.0, 1.0, 2.0, 3.0]},
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                path = self.write_episode(f"invalid_{index}")
                self.rewrite_columns(path, **mutation)
                with self.assertRaises(ValueError):
                    load_episode(path, 8, 6)

    def test_requires_explicit_normalization_metadata(self):
        path = self.write_episode("episode_0")
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["normalized"] = True
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            load_episode(path, 8, 6)

    def test_rollout_split_and_shuffled_epochs_cover_every_target_once(self):
        paths = [self.write_episode(f"episode_{index}", T=index + 1) for index in range(5)]
        config = {
            "preprocessing": {"image_width": 8, "image_height": 6},
            "model": {"history_length": 3},
            "training": {"batch_size": 3, "activity": {"motion_threshold": .01, "window_half_s": .1}},
            "dataset": {"validation_fraction": 0.2},
        }
        first = create_dataloaders(paths, config, seed=7)
        second = create_dataloaders(list(reversed(paths)), config, seed=7)
        self.assertEqual(first.train_episode_ids, second.train_episode_ids)
        self.assertEqual(first.validation_episode_ids, second.validation_episode_ids)
        self.assertEqual(len(first.train_episode_ids), 4)
        self.assertEqual(len(first.validation_episode_ids), 1)
        self.assertFalse(set(first.train_episode_ids) & set(first.validation_episode_ids))

        def records(loader):
            return [
                (episode_id, int(time_index))
                for batch in loader
                for episode_id, time_index in zip(batch["episode_id"], batch["time_index"])
            ]

        initial_order = records(first.train)
        self.assertEqual(initial_order, records(second.train))
        expected = {
            (episode.episode_id, time_index)
            for episode in first.train.dataset.episodes
            for time_index in range(len(episode))
        }
        self.assertEqual(len(initial_order), len(expected))
        self.assertEqual(set(initial_order), expected)
        next_order = records(first.train)
        self.assertEqual(set(next_order), expected)
        self.assertNotEqual(initial_order, next_order)
        validation_order = records(first.validation)
        self.assertEqual(validation_order, records(first.validation))

    def test_selected_config_produces_a_batch_the_actor_accepts(self):
        paths = [self.write_episode(f"episode_{index}") for index in range(2)]
        config = load_config(Path(__file__).parents[1] / "configs/bc.yaml")
        loaders = create_dataloaders(paths, config, seed=7)
        batch = next(iter(loaders.train))
        B = batch["images"].shape[0]
        L = config["model"]["history_length"]
        H_in = config["preprocessing"]["image_height"]
        W_in = config["preprocessing"]["image_width"]
        self.assertEqual(batch["images"].shape, (B, L, 2, C_RGB, H_in, W_in))
        self.assertEqual(batch["positions"].shape, (B, L, A))
        self.assertEqual(batch["expert_action"].shape, (B, A))
        self.assertEqual(batch["non_waiting"].shape, (B,))
        self.assertEqual(batch["non_waiting"].dtype, torch.bool)
        with torch.inference_mode():
            output = DapPolicy(L).eval()(batch["images"], batch["positions"])
        self.assertEqual(output.mean.shape, (B, A))

    def test_discovery_and_split_reject_duplicate_rollouts(self):
        path = self.write_episode("episode_0")
        self.assertEqual(discover_episodes(self.root), [path])
        config = {
            "preprocessing": {"image_width": 8, "image_height": 6},
            "model": {"history_length": 3},
            "training": {"batch_size": 3, "activity": {"motion_threshold": .01, "window_half_s": .1}},
            "dataset": {"validation_fraction": 0.2},
        }
        with self.assertRaises(ValueError):
            create_dataloaders([path, path], config, seed=7)


if __name__ == "__main__":
    unittest.main()
