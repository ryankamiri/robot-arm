"""Verify activity labels distinguish real changes from static command offsets."""

import unittest

import torch

from shared.normalization import JOINT_NAMES
from training.shared.activity import non_waiting_mask, activity_mae_metrics, activity_mae_totals


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.positions = torch.zeros((30,6),dtype=torch.float64)
        self.targets = torch.zeros((30,6),dtype=torch.float64)
        self.times = torch.arange(30,dtype=torch.float64)/30

    def test_static_target_offset_and_isolated_noise_are_not_activity(self):
        self.targets[:] = 20.
        self.positions[15,0] = .9
        mask = non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.01,.1)
        self.assertFalse(mask.any())
        self.assertEqual(mask.dtype,torch.bool)

    def test_measured_motion_or_target_change_can_independently_mark_activity(self):
        for changed in ('positions','targets'):
            with self.subTest(changed=changed):
                values = getattr(self,changed)
                values[15:,0] = 10.
                mask = non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.01,.1)
                self.assertTrue(mask[14:17].all())
                self.assertFalse(mask[:10].any())
                self.assertFalse(mask[20:].any())
                values[:] = 0.

    def test_gripper_change_uses_its_half_range_in_explicit_joint_order(self):
        self.targets[15:,0] = .7
        self.assertFalse(non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.01,.1).any())
        order = ('gripper',)+JOINT_NAMES[:-1]
        self.assertTrue(non_waiting_mask(self.positions,self.targets,self.times,order,.01,.1)[15])

    def test_single_row_and_bad_timestamps(self):
        self.assertFalse(non_waiting_mask(self.positions[:1],self.targets[:1],self.times[:1],JOINT_NAMES,.01,.1).any())
        for times in (torch.zeros(30,dtype=torch.float64),torch.full((30,),float('nan'),dtype=torch.float64)):
            with self.assertRaises(ValueError):
                non_waiting_mask(self.positions,self.targets,times,JOINT_NAMES,.01,.1)

    def test_configured_threshold_and_window_change_classification(self):
        self.targets[15:,0] = 2.
        self.assertTrue(non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.01,.1)[13])
        self.assertFalse(non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.03,.1).any())
        self.assertFalse(non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,.01,.01).any())

    def test_activity_settings_must_be_positive_and_finite(self):
        for invalid in (0.,-1.,float('nan'),float('inf'),True):
            for threshold,window in ((invalid,.1),(.01,invalid)):
                with self.assertRaises(ValueError):
                    non_waiting_mask(self.positions,self.targets,self.times,JOINT_NAMES,threshold,window)

    def test_absent_group_has_count_zero_and_no_fake_mae(self):
        for active in (False,True):
            metrics = activity_mae_metrics(activity_mae_totals(torch.tensor([.1,.3]),torch.full((2,),active)),'train')
            present = 'non_waiting' if active else 'waiting'
            absent = 'waiting' if active else 'non_waiting'
            self.assertEqual(metrics[f'train/{present}_count'],2)
            self.assertAlmostEqual(metrics[f'train/action_mae_{present}'],.2,places=6)
            self.assertEqual(metrics[f'train/{absent}_count'],0)
            self.assertNotIn(f'train/action_mae_{absent}',metrics)
