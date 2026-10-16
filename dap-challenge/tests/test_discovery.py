"""Verify identity-based discovery and persistence without connecting hardware."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from recording.server import create_app
from training.shared.discovery import remember_profile, resolve_connection, resolve_profile


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.profile = {'leader_serial': 'black', 'follower_serial': 'white', 'camera_device_id': 'usb',
                        'leader_id': 'dap_leader', 'follower_id': 'dap_follower', 'width': 640, 'height': 480}
        self.inventory = {'ports': [{'port': '/new/leader', 'serial': 'black'},
                                    {'port': '/new/follower', 'serial': 'white'}],
                          'cameras': [{'index': 0, 'device_id': 'usb', 'name': 'USB'},
                                      {'index': 1, 'device_id': 'built-in', 'name': 'Mac'}], 'warnings': []}

    def test_resolves_reordered_devices_and_preserves_settings(self):
        settings, warnings = resolve_profile(self.profile, self.inventory)
        self.assertEqual(warnings, [])
        self.assertEqual(settings['leader_port'], '/new/leader')
        self.assertEqual(settings['camera_index'], 0)
        self.assertEqual(settings['width'], 640)
        self.assertEqual(settings['leader_id'], 'dap_leader')

    def test_missing_and_duplicate_identities_never_guess(self):
        self.inventory['ports'].append({'port': '/duplicate', 'serial': 'black'})
        self.inventory['cameras'] = []
        settings, warnings = resolve_profile(self.profile, self.inventory)
        self.assertEqual(settings['leader_port'], '')
        self.assertIsNone(settings['camera_index'])
        self.assertEqual(len(warnings), 2)

    def test_connect_resolves_again_instead_of_trusting_old_indices(self):
        settings = {**self.profile, 'leader_port': '/old/leader', 'follower_port': '/old/follower', 'camera_index': 8}
        resolved = resolve_connection(settings, self.inventory)
        self.assertEqual(resolved['camera_index'], 0)
        self.assertEqual(resolved['follower_port'], '/new/follower')
        self.inventory['cameras'] = []
        with self.assertRaises(ValueError):
            resolve_connection(settings, self.inventory)

    def test_mac_camera_identity_resolves_separately_and_cannot_reuse_arm_camera(self):
        profile = {**self.profile, 'mac_camera_device_id': 'built-in'}
        settings, warnings = resolve_profile(profile, self.inventory)
        self.assertEqual(settings['mac_camera_index'], 1)
        self.assertEqual(warnings, [])
        resolved = resolve_connection(settings, self.inventory)
        self.assertEqual(resolved['mac_camera_index'], 1)
        settings['mac_camera_device_id'] = 'usb'
        with self.assertRaises(ValueError):
            resolve_connection(settings, self.inventory)

    def test_capture_uses_names_even_when_indices_change(self):
        settings = {**self.profile, 'leader_port': '/old/leader', 'follower_port': '/old/follower',
                    'camera_index': 8, 'mac_camera_device_id': 'built-in', 'mac_camera_index': 9}
        self.inventory['cameras'].reverse()
        resolved = resolve_connection(settings, self.inventory)
        self.assertEqual(resolved['camera_name'], 'USB')
        self.assertEqual(resolved['mac_camera_name'], 'Mac')
        self.inventory['cameras'].append({'index': 2, 'device_id': 'other', 'name': 'USB identical'})
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            resolve_connection(settings, self.inventory)

    def test_profile_remembers_identities_not_transient_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            settings, _ = resolve_profile(self.profile, self.inventory)
            remember_profile(path, settings, self.inventory)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['camera_device_id'], 'usb')
            self.assertEqual(saved['leader_serial'], 'black')
            self.assertNotIn('camera_index', saved)
            self.assertNotIn('leader_port', saved)

    def test_endpoint_only_discovers_and_does_not_request_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            path.write_text(json.dumps(self.profile))
            with patch('recording.server.device_inventory', return_value=self.inventory), patch('recording.server.RecorderSession') as session:
                client = TestClient(create_app(profile_path=path))
                response = client.get('/api/devices')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['settings']['camera_index'], 0)
                session.return_value.request.assert_not_called()

    def test_disappearing_device_refuses_before_motor_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inventory['cameras'] = []
            with patch('recording.server.device_inventory', return_value=self.inventory), patch('recording.server.RecorderSession') as session:
                client = TestClient(create_app(profile_path=Path(folder) / 'profile.json'))
                response = client.post('/api/connect', json={**self.profile, 'leader_port': '/old/leader',
                                      'follower_port': '/old/follower', 'camera_index': 0})
                self.assertEqual(response.status_code, 409)
                session.return_value.request.assert_not_called()
