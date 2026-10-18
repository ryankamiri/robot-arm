"""Exercise real local WebRTC transport without opening physical devices."""

import asyncio
import time
import unittest

from aiortc import RTCConfiguration, RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
import numpy as np

from runtime.streaming import CameraPeers, CameraTrack
from shared.devices import CameraFrame


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.current = object()
        self.manager = CameraPeers(self.frame, self.identity)
        self.browser = RTCPeerConnection(RTCConfiguration(iceServers=[]))

    def identity(self, owner):
        if owner != 'owner' or self.current is None:
            raise ValueError('Not connected.')

        return self.current

    def frame(self, view, owner, identity):
        if self.identity(owner) is not identity:
            raise ValueError('Session changed.')

        pixels = np.zeros((48,64,3), dtype=np.uint8)
        pixels[:,:,0 if view == 'arm' else 2] = 200
        return CameraFrame(pixels, 1, time.monotonic())

    async def asyncTearDown(self):
        await self.manager.close()
        await self.browser.close()

    async def offer(self):
        self.browser.addTransceiver('video', direction='recvonly')
        self.browser.addTransceiver('video', direction='recvonly')
        await self.browser.setLocalDescription(await self.browser.createOffer())
        return self.browser.localDescription.sdp

    async def test_two_distinct_views_decode_and_disconnect_cleans_peer(self):
        received = {}

        @self.browser.on('track')
        def track(track):
            transceiver = next(item for item in self.browser.getTransceivers() if item.receiver.track is track)
            received[transceiver.mid] = track

        answer = await self.manager.answer('owner', await self.offer())
        await self.browser.setRemoteDescription(RTCSessionDescription(answer['sdp'], answer['type']))
        for mid, view in answer['views'].items():
            frame = await asyncio.wait_for(received[mid].recv(), 10)
            pixels = frame.to_ndarray(format='rgb24')
            self.assertEqual(pixels.shape, (48,64,3))
            channel = 0 if view == 'arm' else 2
            self.assertGreater(pixels[:,:,channel].mean(), 170)
            self.assertLess(pixels[:,:,2-channel].mean(), 30)
            following = await asyncio.wait_for(received[mid].recv(), 3)
            self.assertGreater(following.pts, frame.pts)
        self.current = None
        for _ in range(30):
            if not self.manager.peers:
                break
            await asyncio.sleep(.05)
        self.assertFalse(self.manager.peers)

    async def test_invalid_owner_and_offer_leave_no_peers(self):
        with self.assertRaises(ValueError):
            await self.manager.answer('stranger', await self.offer())
        with self.assertRaises((ValueError, AssertionError)):
            await self.manager.answer('owner', 'invalid')
        self.assertFalse(self.manager.peers)

    async def test_sending_camera_offer_is_rejected(self):
        sdp = (await self.offer()).replace('a=recvonly', 'a=sendrecv')
        with self.assertRaises(ValueError):
            await self.manager.answer('owner', sdp)
        self.assertFalse(self.manager.peers)

    async def test_replacement_and_explicit_release(self):
        sdp = await self.offer()
        first = await self.manager.answer('owner', sdp)
        previous = self.manager.peers[first['peer_id']][1]
        second = await self.manager.answer('owner', sdp)
        self.assertEqual(previous.connectionState, 'closed')
        self.assertEqual(len(self.manager.peers), 1)
        await self.manager.release('stranger', second['peer_id'])
        self.assertEqual(len(self.manager.peers), 1)
        await self.manager.release('owner', second['peer_id'])
        self.assertFalse(self.manager.peers)

    async def test_track_uses_latest_cached_frame_without_queue(self):
        value = [20]

        def latest():
            return CameraFrame(np.full((48,64,3),value[0],dtype=np.uint8),value[0],time.monotonic())

        track = CameraTrack(latest)
        first = await track.recv()
        value[0] = 80
        next_frame = await track.recv()
        self.assertEqual(next_frame.to_ndarray()[0,0,0], 80)
        self.assertGreater(next_frame.pts, first.pts)
        track.stop()
        with self.assertRaises(MediaStreamError):
            await track.recv()
