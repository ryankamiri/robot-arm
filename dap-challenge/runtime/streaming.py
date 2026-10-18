"""Two cached camera views sent over one local WebRTC peer connection."""

import asyncio
from collections.abc import Callable
from uuid import uuid4

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import SessionDescription
from av import VideoFrame

from shared.devices import CameraFrame


class CameraTrack(VideoStreamTrack):
    def __init__(self, frame: Callable[[], CameraFrame]):
        super().__init__()
        self.frame = frame

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        try:
            source = self.frame()
        except ValueError as error:
            self.stop()
            raise MediaStreamError from error

        frame = VideoFrame.from_ndarray(source.rgb, format='rgb24')
        frame.pts, frame.time_base = pts, time_base
        return frame


class CameraPeers:
    def __init__(self, frame: Callable, identity: Callable):
        self.frame, self.identity = frame, identity
        self.peers: dict[str, tuple[str, RTCPeerConnection, asyncio.Task]] = {}
        self.negotiation = asyncio.Lock()

    async def close_peer(self, peer_id: str) -> None:
        entry = self.peers.pop(peer_id, None)
        if entry is None:
            return

        _, peer, watcher = entry
        if watcher is not asyncio.current_task():
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        for sender in peer.getSenders():
            if sender.track:
                sender.track.stop()
        await peer.close()

    async def close(self) -> None:
        for peer_id in list(self.peers):
            await self.close_peer(peer_id)

    async def watch(self, peer_id: str, owner: str, identity: object, peer: RTCPeerConnection) -> None:
        try:
            async with asyncio.timeout(15):
                while peer.connectionState not in ('connected', 'failed', 'closed'):
                    if self.identity(owner) is not identity:
                        return
                    await asyncio.sleep(.1)
            while peer.connectionState == 'connected' and self.identity(owner) is identity:
                await asyncio.sleep(.25)
        except (ValueError, TimeoutError):
            pass
        finally:
            await self.close_peer(peer_id)

    async def answer(self, owner: str, sdp: str) -> dict:
        async with self.negotiation:
            identity = self.identity(owner)
            for peer_id, (client, _, _) in list(self.peers.items()):
                if client == owner:
                    await self.close_peer(peer_id)
            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            tracks = []
            try:
                async with asyncio.timeout(10):
                    await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type='offer'))
                    transceivers = peer.getTransceivers()
                    offered = SessionDescription.parse(sdp).media
                    if len(offered) != 2 or any(item.kind != 'video' or item.direction != 'recvonly' for item in offered):
                        raise ValueError('Offer must request exactly two receive-only video tracks.')

                    views = {}
                    for view, transceiver in zip(('arm', 'mac'), transceivers):
                        track = CameraTrack(lambda view=view: self.frame(view, owner, identity))
                        tracks.append(track)
                        transceiver.sender.replaceTrack(track)
                        transceiver.direction = 'sendonly'
                        views[transceiver.mid] = view
                    await peer.setLocalDescription(await peer.createAnswer())
                    if self.identity(owner) is not identity:
                        raise ValueError('Camera session changed during negotiation.')

                peer_id = uuid4().hex
                watcher = asyncio.create_task(self.watch(peer_id, owner, identity, peer))
                self.peers[peer_id] = (owner, peer, watcher)
                return {'sdp': peer.localDescription.sdp, 'type': 'answer', 'views': views, 'peer_id': peer_id}
            except BaseException:
                for track in tracks:
                    track.stop()
                await peer.close()
                raise

    async def release(self, owner: str, peer_id: str) -> None:
        entry = self.peers.get(peer_id)
        if entry and entry[0] == owner:
            await self.close_peer(peer_id)
