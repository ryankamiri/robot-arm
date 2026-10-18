"""Local inference endpoints with no persistent rollout or observation storage."""

from pathlib import Path
import pickle
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from runtime.policy import PolicyRunner, checkpoint_catalog
from runtime.session import RuntimeSession
from runtime.streaming import CameraPeers
from shared.discovery import device_inventory, resolve_connection


class Owner(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)


class LiveSettings(BaseModel):
    follower_port: str = ''
    follower_id: str = ''
    follower_serial: str | None = None
    calibration_dir: str | None = None
    camera_index: int = Field(default=0, ge=0)
    mac_camera_index: int | None = Field(default=None, ge=0)
    camera_device_id: str | None = None
    mac_camera_device_id: str | None = None
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    camera_fps: int = Field(default=30, gt=0)
    rotation_clockwise: Literal[0,90,180,270] = 90


class Connect(Owner):
    settings: LiveSettings


class Selection(BaseModel):
    checkpoint: str
    client_id: str = ''


class Start(Owner):
    mode: Literal['mean', 'sample'] = 'mean'


class Offer(Owner):
    sdp: str = Field(min_length=1, max_length=100_000)
    type: Literal['offer'] = 'offer'


class PeerRelease(Owner):
    peer_id: str


def inference_router(session: RuntimeSession, runs: Path, demo: bool, media: CameraPeers) -> APIRouter:
    router = APIRouter(prefix='/api/inference')

    def execute(command: str, payload: dict) -> dict:
        try:
            return session.request(command,payload)
        except (ValueError,KeyError,FileNotFoundError) as error:
            raise HTTPException(409,str(error)) from error
        except Exception as error:
            raise HTTPException(503,str(error)) from error

    @router.get('/status')
    def status(client_id: str = ''):
        return {**session.snapshot(),'demo':demo, 'owned_by_you':session.client_id == client_id and session.devices is not None}

    @router.get('/checkpoints')
    def checkpoints():
        return checkpoint_catalog(runs)

    @router.post('/load')
    def load(selection: Selection):
        if session.snapshot()['phase'] in ('running','resetting'):
            raise HTTPException(409,'Stop motion before loading a different checkpoint.')

        try:
            runner = PolicyRunner.from_checkpoint(runs,selection.checkpoint,session.settings['policy']['device'])
        except (ValueError,KeyError,OSError,RuntimeError,EOFError,pickle.UnpicklingError) as error:
            raise HTTPException(409,str(error)) from error

        return execute('load',{'runner':runner,'client_id':selection.client_id})

    @router.post('/connect')
    def connect(connection: Connect):
        values = connection.settings.model_dump()
        if not demo:
            if not values.get('follower_port') or not values.get('follower_id') or values.get('mac_camera_index') is None:
                raise HTTPException(422,'Select a calibrated follower and two distinct live cameras.')
            if values.get('camera_fps') != 30 or values.get('width') != 640 or values.get('height') != 480:
                raise HTTPException(422,'Use the trained 640×480 capture mode at 30 images per second.')

            try:
                inventory = device_inventory()
                values = resolve_connection({**values,'leader_port':'','leader_serial':None},inventory)
            except (ValueError,OSError) as error:
                raise HTTPException(409,str(error)) from error
        return execute('connect',{'settings':values,'client_id':connection.client_id})

    @router.post('/heartbeat')
    def heartbeat(owner: Owner):
        try:
            session.heartbeat(owner.client_id)
        except ValueError as error:
            raise HTTPException(409,str(error)) from error

        return {'ok':True}

    @router.post('/start')
    def start(request: Start):
        return execute('start',request.model_dump())

    @router.post('/stop')
    def stop(owner: Owner):
        return execute('stop',owner.model_dump())

    @router.post('/disconnect')
    def disconnect(owner: Owner):
        return execute('disconnect',owner.model_dump())

    @router.post('/pose/capture')
    def capture(owner: Owner):
        return execute('capture_pose',owner.model_dump())

    @router.post('/pose/return')
    def reset(owner: Owner):
        return execute('reset',owner.model_dump())

    @router.post('/preview/offer')
    async def offer(request: Offer):
        try:
            return await media.answer(request.client_id, request.sdp)
        except TimeoutError as error:
            raise HTTPException(504, 'Camera connection negotiation timed out.') from error
        except (ValueError, AssertionError) as error:
            raise HTTPException(409, str(error) or 'Invalid camera offer.') from error

    @router.post('/preview/close')
    async def close_preview(request: PeerRelease):
        await media.release(request.client_id, request.peer_id)
        return {'ok': True}

    return router
