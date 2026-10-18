"""Local FastAPI recorder API; start with python -m recording.server."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from serial.tools.list_ports import comports
import uvicorn

from shared.discovery import camera_source, device_inventory, load_profile, remember_profile, resolve_profile, resolve_connection
from recording.session import RecorderSession
from shared.devices import SO101Devices
from shared.demo_devices import DemoDevices, DemoFollower
from shared.config import load_config
from runtime.api import inference_router
from runtime.session import RuntimeSession
from runtime.streaming import CameraPeers
from shared.devices import FollowerDevices

PROJECT = Path(__file__).resolve().parents[1]


class ConnectionSettings(BaseModel):
    leader_port: str = ''
    follower_port: str = ''
    leader_id: str = ''
    follower_id: str = ''
    calibration_dir: str | None = None
    camera_index: Annotated[int, Field(ge=0)] = 0
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    camera_fps: Annotated[int, Field(gt=0)] = 30
    rotation_clockwise: Literal[0, 90, 180, 270] = 90
    leader_serial: str | None = None
    follower_serial: str | None = None
    camera_device_id: str | None = None
    mac_camera_index: Annotated[int, Field(ge=0)] | None = None
    mac_camera_device_id: str | None = None

    @model_validator(mode='after')
    def validate_dimensions(self):
        if self.width % 2 or self.height % 2:
            raise ValueError('H.264 yuv420p requires even capture width and height')

        return self


class CameraSettings(BaseModel):
    camera_index: Annotated[int, Field(ge=0)]


class EpisodeSelection(BaseModel):
    episode_id: str


class EpisodeKeep(EpisodeSelection):
    name: str = ''
    notes: str = ''


def create_app(session: RecorderSession | None = None, demo: bool = False, profile_path: Path | None = None,
               runtime: RuntimeSession | None = None) -> FastAPI:
    if profile_path is None:
        profile_path = PROJECT / 'local' / 'connection.json'

    config = load_config(PROJECT / 'configs' / 'bc.yaml')
    root = PROJECT.parent / config['dataset']['root']
    if demo:
        root = root / '.demo'
    if not session:
        session = RecorderSession(root, (lambda config: DemoDevices()) if demo else SO101Devices)
    if runtime is None:
        pose_name = 'runtime-pose-demo.yaml' if demo else 'runtime-pose.yaml'
        runtime = RuntimeSession(load_config(PROJECT / 'configs/runtime.yaml'), profile_path.parent / pose_name,
                                 session.lease, (lambda settings: DemoFollower()) if demo else FollowerDevices)

    media = CameraPeers(runtime.preview_frame, runtime.preview_identity)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session.start()
        runtime.start()
        try:
            yield
        finally:
            try:
                await media.close()
            finally:
                try:
                    runtime.close()
                finally:
                    session.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(inference_router(runtime, PROJECT / 'runs', demo, media))

    @app.middleware('http')
    async def local_commands_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method == 'POST' and origin and urlparse(origin).hostname not in ('localhost', '127.0.0.1'):
            return JSONResponse({'detail': 'Recorder commands must come from the local app.'}, status_code=403)

        return await call_next(request)

    def execute(command: str, payload: dict | None = None):
        try:
            return session.request(command, payload)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, KeyError) as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            raise HTTPException(503, str(error)) from error

    @app.get('/api/status')
    def status():
        return {**session.snapshot(), 'demo': demo}

    @app.get('/api/ports')
    def ports():
        return [{'port': port.device, 'description': port.description} for port in comports()]

    @app.get('/api/devices')
    def devices():
        inventory = device_inventory() if not demo else {'ports': [], 'cameras': [], 'warnings': []}
        try:
            settings, warnings = resolve_profile(load_profile(profile_path), inventory)
        except (OSError, ValueError) as error:
            raise HTTPException(503, f'Cannot read saved connection: {error}') from error

        return {**inventory, 'settings': settings, 'warnings': warnings}

    @app.post('/api/camera/inspect')
    def camera(settings: CameraSettings):
        try:
            source = camera_source(settings.camera_index, device_inventory())
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

        return execute('inspect_camera', {'camera_index': source})

    @app.post('/api/connect')
    def connect(settings: ConnectionSettings):
        if not demo and (not all((settings.leader_port.strip(), settings.follower_port.strip(),
                                 settings.leader_id.strip(), settings.follower_id.strip()))
                         or settings.leader_port == settings.follower_port):
            raise HTTPException(422, 'Provide distinct arm ports and both calibrated arm IDs.')

        values = settings.model_dump()
        if not demo:
            inventory = device_inventory()
            try:
                values = resolve_connection(values, inventory)
                remember_profile(profile_path, values, inventory)
            except (OSError, ValueError) as error:
                raise HTTPException(409, str(error)) from error

        hardware_settings = {key: value for key, value in values.items()
                             if key not in ('leader_serial', 'follower_serial', 'camera_device_id', 'mac_camera_device_id')}
        return execute('connect', hardware_settings)

    @app.post('/api/recording/start')
    def start():
        return execute('start')

    @app.post('/api/recording/stop')
    def stop():
        return execute('stop')

    @app.post('/api/disconnect')
    def disconnect():
        return execute('disconnect')

    @app.post('/api/reset/ready')
    def ready():
        return execute('ready')

    @app.post('/api/episodes/keep')
    def keep(selection: EpisodeKeep):
        return execute('keep', selection.model_dump())

    @app.post('/api/episodes/discard')
    def discard(selection: EpisodeSelection):
        return execute('discard', selection.model_dump())

    @app.get('/api/episodes/{episode_id}/video')
    def video(episode_id: str, camera: Literal['arm', 'mac'] = 'arm'):
        try:
            metadata = session.store.metadata(episode_id)
            if metadata['status'] == 'recording':
                raise HTTPException(409, 'Stop recording before previewing the video.')

            path = session.store.path(episode_id) / ('mac_video.mp4' if camera == 'mac' else 'video.mp4')
            if not path.is_file():
                raise HTTPException(404, 'This incomplete take has no playable video.')

            return FileResponse(path, media_type='video/mp4')
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(404, str(error)) from error

    dist = PROJECT / 'viewer' / 'dist'
    if dist.is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')

        @app.get('/record/bc')
        @app.get('/inference')
        @app.get('/')
        def viewer():
            return FileResponse(dist / 'index.html')

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Use synthetic devices, never physical hardware.')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(create_app(demo=args.demo), host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
