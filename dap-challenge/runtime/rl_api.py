"""Human-scored RL collection endpoints using the shared inference hardware owner."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import Field

from runtime.api import Owner
from runtime.rl_session import RLSession


class RunSelection(Owner):
    run_id: str = Field(min_length=1, max_length=100)


class AttemptStart(Owner):
    tasks: list[str]


class Grade(Owner):
    episode_id: str
    cued_tasks: list[str]
    answers: dict
    notes: str = Field(default='', max_length=5000)


class AttemptSelection(Owner):
    episode_id: str


def rl_router(session: RLSession) -> APIRouter:
    router = APIRouter(prefix='/api/rl')

    def execute(function, *args):
        try:
            return function(*args)
        except (ValueError, KeyError, FileNotFoundError) as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            raise HTTPException(503, str(error)) from error

    @router.get('/status')
    def status(client_id: str = ''):
        return execute(session.status, client_id)

    @router.post('/run/load')
    def load_run(body: RunSelection, request: Request):
        url = str(request.base_url).rstrip('/') + '/api/inference/policy'
        return execute(session.load_run, body.client_id, body.run_id, url)

    @router.post('/training/start')
    def start_training(owner: Owner):
        return execute(session.start_training, owner.client_id)

    @router.post('/training/stop')
    def stop_training(owner: Owner):
        return execute(session.stop_training, owner.client_id)

    @router.post('/session/end')
    def end(owner: Owner):
        return execute(session.end_session, owner.client_id)

    @router.post('/attempt/start')
    def start_attempt(body: AttemptStart):
        return execute(session.start_attempt, body.client_id, body.tasks)

    @router.post('/attempt/stop')
    def stop(owner: Owner):
        return execute(session.stop_attempt, owner.client_id)

    @router.post('/attempt/abort')
    def abort(owner: Owner):
        return execute(session.stop_attempt, owner.client_id, True)

    @router.post('/attempt/discard')
    def discard(body: AttemptSelection):
        return execute(session.discard, body.client_id, body.episode_id)

    @router.post('/attempt/score')
    def score(body: Grade):
        return execute(session.score, body.client_id, body.episode_id, body.cued_tasks, body.answers, body.notes)

    @router.get('/attempt/{episode_id}/video')
    def video(episode_id: str, client_id: str, camera: Literal['arm', 'mac']):
        execute(session.require_owner, client_id)
        path = execute(session.store.path, episode_id) / ('video.mp4' if camera == 'arm' else 'mac_video.mp4')
        if not path.is_file():
            raise HTTPException(404, 'No playable video for this attempt.')

        return FileResponse(path, media_type='video/mp4')

    return router
