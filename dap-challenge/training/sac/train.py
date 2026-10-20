"""Train SAC on finalized scored archives; optionally wait for asynchronously added episodes."""

import argparse
import json
from pathlib import Path
import signal
from threading import Event

import torch

from shared.config import load_config, save_config
from training.sac.config import validate_config
from training.sac.learner import SACLearner
from training.sac.run import LearnerRun, initialize_learner
from training.shared.metrics import log_metrics, start_run
from training.shared.utils import training_directory


def ingest_available(learner: SACLearner, run: LearnerRun, inbox: Path, maximum: int | None) -> None:
    """Poll atomically finalized archives between optimizer rounds, even during active training."""
    known = {record['path'] for record in learner.replay.records}
    for path in sorted(inbox.glob('*.pt')):
        if maximum is not None and len(learner.replay.records) >= maximum:
            break
        if str(path.resolve()) not in known:
            learner.ingest(path)
            run.save()


def train(args: argparse.Namespace) -> None:
    """Consume finite replay credits, saving clean stops without touching robot devices."""
    config = load_config(args.config)
    validate_config(config)
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('dap-challenge/runs').resolve()):
        raise ValueError('SAC output must live under dap-challenge/runs')

    with training_directory(output):
        train_owned(args, config, output)


def train_owned(args: argparse.Namespace, config: dict, output: Path) -> None:
    """Run only while holding exclusive ownership of this learner output directory."""
    if (output / 'learner' / 'latest.pt').exists() and not config['initialization']['resume_checkpoint']:
        raise ValueError('existing learner output requires initialization.resume_checkpoint in YAML')

    if args.watch and args.max_scored_episodes is None:
        raise ValueError('--watch requires an explicit scored-episode stopping budget')

    if args.max_scored_episodes is not None and args.max_scored_episodes < 1:
        raise ValueError('max scored episodes must be positive')

    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available on this Mac')

    learner = initialize_learner(config, args.reward_version, device)
    run = LearnerRun(learner, output)
    inbox = Path(args.episodes).resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    save_config(config, output / 'config.yaml')
    stopped = Event()
    handlers = {code: signal.getsignal(code) for code in (signal.SIGINT, signal.SIGTERM)}
    for code in handlers:
        signal.signal(code, lambda signum, frame: stopped.set())
    tracking = None
    try:
        tracking = start_run(**config['wandb'], config=config, directory=output,
                             run_id=learner.wandb_run_id, resume='must' if learner.wandb_run_id else None)
        learner.wandb_run_id = tracking.id
        run.save()
        while not stopped.is_set():
            run.check_publication()
            ingest_available(learner, run, inbox, args.max_scored_episodes)
            updated = False
            while learner.ready and not stopped.is_set():
                previous = learner.progress['actor_updates']
                metrics = learner.update()
                run.after_update(previous)
                log_metrics(tracking, metrics, learner.progress['critic_updates'])
                updated = True
                ingest_available(learner, run, inbox, args.max_scored_episodes)
            if updated:
                run.publish()
                run.save()
            exhausted_attempts = args.max_scored_episodes is not None and len(learner.replay.records) >= args.max_scored_episodes
            if exhausted_attempts or not args.watch:
                break
            stopped.wait(1.0)
        run.close()
        run.save()
        state = 'budget_complete' if learner.progress['earned_updates'] and learner.progress['critic_updates'] == learner.progress['earned_updates'] else 'waiting_for_scored_data'
        if stopped.is_set():
            state = 'stopped'
        (output / 'status.json').write_text(json.dumps({'state': state, 'publication': run.publisher.snapshot() if run.publisher else None, **learner.progress}, indent=2))
    except Exception as error:
        (output / 'status.json').write_text(json.dumps({'state': 'failed', 'error': str(error), 'publication': run.publisher.snapshot() if run.publisher else None, **learner.progress}, indent=2))
        if tracking is not None:
            tracking.finish(exit_code=1)
            tracking = None
        raise
    finally:
        run.close(flush=False, raise_errors=False)
        if tracking is not None:
            tracking.finish()
        for code, handler in handlers.items():
            signal.signal(code, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dap-challenge/configs/sac.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--reward-version', required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-scored-episodes', type=int)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
