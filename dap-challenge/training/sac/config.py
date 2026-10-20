"""Validate the explicitly selected SAC recipe before constructing learner state."""

import math
from typing import Any, Mapping
from urllib.parse import urlparse


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported schedules and invalid optimizer values before any updates."""
    training = config['training']
    initialization = config['initialization']
    if not isinstance(initialization['critic_is_base'], bool) or not isinstance(training['warmup']['enabled'], bool):
        raise ValueError('critic_is_base and warmup.enabled must be boolean')

    resume = initialization['resume_checkpoint']
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError('resume_checkpoint must be null or a checkpoint path')

    for key in ('actor_checkpoint', 'critic_checkpoint'):
        path = initialization[key]
        if resume and path is None:
            continue
        if not isinstance(path, str) or not path.strip():
            raise ValueError('actor and critic checkpoint paths must be supplied in YAML unless resuming')

    if not resume and initialization['critic_is_base'] and not training['warmup']['enabled']:
        raise ValueError('base critics require warmup.enabled=true')

    publication = training['publication']
    if not isinstance(publication['enabled'], bool):
        raise ValueError('publication.enabled must be boolean')

    endpoint = urlparse(publication['url'])
    if endpoint.scheme != 'http' or endpoint.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('publication.url must identify a local HTTP endpoint')

    counts = [training['batch_size'], training['critic_updates_per_actor_update'],
              training['critic_updates_per_new_transition'], training['replay']['cache_episodes'],
              training['checkpoint']['every_critic_updates'], training['checkpoint']['keep_periodic'],
              training['publication']['every_actor_updates'], config['model']['history_length'],
              config['preprocessing']['image_width'], config['preprocessing']['image_height']]
    counts.extend(training['warmup'][key] for key in
                  ('minimum_scored_episodes', 'minimum_transitions', 'critic_only_updates'))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError('SAC counts must be positive integers')

    positive = [training['optimizer']['actor_learning_rate'], training['optimizer']['critic_learning_rate'],
                training['optimizer']['max_gradient_norm'], training['entropy']['learning_rate'],
                training['entropy']['initial_alpha'], training['publication']['timeout_s']]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError('optimizer rates, gradient norm and initial alpha must be finite and positive')

    for value in (training['discount'], training['target_critics']['tau']):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('discount and target tau must lie strictly between zero and one')

    if not math.isfinite(training['entropy']['target']):
        raise ValueError('entropy target must be finite')

    if training['optimizer']['name'] != 'adam' or training['replay']['sampling'] != 'uniform':
        raise ValueError('this learner supports Adam and uniform replay')

    flags = [training['entropy']['automatic'],
             training['replay']['replacement'], training['replay']['retain_all_scored_episodes'],
             training['checkpoint']['after_ingestion_cycle'], training['checkpoint']['on_clean_stop'],
             training['checkpoint']['keep_latest'], training['publication']['at_budget_end'],
             training['publication']['adopt_between_attempts']]
    if any(value is not True for value in flags):
        raise ValueError('this learner requires the agreed persistence, sampling and publication semantics')

    if training['entropy']['every_actor_updates'] != 1 or training['target_critics']['every_critic_updates'] != 1:
        raise ValueError('temperature updates every actor step and targets every critic round')
