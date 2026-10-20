"""Versioned human rubrics shared by the website, collector and SAC replay."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from shared.config import load_config


def load_rubric(path: Path) -> dict:
    rubric = load_config(path)
    tasks = rubric['tasks']
    if not tasks or any(abs(sum(item['weight'] for item in task['criteria'].values()) - 1) > 1e-8 for task in tasks.values()):
        raise ValueError('Each task must have criterion weights summing to one.')

    for task in tasks.values():
        if any(type(item['weight']) not in (int, float) or not math.isfinite(item['weight']) or not 0 <= item['weight'] <= 1 for item in task['criteria'].values()):
            raise ValueError('Rubric weights must be finite values in [0,1].')

        gates = task['gates']
        if not isinstance(gates, list) or not gates or any(not isinstance(gate, str) or gate not in task['criteria'] for gate in gates):
            raise ValueError('Each task needs a nonempty list of named gate criteria.')

        if len(set(gates)) != len(gates):
            raise ValueError('Task gates must be unique.')

        if any(task['criteria'][gate].get('optional') for gate in gates):
            raise ValueError('A task gate cannot be optional.')

    rubric = deepcopy(rubric)
    digest = hashlib.sha256(json.dumps(rubric, sort_keys=True, allow_nan=False).encode()).hexdigest()
    rubric['version'] = f"{rubric['name']}:{digest}"
    return rubric


def score_answers(rubric: dict, selected: list[str], cued: list[str], answers: dict) -> dict:
    if not selected or len(set(selected)) != len(selected) or any(task not in rubric['tasks'] for task in selected):
        raise ValueError('Select known tasks before collecting an attempt.')

    if not cued or len(set(cued)) != len(cued) or not set(cued).issubset(selected) or set(answers) != set(cued):
        raise ValueError('Grade exactly the selected tasks that were actually cued.')

    scores = {}
    for name in cued:
        task = rubric['tasks'][name]
        values = answers[name]
        if set(values) != set(task['criteria']):
            raise ValueError('Provide every rubric answer, including explicit untested tracking.')

        for key, criterion in task['criteria'].items():
            value = values[key]
            allowed = [True, False] if key in task['gates'] else [0, .5, 1]
            valid = type(value) is bool if key in task['gates'] else type(value) in (int, float) and value in allowed
            if value is None and criterion.get('optional'):
                continue
            if not valid:
                raise ValueError(f'Invalid answer for {name}/{key}.')

        scores[name] = sum(criterion['weight'] * (values[key] or 0) for key, criterion in task['criteria'].items()) if all(values[gate] for gate in task['gates']) else 0.
    return {'reward': sum(scores.values()) / len(scores), 'task_scores': scores, 'reward_version': rubric['version']}
