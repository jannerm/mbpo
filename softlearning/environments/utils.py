from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

import pdb

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}


def get_environment(universe, domain, task, environment_params):
    env = ADAPTERS[universe](domain, task, **environment_params)
    return env


def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, environment_kwargs)
