from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
}

#### aws3
## standard
## cheetah : rollout 1, rep 20, pool 4e6, entropy 0.5
## walker : ""

## entropy 0.35
## cheetah : entropy 0.35
## walker : ""

#### aws2 
## 1-10 rollouts
## cheetah : rollout 1-5, rep 20, pool 8e6, entropy 0.5, linear 1-5 (epochs 100-200) 
## walker : ""

## continued
## walker : rep 20, entropy 0.5
## walker : rep 30, entropy 0.5

## TODO
## 10 rep
## cheetah : rollout 1, rep 10, pool 8e6, entropy 0.5 
## walker : ""

## gcp
## 1 :
## 2 :
## 3 :
## 4 : 10 rep walker

## baselines
## 1 : rollout length (1/5/full)
## 2 : no ensembles
## 3 : no model rollouts : aws.p2.8x.2/3 10/20/30


## run cheetah, walker longer
## rollout length on hopper (full)
## no ensemble on walker
## no ensemble on cheetah

## aws2
## long cheetah : entropy 0.5, rep 20, pool 4e6
## cheetah : ensemble, 1000 freq, time 100
## cheetah : no ensemble, 1000 freq, time 100
## cheetah : ensemble, 250 freq, time 50
## cheetah : ensemble, 1000 freq, time 100, 10 rep 

## aws3
## old cheetah (entropy 0.5, rollout 1, rep 20, pool 4e6)
## old walker (...)
## long walker : ensemble, 1000 freq, time 100 (rep 20, pool 2e6)
## ablation hopper : 200-step rollouts

## gcp3
##
##
##
##

## gcp4
## old walker (entropy 0.35, rep 10)
## old cheetah (entropy 0.35, rep 10)
## free
## free

## cheetah : rep 10, 1000 freq, time 100
 # long cheetah 
## long cheetah
## long walker

## gcp4
## walker : no ensemble
##

## aws1
## no model : 30
## no model : 10

ALGORITHM_PARAMS_BASE = {
    'type': 'MBPO',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 10, #20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        ####
        'model_reset_freq': 1000,
        'model_train_freq': 250, # 250
        # 'retain_model_epochs': 2,
        'model_pool_size': 2e6,
        'rollout_batch': 100e3, # 40e3
        'rollout_length': 1,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'entropy_mult': 0.5,
        'max_model_t': 1e10,
        # 'max_dev': 0.25, 
        # 'marker': 'early-stop_10rep_stochastic',
        'rollout_length_params': [20, 150, 1, 1], ## epoch, loss, length
        'marker': 'dump',
    }
}

####
## WARNING : MVE-SAC
####
# ALGORITHM_PARAMS_BASE = {
#     'type': 'MVE',

#     'kwargs': {
#         'epoch_length': 1000,
#         'train_every_n_steps': 1,
#         'n_train_repeat': 1, #20,
#         'eval_render_mode': None,
#         'eval_n_episodes': 1,
#         'eval_deterministic': True,

#         'discount': 0.99,
#         'tau': 5e-3,
#         'reward_scale': 1.0,
#         ####
#         # 'model_reset_freq': 1000,
#         'model_train_freq': 250, # 250
#         # 'retain_model_epochs': 2,
#         # 'model_pool_size': 2e6,
#         # 'rollout_batch': 100e3, # 40e3
#         # 'rollout_length': 5,
#         'deterministic': False,
#         'num_networks': 7,
#         'num_elites': 5,
#         # 'real_ratio': 0.05,
#         'entropy_mult': 1.0,
#         'max_model_t': 1e10,
#         'H': 5,
#         # 'max_dev': 0.25, 
#         # 'marker': 'early-stop_10rep_stochastic',
#         # 'rollout_length_params': [2000, 15000, 5, 25], ## epoch, loss, length
#         'marker': 'dump',
#     }
# }


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    },
    'MVE': {
        'type': 'MVE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
    'MBPO': {
        'type': 'MBPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e3),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'Pendulum': 10,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v3': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    }
}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': (
                    variant_spec
                    ['training']
                    ['environment_params']
                    ['image_shape']),
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy, args.algorithm)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
