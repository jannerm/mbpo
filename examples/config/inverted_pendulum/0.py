params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'InvertedPendulum',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'n_epochs': 80, ## 20k steps
        'epoch_length': 250,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -0.05,
        'max_model_t': None,
        'rollout_schedule': [1, 15, 1, 1],
        'hidden_dim': 200,
        'n_initial_exploration_steps': 500,
    }
}