params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'Hopper',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 2, #20,
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
        # 'target_entropy': -1.5,
        'max_model_t': 1e10,
        # 'max_dev': 0.25, 
        # 'marker': 'early-stop_10rep_stochastic',
        'rollout_length_params': [20, 150, 1, 1], ## epoch, loss, length
        # 'marker': 'dump',
    }
}