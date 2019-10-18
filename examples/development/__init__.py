"""Provides functions that are utilized by the command line interface.

In particular, the examples are exposed to the command line interface
(defined in `softlearning.scripts.console_scripts`) through the
`get_trainable_class`, `get_variant_spec`, and `get_parser` functions.
"""


def get_trainable_class(*args, **kwargs):
    from .main import ExperimentRunner
    return ExperimentRunner


# def get_variant_spec(command_line_args, *args, **kwargs):
#     from .variants import get_variant_spec
#     variant_spec = get_variant_spec(command_line_args, *args, **kwargs)
#     return variant_spec

def get_params_from_file(filepath, params_name='params'):
	import importlib
	from dotmap import DotMap
	module = importlib.import_module(filepath)
	params = getattr(module, params_name)
	params = DotMap(params)
	return params

def get_variant_spec(command_line_args, *args, **kwargs):
    from .base import get_variant_spec
    import importlib
    params = get_params_from_file(command_line_args.config)
    variant_spec = get_variant_spec(command_line_args, *args, params, **kwargs)
    return variant_spec

def get_parser():
    from examples.utils import get_parser
    parser = get_parser()
    return parser
