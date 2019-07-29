import sys

from .ant import AntEnv
from .humanoid import HumanoidEnv

env_overwrite = {'Ant': AntEnv, 'Humanoid': HumanoidEnv}

sys.modules[__name__] = env_overwrite