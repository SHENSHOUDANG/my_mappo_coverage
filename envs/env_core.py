"""
兼容层:
- 保持现有 `from envs.env_core import EnvCore` 不变
- 真实环境实现放在 `envs/env_uav_pursuit.py`
"""

from envs.env_uav_pursuit import UAVPursuitEnv


class EnvCore(UAVPursuitEnv):
    pass
