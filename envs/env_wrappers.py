"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None
        # 仅在需要录制GIF的阶段抓取终止帧，避免常规训练额外渲染开销。
        self.capture_terminal_frame = False

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            done_flag = False
            if 'bool' in done.__class__.__name__:
                done_flag = bool(done)
            else:
                done_flag = bool(np.all(done))

            if done_flag:
                if self.capture_terminal_frame:
                    terminal_frame = self.envs[i].render(mode="rgb_array")
                    env_infos = infos[i]
                    for agent_info in env_infos:
                        if isinstance(agent_info, dict):
                            agent_info["terminal_frame"] = terminal_frame
                obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs] # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human", env_id=None, **kwargs):
        if mode == "rgb_array":
            if env_id is None:
                return np.array([env.render(mode=mode, **kwargs) for env in self.envs])
            env_idx = int(env_id)
            if env_idx < 0 or env_idx >= self.num_envs:
                raise IndexError(f"env_id out of range: {env_idx}")
            return self.envs[env_idx].render(mode=mode, **kwargs)
        elif mode == "human":
            if env_id is None:
                for env in self.envs:
                    env.render(mode=mode, **kwargs)
            else:
                env_idx = int(env_id)
                if env_idx < 0 or env_idx >= self.num_envs:
                    raise IndexError(f"env_id out of range: {env_idx}")
                self.envs[env_idx].render(mode=mode, **kwargs)
        else:
            raise NotImplementedError
