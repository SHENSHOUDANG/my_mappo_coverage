try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
import numpy as np
from envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self, config):
        # EnvCore 读取合并后的配置（defaults.yaml + 用户yaml）
        # 并构建实际的 UAV Pursuit 环境。
        self.env = EnvCore(config)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # 动作空间使用 [-1, 1] 的归一化控制量：
            # - policy 网络输出 action_norm ∈ [-1, 1]
            # - 环境内由 BaseAgent.step 做物理映射：
            #   velocity(米/秒) = action_norm * role_max_speed
            # 这样可避免高斯策略输出无界动作带来的不稳定问题，
            # 且与 agent_docs/pursuit_role.md 的定义保持一致。
            u_action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        功能:
            将向量化动作传入底层环境并返回标准化后的step结果。
        输入:
            actions (np.ndarray | list): shape=(agent_num, action_dim) 的动作集合。
        输出:
            tuple:
                - obs: np.ndarray, shape=(agent_num, obs_dim)
                - rews: np.ndarray, shape=(agent_num, 1)
                - dones: np.ndarray, shape=(agent_num,)
                - infos: list[dict], 长度=agent_num

        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
        功能:
            重置底层环境并返回初始观测。
        输入:
            无。
        输出:
            np.ndarray: shape=(agent_num, obs_dim)。
        """
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        """
        功能:
            关闭环境资源（当前无显式资源需要释放）。
        输入:
            无。
        输出:
            无。
        """
        pass

    def render(self, mode="rgb_array"):
        """
        功能:
            渲染环境（当前版本未实现渲染逻辑）。
        输入:
            mode (str): 渲染模式标识。
        输出:
            无。
        """
        pass

    def seed(self, seed):
        """
        功能:
            设置底层环境随机种子。
        输入:
            seed (int): 随机种子。
        输出:
            无。
        """
        self.env.seed(seed)
