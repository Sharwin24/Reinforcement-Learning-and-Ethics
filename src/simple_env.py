import gymnasium
from gymnasium import spaces


class SimpleEnv(gymnasium.Env):
    """
    A deterministic environment to test your code.

    Arguments:
    """

    def __init__(self):
        self.reward = 0
        self.max_reward = 100

        # Required by OpenAI Gym
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        """
        Perform an action within the slot machine environment

        Arguments:
          action - (int) An action to perform

        Returns:
          observation - (int) The new environment state. This is always 0 for
            SlotMachines.
          reward - (float) The reward gained by taking an action.
          terminated - (bool) Whether the environment has been completed and requires
            resetting. This is always True for SlotMachines.
          truncated - (bool) Whether the environment has been completed and requires
            resetting. This is always True for SlotMachines.
          info - (dict) A dictionary of additional return values used for
            debugging purposes.
        """
        self.reward += 1
        done = self.reward >= self.max_reward
        return 0, self.reward, done, done, {}

    def reset(self):
        """
        Resets the environment.
        """
        self.reward = 0
        return 0, {'prob': 1}

    def render(self, mode='human', close=False):
        """
        Render the environment display. This is a no-op.
        """
        pass
    