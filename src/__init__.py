from gymnasium.envs.registration import register

from src.slot_machines import SlotMachines
from src.simple_env import SimpleEnv
from src.multi_armed_bandit import MultiArmedBandit
from src.q_learning import QLearning
from src.pyplot import plt, matplotlib

register(
    id='{}-{}'.format('SlotMachines', 'v0'),
    entry_point='src:{}'.format('SlotMachines'),
    max_episode_steps=1,
    nondeterministic=True)

register(
    id='{}-{}'.format('SimpleEnv', 'v0'),
    entry_point='src:{}'.format('SimpleEnv'),
    max_episode_steps=100,
    nondeterministic=False)

register(
    id='FrozenLakeNoSlippery-v0',
    entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False})