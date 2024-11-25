import gymnasium
import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_simple():
    """
    Tests your MultiArmedBandit implementation on an extremely simple
    environment
    """
    from src import MultiArmedBandit

    env = gymnasium.make('SimpleEnv-v0')
    agent = MultiArmedBandit(epsilon=0.2)

    _, rewards = agent.fit(env, steps=10, num_bins=10)
    assert len(rewards) == 10, "Should have one reward per step"
    assert np.all(rewards == np.arange(1, 11)), "Each bin contains its own reward"

    _, rewards = agent.fit(env, steps=20, num_bins=3)
    msg = "Bin computes average rewards"
    assert rewards.shape == (3, ), "num_bins = 3"
    assert np.all(np.isclose(rewards[:2], np.array([4, 11]))), msg
    assert np.isclose(rewards[2], 15) or np.isclose(rewards[2], 17.5), msg

    _, rewards = agent.fit(env, steps=1000, num_bins=10)
    assert rewards.shape == (10, ), "num_bins = 10"
    assert np.all(np.isclose(rewards, 50.5)), msg


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_slots():
    """
    Tests that the MultiArmedBandit implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    env = gymnasium.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=10000, num_bins=100)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    _, rewards = agent.fit(env, steps=1000, num_bins=42)
    assert len(rewards) == 42
    _, rewards = agent.fit(env, steps=777, num_bins=100)
    assert len(rewards) == 100

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1
    assert len(rewards) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_random_argmax():
    """
    When choosing to exploit the best action, do not use np.argmax: it will
    deterministically break ties by choosing the action with the lowest index.
    Instead, please *randomly choose* one of those tied-for-the-largest values.
    """

    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    n_machines = 10
    env = gymnasium.make('SlotMachines-v0', n_machines=n_machines,
                   mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values = np.zeros([1, n_machines])

    actions = []
    for _ in range(1000):
        _, a, _ = agent.predict(env, state_action_values)
        actions.append(a[0])

    msg = "Should eventually try all slot machines"
    assert np.unique(actions).shape[0] == n_machines, msg


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_frozen_lake():
    """
    Tests the MultiArmedBandit implementation on the FrozenLake-v1 environment.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    env = gymnasium.make('FrozenLake-v1')
    env.reset()

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (16, 4)
    msg = "Rewards should have 100 elements regardless of the number of steps"
    assert len(rewards) == 100, msg


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_deterministic():
    """
    Tests that the MultiArmedBandit implementation successfully navigates a
    deterministic environment with provided state-action-values.
    """
    from src import MultiArmedBandit

    from src.random import rng
    rng.seed()

    env = gymnasium.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])
    state_action_values = means.reshape(1, 10)

    agent = MultiArmedBandit(epsilon=0.2)
    actions = []
    for _ in range(100):
        _, action, _ = agent.predict(env, state_action_values)
        actions.append(action)

    msg = "With known means, should always pick best action"
    actions = np.array(actions).reshape(-1)
    assert np.all(actions == np.argmax(means)), msg
