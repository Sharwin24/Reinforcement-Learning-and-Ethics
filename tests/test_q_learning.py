import gymnasium
import pytest
import numpy as np


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_simple():
    """
    Tests your QLearning implementation on an extremely simple
    environment
    """
    from src import QLearning

    env = gymnasium.make('SimpleEnv-v0')
    agent = QLearning()

    _, rewards = agent.fit(env, steps=10, num_bins=10)
    assert len(rewards) == 10, "Should have one reward per step"
    assert np.all(rewards == np.arange(1, 11)), "Each bin contains its own reward"

    _, rewards = agent.fit(env, steps=20, num_bins=3)
    assert rewards.shape == (3, ), "num_bins = 3"
    msg = "Bin computes average rewards"
    assert np.all(np.isclose(rewards[:2], np.array([4, 11]))), msg
    assert np.isclose(rewards[2], 15) or np.isclose(rewards[2], 17.5), msg

    _, rewards = agent.fit(env, steps=1000, num_bins=10)
    assert rewards.shape == (10, ), "num_bins = 10"
    assert np.all(np.isclose(rewards, 50.5)), msg


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_slots():
    """
    Tests that the Qlearning implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    env = gymnasium.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(1, 5))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = QLearning(epsilon=0.2, gamma=0)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1 and states[0] == 0
    assert len(rewards) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_frozen_lake():
    """
    Tests that the QLearning implementation successfully learns the
    FrozenLake-v1 environment.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    env = gymnasium.make('FrozenLake-v1')
    env.reset()

    agent = QLearning(epsilon=0.4, gamma=0.9, alpha=0.5)
    state_action_values, rewards = agent.fit(env, steps=10000)

    state_values = np.max(state_action_values, axis=1)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100

    assert np.allclose(state_values[np.array([5, 7, 11, 12, 15])], np.zeros(5))
    assert np.all(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])] > 0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_random_argmax():
    """
    When choosing to exploit the best action, do not use np.argmax: it will
    deterministically break ties by choosing the action with the lowest index.
    Instead, please *randomly choose* one of those tied-for-the-largest values.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    n_machines = 10
    env = gymnasium.make('SlotMachines-v0', n_machines=n_machines,
                   mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)

    agent = QLearning()
    state_action_values = np.zeros([1, n_machines])

    actions = []
    for _ in range(1000):
        _, a, _ = agent.predict(env, state_action_values)
        actions.append(a[0])

    msg = "Should eventually try all slot machines"
    assert np.unique(actions).shape[0] == n_machines, msg


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_deterministic():
    """
    Tests that the QLearning implementation successfully navigates a
    deterministic environment with provided state-action-values.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    env = gymnasium.make('FrozenLake-v1', map_name="4x4", is_slippery=False)

    agent = QLearning(epsilon=0.5)
    state_action_values = np.array([
        [0.0, 0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.51, 0.49, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    msg = " ".join([
        "With fixed state action values dictionary,",
        "should navigate the lake in the exactly specified way."
    ])
    states, actions, rewards = agent.predict(env, state_action_values)
    assert np.all(states == np.array([4, 8, 9, 10, 14, 15])), msg
    assert np.all(actions == np.array([1, 1, 2, 2, 1, 2])), msg
    assert np.all(rewards == np.array([0, 0, 0, 0, 0, 1])), msg
