from __future__ import annotations

import numpy as np
import pytest
import torch

from brains.q_brain import (
    DQNetwork,
    ReplayBuffer,
    QBrainTrainer,
    QTrainerConfig,
    QBrain,
    decode_move_action,
    decode_grow_action,
    STATE_SIZE,
    MOVE_ACTIONS,
    GROW_ACTIONS,
    GROW_CELL_TYPES,
    DEFAULT_REPRODUCE_ENERGY,
    reset_trainer,
)
from interfaces.brain import SensorInputs, OrganismView, BrainOutput
from simulator.cell_types import CellType


class TestDQNetwork:
    def test_output_shapes(self):
        net = DQNetwork(STATE_SIZE, 64)
        x = torch.randn(4, STATE_SIZE)
        move_q, grow_q = net(x)
        assert move_q.shape == (4, MOVE_ACTIONS)
        assert grow_q.shape == (4, GROW_ACTIONS)

    def test_single_input(self):
        net = DQNetwork(STATE_SIZE, 64)
        x = torch.randn(1, STATE_SIZE)
        move_q, grow_q = net(x)
        assert move_q.shape == (1, MOVE_ACTIONS)
        assert grow_q.shape == (1, GROW_ACTIONS)


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        next_state = np.ones(STATE_SIZE, dtype=np.float32)
        buf.add(state, 1, 2, 0.5, next_state, False)
        assert len(buf) == 1

    def test_circular_wrap(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            state = np.full(STATE_SIZE, i, dtype=np.float32)
            buf.add(state, 0, 0, float(i), state, False)
        assert len(buf) == 5
        assert buf.pos == 0

    def test_sample_returns_correct_shapes(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        for _ in range(20):
            buf.add(state, 1, 2, 0.5, state, False)
        states, ma, ga, rewards, ns, dones = buf.sample(8)
        assert states.shape == (8, STATE_SIZE)
        assert ma.shape == (8,)
        assert ga.shape == (8,)
        assert rewards.shape == (8,)
        assert ns.shape == (8, STATE_SIZE)
        assert dones.shape == (8,)


class TestStateEncoding:
    def _make_trainer(self) -> QBrainTrainer:
        return QBrainTrainer(QTrainerConfig())

    def test_state_size(self):
        trainer = self._make_trainer()
        sensors = SensorInputs()
        view = OrganismView(
            organism_id=1, age=10, energy=500,
            cell_count=5, total_mass=10, locomotion_power=10,
            cells=[],
        )
        state = trainer.encode_state(sensors, view)
        assert state.shape == (STATE_SIZE,)
        assert state.dtype == np.float32

    def test_energy_encoding(self):
        trainer = self._make_trainer()
        sensors = SensorInputs(own_energy=0.5)
        view = OrganismView(
            organism_id=1, age=10, energy=500,
            cell_count=10, total_mass=10, locomotion_power=10,
            cells=[],
        )
        state = trainer.encode_state(sensors, view)
        assert state[26] == pytest.approx(0.5)

    def test_cell_count_encoding(self):
        trainer = self._make_trainer()
        sensors = SensorInputs()
        view = OrganismView(
            organism_id=1, age=10, energy=500,
            cell_count=25, total_mass=10, locomotion_power=10,
            cells=[],
        )
        state = trainer.encode_state(sensors, view)
        assert state[27] == pytest.approx(0.5)

    def test_cell_count_clamped(self):
        trainer = self._make_trainer()
        sensors = SensorInputs()
        view = OrganismView(
            organism_id=1, age=10, energy=500,
            cell_count=100, total_mass=10, locomotion_power=10,
            cells=[],
        )
        state = trainer.encode_state(sensors, view)
        assert state[27] == pytest.approx(1.0)


class TestActionDecoding:
    def test_move_stay(self):
        assert decode_move_action(0) == -1

    def test_move_directions(self):
        for i in range(6):
            assert decode_move_action(i + 1) == i

    def test_grow_no_grow(self):
        out = decode_grow_action(0)
        assert out.wants_grow is False

    def test_grow_mouth_dir0(self):
        out = decode_grow_action(1)
        assert out.wants_grow is True
        assert out.grow_cell_type == int(CellType.MOUTH)
        assert out.grow_direction == 0

    def test_grow_flagella_dir3(self):
        out = decode_grow_action(7 + 3)
        assert out.wants_grow is True
        assert out.grow_cell_type == int(CellType.FLAGELLA)
        assert out.grow_direction == 3

    def test_grow_soft_tissue_dir5(self):
        out = decode_grow_action(19 + 5)
        assert out.wants_grow is True
        assert out.grow_cell_type == int(CellType.SOFT_TISSUE)
        assert out.grow_direction == 5

    def test_seed_reproduce(self):
        out = decode_grow_action(25)
        assert out.reproduce_cell_idx == -2
        assert out.reproduce_direction == 0
        assert out.reproduce_energy == DEFAULT_REPRODUCE_ENERGY

    def test_seed_reproduce_dir4(self):
        out = decode_grow_action(29)
        assert out.reproduce_cell_idx == -2
        assert out.reproduce_direction == 4

    def test_all_grow_actions_valid(self):
        for a in range(GROW_ACTIONS):
            out = decode_grow_action(a)
            assert isinstance(out, BrainOutput)


class TestQBrainTrainer:
    def test_epsilon_decay(self):
        cfg = QTrainerConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_matches=100)
        trainer = QBrainTrainer(cfg)
        assert trainer.epsilon == pytest.approx(1.0)
        trainer.total_matches = 50
        assert trainer.epsilon == pytest.approx(0.525)
        trainer.total_matches = 100
        assert trainer.epsilon == pytest.approx(0.05)
        trainer.total_matches = 200
        assert trainer.epsilon == pytest.approx(0.05)

    def test_train_step_empty_buffer(self):
        trainer = QBrainTrainer(QTrainerConfig())
        loss = trainer.train_step(batch_size=32)
        assert loss == 0.0

    def test_train_step_with_data(self):
        trainer = QBrainTrainer(QTrainerConfig(batch_size=8))
        state = np.random.randn(STATE_SIZE).astype(np.float32)
        for _ in range(20):
            trainer.replay_buffer.add(state, 0, 0, 1.0, state, False)
        loss = trainer.train_step(batch_size=8)
        assert loss > 0.0


class TestQBrain:
    def test_evaluate_returns_valid_output(self):
        trainer = QBrainTrainer(QTrainerConfig())
        brain = QBrain(trainer, epsilon=0.0)
        sensors = SensorInputs()
        view = OrganismView(
            organism_id=1, age=10, energy=500,
            cell_count=5, total_mass=10, locomotion_power=10,
            cells=[],
        )
        out = brain.evaluate(view, sensors, {})
        assert isinstance(out, BrainOutput)
        assert out.move_direction >= -1
        assert out.move_direction <= 5
        assert 1 in brain.organism_states

    def test_stores_state_for_reward_tracking(self):
        trainer = QBrainTrainer(QTrainerConfig())
        brain = QBrain(trainer, epsilon=1.0)
        sensors = SensorInputs()
        view = OrganismView(
            organism_id=42, age=10, energy=500,
            cell_count=5, total_mass=10, locomotion_power=10,
            cells=[],
        )
        brain.evaluate(view, sensors, {})
        assert 42 in brain.organism_states
        state, move_a, grow_a = brain.organism_states[42]
        assert state.shape == (STATE_SIZE,)
        assert 0 <= move_a < MOVE_ACTIONS
        assert 0 <= grow_a < GROW_ACTIONS
