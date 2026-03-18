from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from interfaces.brain import (
    BrainProvider,
    BrainOutput,
    OrganismView,
    SensorInputs,
    CH_FOOD,
    CH_FRIENDLY,
    CH_FOREIGN_ALIVE,
    NUM_SECTORS,
)
from simulator.cell_types import CellType

STATE_SIZE: int = 28
MOVE_ACTIONS: int = 7
GROW_ACTIONS: int = 31

ORGANISM_CELL_TYPES: list[int] = [
    int(CellType.SOFT_TISSUE),
    int(CellType.MOUTH),
    int(CellType.FLAGELLA),
    int(CellType.EYE),
]
TYPE_ID_NORM: dict[int, float] = {
    int(CellType.SOFT_TISSUE): 0.0,
    int(CellType.MOUTH): 0.33,
    int(CellType.FLAGELLA): 0.66,
    int(CellType.EYE): 1.0,
}

GROW_CELL_TYPES: list[int] = [
    int(CellType.MOUTH),
    int(CellType.FLAGELLA),
    int(CellType.EYE),
    int(CellType.SOFT_TISSUE),
]

REWARD_EAT: float = 3.0
REWARD_DESTROY: float = 3.0
REWARD_GROW: float = 0.3
REWARD_REPRODUCE: float = 5.0
REWARD_MOVE: float = 0.5
REWARD_TICK_PENALTY: float = -0.01
REWARD_TERMINAL_PER_CELL: float = 0.1

DEFAULT_REPRODUCE_ENERGY: int = 50


class DQNetwork(nn.Module):
    def __init__(self, state_size: int = STATE_SIZE, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden, MOVE_ACTIONS)
        self.grow_head = nn.Linear(hidden, GROW_ACTIONS)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.move_head(features), self.grow_head(features)


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity: int = capacity
        self.size: int = 0
        self.pos: int = 0
        self.states: NDArray[np.float32] = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.move_actions: NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self.grow_actions: NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self.rewards: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self.next_states: NDArray[np.float32] = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.dones: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: NDArray[np.float32],
        move_action: int,
        grow_action: int,
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        idx = self.pos
        self.states[idx] = state
        self.move_actions[idx] = move_action
        self.grow_actions[idx] = grow_action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.move_actions[indices]),
            torch.from_numpy(self.grow_actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class QTrainerConfig:
    replay_capacity: int = 500_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    target_update_interval: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_matches: int = 1000
    hidden_size: int = 64


class QBrainTrainer:
    def __init__(self, config: QTrainerConfig | None = None):
        cfg = config or QTrainerConfig()
        self.config: QTrainerConfig = cfg

        self.policy_net: DQNetwork = DQNetwork(STATE_SIZE, cfg.hidden_size)
        self.target_net: DQNetwork = DQNetwork(STATE_SIZE, cfg.hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer: ReplayBuffer = ReplayBuffer(cfg.replay_capacity)

        self.total_train_steps: int = 0
        self.total_matches: int = 0
        self.recent_rewards: list[float] = []
        self.recent_cell_counts: list[float] = []
        self.last_loss: float = 0.0

    @property
    def epsilon(self) -> float:
        cfg = self.config
        progress = min(self.total_matches / max(cfg.epsilon_decay_matches, 1), 1.0)
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * progress

    def encode_state(
        self,
        sensors: SensorInputs,
        organism: OrganismView,
    ) -> NDArray[np.float32]:
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        for s in range(NUM_SECTORS):
            state[s * 3 + 0] = sensors.sector_data[s, CH_FOOD]
            state[s * 3 + 1] = sensors.sector_data[s, CH_FRIENDLY]
            state[s * 3 + 2] = sensors.sector_data[s, CH_FOREIGN_ALIVE]

        counts: list[tuple[float, float]] = []
        total = max(organism.cell_count, 1)
        for ct in ORGANISM_CELL_TYPES:
            c = int(sensors.own_cell_counts[ct])
            ratio = c / total
            counts.append((ratio, TYPE_ID_NORM[ct]))
        counts.sort(key=lambda x: -x[0])

        for i, (ratio, tid) in enumerate(counts):
            state[18 + i * 2] = ratio
            state[18 + i * 2 + 1] = tid

        state[26] = sensors.own_energy
        state[27] = min(organism.cell_count / 50.0, 1.0)

        return state

    def select_actions(
        self,
        state: NDArray[np.float32],
        epsilon: float | None = None,
    ) -> tuple[int, int]:
        eps = epsilon if epsilon is not None else self.epsilon
        if np.random.random() < eps:
            return (
                np.random.randint(0, MOVE_ACTIONS),
                np.random.randint(0, GROW_ACTIONS),
            )

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0)
            move_q, grow_q = self.policy_net(state_t)
            return int(move_q.argmax(dim=1).item()), int(grow_q.argmax(dim=1).item())

    def train_step(self, batch_size: int | None = None) -> float:
        bs = batch_size or self.config.batch_size
        if len(self.replay_buffer) < bs:
            return 0.0

        states, move_acts, grow_acts, rewards, next_states, dones = self.replay_buffer.sample(bs)

        move_q, grow_q = self.policy_net(states)
        move_q_selected = move_q.gather(1, move_acts.unsqueeze(1)).squeeze(1)
        grow_q_selected = grow_q.gather(1, grow_acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_move_q, next_grow_q = self.target_net(next_states)
            next_move_max = next_move_q.max(dim=1).values
            next_grow_max = next_grow_q.max(dim=1).values
            next_q = (next_move_max + next_grow_max) / 2.0
            target = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss_move = nn.functional.mse_loss(move_q_selected, target)
        loss_grow = nn.functional.mse_loss(grow_q_selected, target)
        loss = loss_move + loss_grow

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_train_steps += 1
        self.last_loss = loss.item()

        if self.total_train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def record_match_stats(self, avg_reward: float, avg_cell_count: float) -> None:
        self.total_matches += 1
        self.recent_rewards.append(avg_reward)
        self.recent_cell_counts.append(avg_cell_count)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        if len(self.recent_cell_counts) > 100:
            self.recent_cell_counts.pop(0)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_matches": self.total_matches,
            "epsilon": round(self.epsilon, 4),
            "replay_size": len(self.replay_buffer),
            "total_train_steps": self.total_train_steps,
            "last_loss": round(self.last_loss, 6),
            "avg_reward_100": round(
                sum(self.recent_rewards) / max(len(self.recent_rewards), 1), 3,
            ),
            "avg_cells_100": round(
                sum(self.recent_cell_counts) / max(len(self.recent_cell_counts), 1), 3,
            ),
        }

    def save(self, path: str) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_train_steps": self.total_train_steps,
            "total_matches": self.total_matches,
            "recent_rewards": self.recent_rewards,
            "recent_cell_counts": self.recent_cell_counts,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_train_steps = checkpoint["total_train_steps"]
        self.total_matches = checkpoint["total_matches"]
        self.recent_rewards = checkpoint.get("recent_rewards", [])
        self.recent_cell_counts = checkpoint.get("recent_cell_counts", [])


def decode_move_action(action: int) -> int:
    """Convert move action index to BrainOutput move_direction. 0=stay (-1), 1-6=dir 0-5."""
    if action == 0:
        return -1
    return action - 1


def decode_grow_action(action: int) -> BrainOutput:
    """Convert grow action index to BrainOutput growth/reproduction fields."""
    if action == 0:
        return BrainOutput(wants_grow=False)

    if action <= 24:
        type_idx = (action - 1) // 6
        direction = (action - 1) % 6
        cell_type = GROW_CELL_TYPES[type_idx]
        return BrainOutput(
            wants_grow=True,
            grow_direction=direction,
            grow_cell_type=cell_type,
        )

    # Actions 25-30: SEED (reproduce) in direction 0-5
    direction = action - 25
    return BrainOutput(
        reproduce_cell_idx=-2,
        reproduce_direction=direction,
        reproduce_energy=DEFAULT_REPRODUCE_ENERGY,
    )


class QBrain(BrainProvider):
    def __init__(self, trainer: QBrainTrainer, epsilon: float | None = None):
        self.trainer: QBrainTrainer = trainer
        self.epsilon_override: float | None = epsilon
        self.organism_states: dict[int, tuple[NDArray[np.float32], int, int]] = {}

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        state = self.trainer.encode_state(sensors, organism)
        eps = self.epsilon_override if self.epsilon_override is not None else self.trainer.epsilon
        move_action, grow_action = self.trainer.select_actions(state, eps)

        self.organism_states[organism.organism_id] = (state, move_action, grow_action)

        move_dir = decode_move_action(move_action)
        grow_out = decode_grow_action(grow_action)

        return BrainOutput(
            move_direction=move_dir,
            wants_grow=grow_out.wants_grow,
            grow_direction=grow_out.grow_direction,
            grow_cell_type=grow_out.grow_cell_type,
            reproduce_cell_idx=grow_out.reproduce_cell_idx,
            reproduce_direction=grow_out.reproduce_direction,
            reproduce_energy=grow_out.reproduce_energy,
        )

    def clear_states(self) -> None:
        self.organism_states.clear()


_global_trainer: QBrainTrainer | None = None


def get_trainer(config: QTrainerConfig | None = None) -> QBrainTrainer:
    global _global_trainer
    if _global_trainer is None:
        _global_trainer = QBrainTrainer(config)
    return _global_trainer


def reset_trainer() -> None:
    global _global_trainer
    _global_trainer = None
