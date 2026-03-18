from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from brains.q_brain import (
    QBrain,
    QBrainTrainer,
    REWARD_EAT,
    REWARD_DESTROY,
    REWARD_GROW,
    REWARD_REPRODUCE,
    REWARD_MOVE,
    REWARD_TICK_PENALTY,
    REWARD_TERMINAL_PER_CELL,
    STATE_SIZE,
)

if TYPE_CHECKING:
    from simulator.engine import SimulationEngine


@dataclass
class OrganismSnapshot:
    cells_eaten: int
    cells_destroyed: int
    moves: int
    reproductions: int
    cells_grown: int
    energy: int
    cell_count: int
    alive: int


class RewardTracker:
    def __init__(self, trainer: QBrainTrainer, brain: QBrain):
        self.trainer: QBrainTrainer = trainer
        self.brain: QBrain = brain
        self.snapshots: dict[int, OrganismSnapshot] = {}
        self.total_reward: float = 0.0
        self.transition_count: int = 0

    def snapshot_before(self, engine: SimulationEngine) -> None:
        """Capture organism counters before a tick."""
        self.snapshots.clear()
        for org_id in range(1, engine.next_org_id):
            org = engine.organisms[org_id]
            if int(org.alive) == 0:
                continue
            self.snapshots[org_id] = OrganismSnapshot(
                cells_eaten=int(org.lifetime_cells_eaten),
                cells_destroyed=int(org.lifetime_cells_destroyed),
                moves=int(org.lifetime_moves),
                reproductions=int(org.lifetime_reproductions),
                cells_grown=int(org.lifetime_cells_grown),
                energy=int(org.energy),
                cell_count=int(org.cell_count),
                alive=int(org.alive),
            )

    def process_tick(
        self,
        engine: SimulationEngine,
        is_terminal: bool,
    ) -> None:
        """Compute rewards from counter diffs and store transitions."""
        for org_id, snap in self.snapshots.items():
            if org_id not in self.brain.organism_states:
                continue

            state, move_action, grow_action = self.brain.organism_states[org_id]
            org = engine.organisms[org_id]
            alive_now = int(org.alive)

            d_eaten = int(org.lifetime_cells_eaten) - snap.cells_eaten
            d_destroyed = int(org.lifetime_cells_destroyed) - snap.cells_destroyed
            d_moves = int(org.lifetime_moves) - snap.moves
            d_repro = int(org.lifetime_reproductions) - snap.reproductions
            d_grown = int(org.lifetime_cells_grown) - snap.cells_grown

            reward = (
                REWARD_EAT * d_eaten
                + REWARD_DESTROY * d_destroyed
                + REWARD_GROW * d_grown
                + REWARD_REPRODUCE * d_repro
                + REWARD_MOVE * (1 if d_moves > 0 else 0)
                + REWARD_TICK_PENALTY
            )

            done = alive_now == 0 or is_terminal

            if done and alive_now == 1:
                reward += REWARD_TERMINAL_PER_CELL * int(org.cell_count)

            sensors = engine.build_sensor_inputs(org_id) if alive_now == 1 else None
            if sensors is not None:
                from interfaces.brain import OrganismView
                view = OrganismView(
                    organism_id=org_id,
                    age=int(org.age),
                    energy=int(org.energy),
                    cell_count=int(org.cell_count),
                    total_mass=int(org.total_mass),
                    locomotion_power=int(org.locomotion_power),
                    cells=[],
                    grid_width=engine.width,
                    grid_height=engine.height,
                )
                next_state = self.trainer.encode_state(sensors, view)
            else:
                next_state = np.zeros(STATE_SIZE, dtype=np.float32)

            self.trainer.replay_buffer.add(
                state, move_action, grow_action, reward, next_state, done,
            )
            self.total_reward += reward
            self.transition_count += 1

    def get_avg_reward(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.total_reward / self.transition_count
