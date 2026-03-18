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
        """Capture organism counters before a tick using bulk numpy reads."""
        self.snapshots.clear()
        n = engine.next_org_id
        alive_np = engine.organisms.alive.to_numpy()[:n]
        eaten_np = engine.organisms.lifetime_cells_eaten.to_numpy()[:n]
        destroyed_np = engine.organisms.lifetime_cells_destroyed.to_numpy()[:n]
        moves_np = engine.organisms.lifetime_moves.to_numpy()[:n]
        repro_np = engine.organisms.lifetime_reproductions.to_numpy()[:n]
        grown_np = engine.organisms.lifetime_cells_grown.to_numpy()[:n]
        energy_np = engine.organisms.energy.to_numpy()[:n]
        cell_count_np = engine.organisms.cell_count.to_numpy()[:n]

        for org_id in range(1, n):
            if int(alive_np[org_id]) == 0:
                continue
            self.snapshots[org_id] = OrganismSnapshot(
                cells_eaten=int(eaten_np[org_id]),
                cells_destroyed=int(destroyed_np[org_id]),
                moves=int(moves_np[org_id]),
                reproductions=int(repro_np[org_id]),
                cells_grown=int(grown_np[org_id]),
                energy=int(energy_np[org_id]),
                cell_count=int(cell_count_np[org_id]),
                alive=1,
            )

    def process_tick(
        self,
        engine: SimulationEngine,
        is_terminal: bool,
    ) -> None:
        """Compute rewards from counter diffs and store transitions."""
        from interfaces.brain import OrganismView

        # Bulk-read all needed organism fields once
        n = engine.next_org_id
        alive_np = engine.organisms.alive.to_numpy()[:n]
        eaten_np = engine.organisms.lifetime_cells_eaten.to_numpy()[:n]
        destroyed_np = engine.organisms.lifetime_cells_destroyed.to_numpy()[:n]
        moves_np = engine.organisms.lifetime_moves.to_numpy()[:n]
        repro_np = engine.organisms.lifetime_reproductions.to_numpy()[:n]
        grown_np = engine.organisms.lifetime_cells_grown.to_numpy()[:n]
        cell_count_np = engine.organisms.cell_count.to_numpy()[:n]
        energy_np = engine.organisms.energy.to_numpy()[:n]
        age_np = engine.organisms.age.to_numpy()[:n]
        mass_np = engine.organisms.total_mass.to_numpy()[:n]
        loco_np = engine.organisms.locomotion_power.to_numpy()[:n]

        # Bulk-read sensor arrays once for all organisms
        sd_np, s_energy_np, s_age_np, ct_counts_np, max_range = engine.bulk_read_sensor_arrays()

        for org_id, snap in self.snapshots.items():
            if org_id not in self.brain.organism_states:
                continue

            state, move_action, grow_action = self.brain.organism_states[org_id]
            alive_now = int(alive_np[org_id])

            d_eaten = int(eaten_np[org_id]) - snap.cells_eaten
            d_destroyed = int(destroyed_np[org_id]) - snap.cells_destroyed
            d_moves = int(moves_np[org_id]) - snap.moves
            d_repro = int(repro_np[org_id]) - snap.reproductions
            d_grown = int(grown_np[org_id]) - snap.cells_grown

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
                reward += REWARD_TERMINAL_PER_CELL * int(cell_count_np[org_id])

            if alive_now == 1:
                sensors = engine.build_sensor_inputs_from_numpy(
                    org_id, sd_np, s_energy_np, s_age_np, ct_counts_np, max_range,
                )
                view = OrganismView(
                    organism_id=org_id,
                    age=int(age_np[org_id]),
                    energy=int(energy_np[org_id]),
                    cell_count=int(cell_count_np[org_id]),
                    total_mass=int(mass_np[org_id]),
                    locomotion_power=int(loco_np[org_id]),
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
