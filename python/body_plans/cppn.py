from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from interfaces.body_plan import BodyPlanProvider, GrowthRequest
from interfaces.brain import OrganismView
from simulator.cell_types import CellType, NUM_ORGANISM_CELL_TYPES

CPPN_INPUTS = 5      # distance, angle, dev_time, neighbor_count, bias
CPPN_HIDDEN1 = 12
CPPN_HIDDEN2 = 12
CPPN_OUTPUTS = 1 + NUM_ORGANISM_CELL_TYPES  # cell_exists + cell_type logits
CPPN_NUM_HIDDEN = CPPN_HIDDEN1 + CPPN_HIDDEN2
CPPN_TOTAL_WEIGHTS = (
    CPPN_INPUTS * CPPN_HIDDEN1
    + CPPN_HIDDEN1 * CPPN_HIDDEN2
    + CPPN_HIDDEN2 * CPPN_OUTPUTS
)

NUM_ACTIVATIONS = 8
ACTIVATION_NAMES = ['sin', 'cos', 'gaussian', 'sigmoid', 'tanh', 'relu', 'step', 'abs']

SYMMETRY_ASYMMETRIC = 0
SYMMETRY_BILATERAL = 1
SYMMETRY_RADIAL = 2  # value >= 2 means radial-N where N = value

TemplateMap: TypeAlias = dict[tuple[int, int], int]


def _apply_activations(z: NDArray[np.float32], act_ids: NDArray[np.int32]) -> NDArray[np.float32]:
    out = np.empty_like(z)
    for i in range(z.shape[-1]):
        a = act_ids[i] % NUM_ACTIVATIONS
        v = z[..., i]
        if a == 0:
            out[..., i] = np.sin(v)
        elif a == 1:
            out[..., i] = np.cos(v)
        elif a == 2:
            out[..., i] = np.exp(-v * v)
        elif a == 3:
            out[..., i] = 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))
        elif a == 4:
            out[..., i] = np.tanh(v)
        elif a == 5:
            out[..., i] = np.maximum(v, 0)
        elif a == 6:
            out[..., i] = np.where(v > 0, 1.0, 0.0)
        else:
            out[..., i] = np.abs(v)
    return out


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


ORGANISM_CELL_TYPES = [ct for ct in CellType if ct not in (CellType.NULL, CellType.FOOD)]


class CppnBodyPlan(BodyPlanProvider):
    def __init__(
        self,
        weights: NDArray[np.float32],
        activations: NDArray[np.int32],
        symmetry_mode: int = SYMMETRY_BILATERAL,
        max_radius: int = 12,
    ) -> None:
        self.max_radius = max_radius
        self.symmetry_mode = symmetry_mode

        idx = 0
        w1_size = CPPN_INPUTS * CPPN_HIDDEN1
        self.w1 = weights[idx:idx + w1_size].reshape(CPPN_INPUTS, CPPN_HIDDEN1)
        idx += w1_size

        w2_size = CPPN_HIDDEN1 * CPPN_HIDDEN2
        self.w2 = weights[idx:idx + w2_size].reshape(CPPN_HIDDEN1, CPPN_HIDDEN2)
        idx += w2_size

        w3_size = CPPN_HIDDEN2 * CPPN_OUTPUTS
        self.w3 = weights[idx:idx + w3_size].reshape(CPPN_HIDDEN2, CPPN_OUTPUTS)

        self.act1 = activations[:CPPN_HIDDEN1]
        self.act2 = activations[CPPN_HIDDEN1:CPPN_NUM_HIDDEN]

    def _transform_coords(
        self,
        q: NDArray[np.float32],
        r: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.symmetry_mode == SYMMETRY_BILATERAL:
            return np.abs(q), r
        elif self.symmetry_mode >= SYMMETRY_RADIAL:
            n = self.symmetry_mode
            d = np.sqrt(q * q + r * r)
            a = np.arctan2(r, q)
            a = np.mod(a, 2.0 * np.pi / n)
            return d * np.cos(a), d * np.sin(a)
        return q, r

    def evaluate_batch(
        self,
        positions: NDArray[np.float32],
        dev_time: float,
        neighbor_counts: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Evaluate CPPN for a batch of positions.
        Returns (cell_exists_prob, cell_type_indices).
        """
        q_raw = positions[:, 0]
        r_raw = positions[:, 1]
        q_t, r_t = self._transform_coords(q_raw, r_raw)

        dist = np.sqrt(q_t * q_t + r_t * r_t) / self.max_radius
        angle = np.arctan2(r_t, q_t) / np.pi
        t_norm = np.full_like(dist, min(dev_time / 500.0, 1.0))
        n_norm = neighbor_counts / 6.0
        bias = np.ones_like(dist)

        inputs = np.stack([dist, angle, t_norm, n_norm, bias], axis=-1)

        h1 = _apply_activations(inputs @ self.w1, self.act1)
        h2 = _apply_activations(h1 @ self.w2, self.act2)
        out = h2 @ self.w3

        cell_exists = 1.0 / (1.0 + np.exp(-np.clip(out[:, 0], -10, 10)))
        type_probs = _softmax(out[:, 1:])
        cell_type_indices = np.argmax(type_probs, axis=-1)

        return cell_exists, cell_type_indices

    def generate_template(self, dev_time: float = 0.0) -> TemplateMap:
        """Generate full body template centered at (0,0)."""
        positions: list[tuple[int, int]] = []
        for dq in range(-self.max_radius, self.max_radius + 1):
            for dr in range(-self.max_radius, self.max_radius + 1):
                if abs(dq + dr) <= self.max_radius:
                    positions.append((dq, dr))

        if not positions:
            return {}

        pos_arr = np.array(positions, dtype=np.float32)
        n_counts = np.full(len(positions), 3.0, dtype=np.float32)
        exists, type_idx = self.evaluate_batch(pos_arr, dev_time, n_counts)

        template: TemplateMap = {}
        for i, (dq, dr) in enumerate(positions):
            if dq == 0 and dr == 0:
                template[(0, 0)] = int(ORGANISM_CELL_TYPES[type_idx[i]])
                continue
            if exists[i] > 0.5:
                template[(dq, dr)] = int(ORGANISM_CELL_TYPES[type_idx[i]])

        if (0, 0) not in template:
            template[(0, 0)] = int(CellType.SOFT_TISSUE)

        return template

    def query_growth(
        self,
        organism: OrganismView,
        border_empty_neighbors: list[tuple[int, int]],
        genome_data: dict,
    ) -> list[GrowthRequest]:
        template: TemplateMap | None = genome_data.get("body_template")
        if template is None:
            return []

        origin_q = genome_data.get("origin_q", 0)
        origin_r = genome_data.get("origin_r", 0)

        requests: list[GrowthRequest] = []
        for q, r in border_empty_neighbors:
            rel_q = q - origin_q
            rel_r = r - origin_r
            ct = template.get((rel_q, rel_r))
            if ct is not None:
                dist = abs(rel_q) + abs(rel_r) + abs(rel_q + rel_r)
                priority = 1.0 / (1.0 + dist * 0.1)
                requests.append(GrowthRequest(q=q, r=r, cell_type=ct, priority=priority))
        return requests
