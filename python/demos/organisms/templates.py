"""Body templates for demo organisms.

Each template is a dict mapping (dq, dr) offsets from the seed cell
to CellType int values.
"""
from __future__ import annotations

from typing import TypeAlias

from simulator.cell_types import CellType
from simulator.hex_grid import NEIGHBOR_OFFSETS

Template: TypeAlias = dict[tuple[int, int], int]


def _rotate_hex_60(q: int, r: int) -> tuple[int, int]:
    """Rotate axial hex coord 60 degrees clockwise around origin."""
    return -r, q + r


def make_radial_template(wedge: Template, n_fold: int) -> Template:
    """Generate an n-fold radially symmetric template from a single wedge."""
    result: Template = {}
    for (q, r), ct in wedge.items():
        cq, cr = q, r
        for _ in range(n_fold):
            result[(cq, cr)] = ct
            cq, cr = _rotate_hex_60(cq, cr)
    return result


def _mirror_q(template: Template) -> Template:
    """Mirror a template across the r-axis (bilateral symmetry)."""
    result: Template = dict(template)
    for (q, r), ct in template.items():
        result[(-q, q + r)] = ct
    return result


# ── A. Symmetric Plant (~25 cells, 6-fold radial) ──────────────────

_plant_wedge: Template = {
    (0, 0): int(CellType.PHOTOSYNTHETIC),
    # Ring 1: photosynthetic
    (1, 0): int(CellType.PHOTOSYNTHETIC),
    # Ring 2: photosynthetic
    (2, 0): int(CellType.PHOTOSYNTHETIC),
    (1, 1): int(CellType.PHOTOSYNTHETIC),
    # Ring 3: skin border
    (3, 0): int(CellType.SKIN),
    (2, 1): int(CellType.SKIN),
}

PLANT_TEMPLATE: Template = make_radial_template(_plant_wedge, 6)
# Center cell is duplicated by rotation; that's fine, dict dedupes.


# ── B. Fast Predator (~15 cells, bilateral) ────────────────────────

_predator_half: Template = {
    # Center core
    (0, 0): int(CellType.SOFT_TISSUE),
    # Front mouths
    (1, 0): int(CellType.MOUTH),
    (1, -1): int(CellType.MOUTH),
    # Front eyes (directional, facing east)
    (2, 0): int(CellType.EYE),
    (2, -1): int(CellType.EYE),
    # Side flagella
    (0, -1): int(CellType.FLAGELLA),
    (0, -2): int(CellType.FLAGELLA),
    # Rear core
    (-1, 0): int(CellType.SOFT_TISSUE),
    (-1, 1): int(CellType.SOFT_TISSUE),
    # Spike tips
    (0, -3): int(CellType.SPIKE),
    (-2, 1): int(CellType.SPIKE),
}

PREDATOR_TEMPLATE: Template = _mirror_q(_predator_half)


# ── C. Slow Herbivore/Prey (~20 cells, rounder) ────────────────────

_prey_wedge: Template = {
    (0, 0): int(CellType.PHOTOSYNTHETIC),
    (1, 0): int(CellType.PHOTOSYNTHETIC),
    (0, 1): int(CellType.SKIN),
    (1, -1): int(CellType.SKIN),
    (2, 0): int(CellType.SKIN),
    (1, 1): int(CellType.MOUTH),
}

PREY_TEMPLATE: Template = make_radial_template(_prey_wedge, 6)
# Add flagella at two specific spots (not radial)
PREY_TEMPLATE[(-1, 0)] = int(CellType.FLAGELLA)
PREY_TEMPLATE[(-2, 0)] = int(CellType.FLAGELLA)


# ── D. Spike Warrior (~18 cells, 3-fold radial) ────────────────────

_warrior_wedge: Template = {
    (0, 0): int(CellType.ARMOR),
    (1, 0): int(CellType.ARMOR),
    (2, 0): int(CellType.SPIKE),
    (1, -1): int(CellType.SKIN),
    (0, -1): int(CellType.FLAGELLA),
    (3, 0): int(CellType.SPIKE),
}

WARRIOR_TEMPLATE: Template = make_radial_template(_warrior_wedge, 3)


# ── E. Reproducer (~12 cells, compact) ─────────────────────────────

REPRODUCER_TEMPLATE: Template = {
    (0, 0): int(CellType.PHOTOSYNTHETIC),
    (1, 0): int(CellType.PHOTOSYNTHETIC),
    (0, 1): int(CellType.PHOTOSYNTHETIC),
    (-1, 1): int(CellType.SOFT_TISSUE),
    (1, -1): int(CellType.SOFT_TISSUE),
    (-1, 0): int(CellType.SOFT_TISSUE),
    (0, -1): int(CellType.SOFT_TISSUE),
    (2, 0): int(CellType.SKIN),
    (0, 2): int(CellType.SKIN),
    (-2, 1): int(CellType.SKIN),
    (1, 1): int(CellType.FLAGELLA),
    (-1, -1): int(CellType.FLAGELLA),
}
