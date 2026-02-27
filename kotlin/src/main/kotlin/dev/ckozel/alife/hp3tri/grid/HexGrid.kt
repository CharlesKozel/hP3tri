package dev.ckozel.alife.hp3tri.grid

import kotlinx.serialization.Serializable

@Serializable
data class TileState(
    val q: Int,
    val r: Int,
    val terrainType: Int,
    val cellType: Int,
    val organismId: Int,
)

@Serializable
data class GridState(
    val width: Int,
    val height: Int,
    val tiles: List<TileState>,
)

@Serializable
data class OrganismState(
    val id: Int,
    val genomeId: Int,
    val energy: Int,
    val alive: Boolean,
    val cellCount: Int,
)

@Serializable
data class SimulationState(
    val tick: Int,
    val status: String,
    val grid: GridState,
    val organisms: List<OrganismState>,
)

@Serializable
data class CellTypeInfo(
    val id: Int,
    val name: String,
    val color: String,
)
