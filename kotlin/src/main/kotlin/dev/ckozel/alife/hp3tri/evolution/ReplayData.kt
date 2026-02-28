package dev.ckozel.alife.hp3tri.evolution

import kotlinx.serialization.Serializable

@Serializable
data class ReplayIndex(
    val generation: Int,
    val gridWidth: Int,
    val gridHeight: Int,
    val tickLimit: Int,
    val matches: List<ReplayMatchEntry>,
)

@Serializable
data class ReplayMatchEntry(
    val matchIndex: Int,
    val filename: String,
    val genomeIds: List<Int>,
    val totalTicks: Int,
)
