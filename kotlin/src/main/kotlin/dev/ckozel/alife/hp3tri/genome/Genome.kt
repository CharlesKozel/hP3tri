package dev.ckozel.alife.hp3tri.genome

import kotlinx.serialization.Serializable

@Serializable
data class Genome(
    val id: Int,
    val cppnWeights: List<Float>,
    val cppnActivations: List<Int>,
    val symmetryMode: Int,
    val brainParams: List<Float>,
    val reproductionThreshold: Float,
    val offspringEnergy: Float,
    val growthDesire: Float,
    val movementWillingness: Float,
    val seedCellType: Int = 1,
)

val VALID_SEED_CELL_TYPES: List<Int> = listOf(1, 2, 3, 4, 5, 7, 8, 9)

fun Genome.toDict(): Map<String, Any> = mapOf(
    "id" to id,
    "cppn_weights" to cppnWeights.toFloatArray(),
    "cppn_activations" to cppnActivations.toIntArray(),
    "symmetry_mode" to symmetryMode,
    "brain_params" to brainParams.toFloatArray(),
    "reproduction_threshold" to reproductionThreshold,
    "offspring_energy" to offspringEnergy,
    "growth_desire" to growthDesire,
    "movement_willingness" to movementWillingness,
    "seed_cell_type" to seedCellType,
)
