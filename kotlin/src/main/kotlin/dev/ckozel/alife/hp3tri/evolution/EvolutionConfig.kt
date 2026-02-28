package dev.ckozel.alife.hp3tri.evolution

import kotlinx.serialization.Serializable

@Serializable
data class EvolutionConfig(
    val populationSize: Int = 100,
    val generations: Int = 50,
    val matchesPerGeneration: Int = 200,
    val genomesPerMatch: Int = 3,
    val gridWidth: Int = 64,
    val gridHeight: Int = 64,
    val tickLimit: Int = 500,
    val foodCount: Int = 80,
    val foodRespawnRate: Int = 5,
    val symbiosisWeight: Float = 0.3f,
    val mapElitesBinsX: Int = 8,
    val mapElitesBinsY: Int = 8,
    val mutationSigma: Float = 0.1f,
    val crossoverRate: Float = 0.2f,
    val seed: Int = 42,
    val sampleMatchWidth: Int = 64,
    val sampleMatchHeight: Int = 64,
    val sampleMatchTickLimit: Int = 200,
    val showcaseInterval: Int = 5,
)
