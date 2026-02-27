package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.genome.Genome

data class MatchResult(
    val genomeId: Int,
    val finalCellCount: Int,
    val survived: Boolean,
    val peakCellCount: Int,
    val finalEnergy: Int,
    val mobility: Float,
    val aggression: Float,
)

data class ScoredGenome(
    val genome: Genome,
    val fitness: Float,
    val behaviorDescriptor: Pair<Float, Float>,
)

fun computeFitness(
    genomes: List<Genome>,
    perMatchResults: List<List<MatchResult>>,
    symbiosisWeight: Float,
): List<ScoredGenome> {
    val genomeMap = genomes.associateBy { it.id }
    val genomeFitness = mutableMapOf<Int, MutableList<Float>>()
    val genomeMobility = mutableMapOf<Int, MutableList<Float>>()
    val genomeAggression = mutableMapOf<Int, MutableList<Float>>()
    val genomeSurvived = mutableMapOf<Int, MutableList<Boolean>>()

    for (matchResults in perMatchResults) {
        val survivedInMatch = matchResults.filter { it.survived }
        for (result in matchResults) {
            var competitive = result.finalCellCount.toFloat()

            val symbiosisBonus = if (result.survived) {
                survivedInMatch.filter { it.genomeId != result.genomeId }
                    .sumOf { it.finalCellCount.toDouble() }
                    .toFloat() * symbiosisWeight
            } else 0f

            val score = competitive + symbiosisBonus

            genomeFitness.getOrPut(result.genomeId) { mutableListOf() }.add(score)
            genomeMobility.getOrPut(result.genomeId) { mutableListOf() }.add(result.mobility)
            genomeAggression.getOrPut(result.genomeId) { mutableListOf() }.add(result.aggression)
            genomeSurvived.getOrPut(result.genomeId) { mutableListOf() }.add(result.survived)
        }
    }

    return genomes.mapNotNull { genome ->
        val fitnesses = genomeFitness[genome.id] ?: return@mapNotNull null
        if (fitnesses.isEmpty()) return@mapNotNull null

        val avgFitness = fitnesses.average().toFloat()
        val survivalRate = genomeSurvived[genome.id]?.count { it }?.toFloat()
            ?.div(genomeSurvived[genome.id]!!.size) ?: 0f
        val viability = if (survivalRate > 0.5f) 50f else 0f

        val totalFitness = avgFitness + viability

        val avgMobility = genomeMobility[genome.id]?.average()?.toFloat() ?: 0f
        val avgAggression = genomeAggression[genome.id]?.average()?.toFloat() ?: 0f

        ScoredGenome(
            genome = genome,
            fitness = totalFitness,
            behaviorDescriptor = Pair(avgMobility, avgAggression),
        )
    }
}
