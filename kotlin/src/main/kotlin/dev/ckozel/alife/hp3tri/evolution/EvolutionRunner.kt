package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.genome.*
import java.util.Random

class EvolutionRunner(
    val bridge: JepBridge,
    val config: EvolutionConfig,
) {
    private val rng = Random(config.seed.toLong())
    private var nextGenomeId = config.populationSize + 1

    @Volatile
    var shouldStop = false

    var currentGeneration = 0
        private set
    var matchesCompletedThisGen = 0
        private set
    private val _log = mutableListOf<String>()
    val historyEntries = mutableListOf<HistoryData>()

    var archive = MapElitesArchive(
        binsX = config.mapElitesBinsX,
        binsY = config.mapElitesBinsY,
        rangeX = Pair(0f, 1f),
        rangeY = Pair(0f, 1f),
    )
        private set

    @Volatile
    var running = false
        private set

    fun run() {
        running = true
        shouldStop = false

        try {
            var population = List(config.populationSize) { i ->
                randomGenome(id = i + 1, rng = rng)
            }

            for (gen in 0 until config.generations) {
                if (shouldStop) {
                    addLog("Evolution stopped by user at generation $gen")
                    break
                }

                currentGeneration = gen
                matchesCompletedThisGen = 0
                val perMatchResults = mutableListOf<List<MatchResult>>()
                val participatingGenomes = mutableMapOf<Int, Genome>()

                for (matchIdx in 0 until config.matchesPerGeneration) {
                    if (shouldStop) break

                    val competitors = if (gen == 0 || archive.allGenomes().isEmpty()) {
                        List(config.genomesPerMatch) { population[rng.nextInt(population.size)] }
                    } else {
                        archive.sampleDiverse(config.genomesPerMatch, rng)
                    }

                    competitors.forEach { participatingGenomes[it.id] = it }

                    val matchConfig = buildMatchConfig(gen, matchIdx)
                    val genomeDicts = competitors.map { it.toDict() }

                    val rawResults = bridge.runEvolutionMatch(matchConfig, genomeDicts)
                    val matchResults = rawResults.map { parseMatchResult(it) }
                    perMatchResults.add(matchResults)
                    matchesCompletedThisGen++
                }

                val scored = computeFitness(
                    genomes = participatingGenomes.values.toList(),
                    perMatchResults = perMatchResults,
                    symbiosisWeight = config.symbiosisWeight,
                )

                for (sg in scored) {
                    archive.add(sg.genome, sg.fitness, sg.behaviorDescriptor)
                }

                population = generateNextPopulation(archive, scored)

                val bestFit = archive.bestFitness()
                val avgFit = if (scored.isNotEmpty()) scored.map { it.fitness }.average().toFloat() else 0f
                val fillRate = archive.fillRate()

                historyEntries.add(HistoryData(gen, bestFit, avgFit, fillRate))

                val msg = "Gen $gen | archive fill: ${"%.1f".format(fillRate * 100)}%" +
                        " | best fitness: ${"%.1f".format(bestFit)}" +
                        " | pop size: ${population.size}" +
                        " | genomes evaluated: ${participatingGenomes.size}"
                addLog(msg)
                println(msg)

                if (gen % 5 == 0 || gen == config.generations - 1) {
                    val checkpointData = CheckpointData(
                        generation = gen,
                        genomes = archive.allGenomes().map { it.first },
                        fitnesses = archive.allGenomes().map { it.second },
                        behaviorDescriptors = scored.map { sg ->
                            BehaviorDescriptor(sg.behaviorDescriptor.first, sg.behaviorDescriptor.second)
                        },
                    )
                    saveCheckpoint("data/checkpoints/gen_$gen.json", checkpointData)
                }
            }

            if (!shouldStop) {
                addLog("Evolution complete. Final archive fill: ${"%.1f".format(archive.fillRate() * 100)}%")
            }
        } finally {
            running = false
        }
    }

    private fun addLog(msg: String) {
        synchronized(_log) {
            _log.add(msg)
            if (_log.size > 500) _log.removeFirst()
        }
    }

    fun getLog(): List<String> = synchronized(_log) { _log.toList() }

    private fun generateNextPopulation(
        archive: MapElitesArchive,
        scored: List<ScoredGenome>,
    ): List<Genome> {
        val eliteCount = (config.populationSize * 0.1).toInt()
        val crossoverCount = (config.populationSize * 0.2).toInt()
        val mutationCount = config.populationSize - eliteCount - crossoverCount

        val result = mutableListOf<Genome>()

        val elites = scored.sortedByDescending { it.fitness }.take(eliteCount)
        result.addAll(elites.map { it.genome })

        repeat(mutationCount) {
            val parent = archive.sample(1, rng).firstOrNull()
                ?: scored.random().genome
            result.add(mutate(parent, rng).copy(id = nextGenomeId++))
        }

        repeat(crossoverCount) {
            val parents = archive.sample(2, rng)
            if (parents.size >= 2) {
                result.add(crossover(parents[0], parents[1], nextGenomeId++, rng))
            } else {
                val parent = archive.sample(1, rng).firstOrNull()
                    ?: scored.random().genome
                result.add(mutate(parent, rng).copy(id = nextGenomeId++))
            }
        }

        return result
    }

    private fun buildMatchConfig(generation: Int, matchIndex: Int): Map<String, Any> = mapOf(
        "width" to config.gridWidth,
        "height" to config.gridHeight,
        "tick_limit" to config.tickLimit,
        "seed" to config.seed + generation * 1000 + matchIndex,
        "food_count" to config.foodCount,
        "food_respawn_rate" to config.foodRespawnRate,
    )

    private fun parseMatchResult(raw: Map<String, Any>): MatchResult = MatchResult(
        genomeId = (raw["genome_id"] as Number).toInt(),
        finalCellCount = (raw["final_cell_count"] as Number).toInt(),
        survived = raw["survived"] as Boolean,
        peakCellCount = (raw["peak_cell_count"] as Number).toInt(),
        finalEnergy = (raw["final_energy"] as Number).toInt(),
        mobility = (raw["mobility"] as Number).toFloat(),
        aggression = (raw["aggression"] as Number).toFloat(),
    )
}

data class HistoryData(
    val generation: Int,
    val bestFitness: Float,
    val avgFitness: Float,
    val fillRate: Float,
)
