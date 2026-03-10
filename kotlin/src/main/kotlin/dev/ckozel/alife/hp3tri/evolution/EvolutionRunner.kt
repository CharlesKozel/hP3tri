package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.genome.Genome
import dev.ckozel.alife.hp3tri.genome.crossover
import dev.ckozel.alife.hp3tri.genome.mutate
import dev.ckozel.alife.hp3tri.genome.randomGenome
import dev.ckozel.alife.hp3tri.genome.toDict
import dev.ckozel.alife.hp3tri.grid.SimulationState
import dev.ckozel.alife.hp3tri.queue.JobRunner
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.Random

class EvolutionRunner(
    val bridge: JepBridge,
    val config: EvolutionConfig,
    val checkpointDir: String = "data/checkpoints",
    val replayDir: String? = null,
    val initialCheckpoint: CheckpointData? = null,
    val onLog: ((String) -> Unit)? = null,
    val onGenerationComplete: ((Int) -> Unit)? = null,
) : JobRunner {
    private val replayJson = Json { prettyPrint = false; encodeDefaults = true }
    private val rng = Random(config.seed.toLong())
    private var nextGenomeId = config.populationSize + 1

    @Volatile
    override var shouldStop = false

    override var currentGeneration = 0
        private set
    override var matchesCompletedThisGen = 0
        private set
    override val totalGenerations: Int get() = config.generations
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
    override var running = false
        private set

    override fun run() {
        running = true
        shouldStop = false

        try {
            var population: List<Genome>
            var startGen = 0

            if (initialCheckpoint != null) {
                population = initialCheckpoint.genomes
                startGen = initialCheckpoint.generation + 1
                nextGenomeId = population.maxOf { it.id } + 1
                for (i in population.indices) {
                    val bd = initialCheckpoint.behaviorDescriptors.getOrNull(i)
                    val fitness = initialCheckpoint.fitnesses.getOrNull(i) ?: 0f
                    val bdPair = if (bd != null) Pair(bd.mobility, bd.aggression) else Pair(0.5f, 0.5f)
                    archive.add(population[i], fitness, bdPair)
                }
                addLog("Resumed from checkpoint at generation ${initialCheckpoint.generation}")
            } else {
                population = List(config.populationSize) { i ->
                    randomGenome(id = i + 1, rng = rng)
                }
            }

            for (gen in startGen until config.generations) {
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

                    // Always sample from the full population to evaluate new mutants
                    val competitors = List(config.genomesPerMatch) {
                        population[rng.nextInt(population.size)]
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
                    totalGridArea = config.gridWidth * config.gridHeight,
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
                    saveCheckpoint("$checkpointDir/gen_$gen.json", checkpointData)
                }

                if (config.showcaseInterval > 0 && gen % config.showcaseInterval == 0) {
                    saveShowcaseReplays(gen)
                }

                onGenerationComplete?.invoke(gen)
            }

            if (!shouldStop) {
                addLog("Evolution complete. Final archive fill: ${"%.1f".format(archive.fillRate() * 100)}%")
            }
        } finally {
            running = false
        }
    }

    private fun saveShowcaseReplays(gen: Int) {
        if (replayDir == null) return
        val allGenomes = archive.allGenomes().map { it.first }
        if (allGenomes.isEmpty()) return

        val batches = allGenomes.chunked(config.genomesPerMatch)
        val genDir = File(replayDir, "gen_$gen")
        genDir.mkdirs()

        val matchEntries = mutableListOf<ReplayMatchEntry>()
        val matchConfig = mapOf(
            "width" to config.sampleMatchWidth,
            "height" to config.sampleMatchHeight,
            "tick_limit" to config.sampleMatchTickLimit,
            "seed" to config.seed + gen * 10000,
            "food_count" to config.foodCount,
            "food_respawn_rate" to config.foodRespawnRate,
        )

        for ((i, batch) in batches.withIndex()) {
            if (shouldStop) break
            try {
                val genomeDicts = batch.map { it.toDict() }
                val frames: List<SimulationState> = bridge.runVisualizableMatch(matchConfig, genomeDicts)

                val filename = "match_$i.json"
                val framesJson = replayJson.encodeToString(ListSerializer(SimulationState.serializer()), frames)
                File(genDir, filename).writeText(framesJson)

                matchEntries.add(ReplayMatchEntry(
                    matchIndex = i,
                    filename = filename,
                    genomeIds = batch.map { it.id },
                    totalTicks = config.sampleMatchTickLimit,
                ))
            } catch (e: Exception) {
                addLog("Warning: showcase match $i failed: ${e.message}")
            }
        }

        val index = ReplayIndex(
            generation = gen,
            gridWidth = config.sampleMatchWidth,
            gridHeight = config.sampleMatchHeight,
            tickLimit = config.sampleMatchTickLimit,
            matches = matchEntries,
        )
        File(genDir, "index.json").writeText(replayJson.encodeToString(ReplayIndex.serializer(), index))
        addLog("Saved ${matchEntries.size} showcase replays for gen $gen")
    }

    private fun addLog(msg: String) {
        synchronized(_log) {
            _log.add(msg)
            if (_log.size > 500) _log.removeFirst()
        }
        onLog?.invoke(msg)
    }

    override fun getLog(): List<String> = synchronized(_log) { _log.toList() }

    override fun bestMetric(): Float = archive.bestFitness()

    override fun progressMetric(): Float = archive.fillRate()

    private fun generateNextPopulation(
        archive: MapElitesArchive,
        scored: List<ScoredGenome>,
    ): List<Genome> {
        val eliteCount = (config.populationSize * 0.1).toInt()
        val freshCount = (config.populationSize * 0.1).toInt()
        val crossoverCount = (config.populationSize * 0.2).toInt()
        val mutationCount = config.populationSize - eliteCount - crossoverCount - freshCount

        val result = mutableListOf<Genome>()

        // Elites: top performers pass through unchanged
        val elites = scored.sortedByDescending { it.fitness }.take(eliteCount)
        result.addAll(elites.map { it.genome })

        // Combine archive + scored genomes as parent pool for diversity
        val parentPool = (archive.allGenomes().map { it.first } + scored.map { it.genome })
            .distinctBy { it.id }

        // Tournament selection helper: pick best of 3 random candidates
        fun tournamentSelect(): Genome {
            if (parentPool.size <= 3) return parentPool[rng.nextInt(parentPool.size)]
            val candidates = List(3) { parentPool[rng.nextInt(parentPool.size)] }
            val scoredMap = scored.associateBy { it.genome.id }
            return candidates.maxByOrNull { scoredMap[it.id]?.fitness ?: 0f } ?: candidates[0]
        }

        repeat(mutationCount) {
            result.add(mutate(tournamentSelect(), rng).copy(id = nextGenomeId++))
        }

        repeat(crossoverCount) {
            val p1 = tournamentSelect()
            val p2 = tournamentSelect()
            result.add(crossover(p1, p2, nextGenomeId++, rng))
        }

        // Fresh random genomes to maintain exploration
        repeat(freshCount) {
            result.add(randomGenome(nextGenomeId++, rng))
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
