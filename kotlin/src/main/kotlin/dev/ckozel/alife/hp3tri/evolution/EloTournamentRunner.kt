package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.genome.Genome
import dev.ckozel.alife.hp3tri.genome.crossover
import dev.ckozel.alife.hp3tri.genome.mutate
import dev.ckozel.alife.hp3tri.genome.randomGenome
import dev.ckozel.alife.hp3tri.genome.toDict
import dev.ckozel.alife.hp3tri.grid.SimulationState
import dev.ckozel.alife.hp3tri.queue.JobRunner
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.Random

@Serializable
data class EloTournamentConfig(
    val populationSize: Int = 100,
    val generations: Int = 10,
    val matchesPerGeneration: Int = 500,
    val gridWidth: Int = 64,
    val gridHeight: Int = 64,
    val matchTickLimit: Int = 500,
    val previewTickLimit: Int = 100,
    val previewGridSize: Int = 128,
    val foodCount: Int = 80,
    val foodRespawnRate: Int = 5,
    val matchPopulationSize: Int = 1,
    val kFactor: Float = 32f,
    val eliteRate: Float = 0.1f,
    val crossoverRate: Float = 0.1f,
    val freshRate: Float = 0.1f,
    val diversityRate: Float = 0.3f,
    val seed: Int = 42,
    val saveTopMatchReplays: Int = 5,
    val showcaseInterval: Int = 1,
    val cellCountWeight: Float = 0.25f,
    val movementWeight: Float = 0.20f,
    val reproductionWeight: Float = 0.20f,
    val interactionWeight: Float = 0.20f,
    val avgCellsWeight: Float = 0.15f,
    val varyMatchConditions: Boolean = true,
)

@Serializable
data class EloGenomeEntry(
    val genome: Genome,
    var elo: Float = 1000f,
    var wins: Int = 0,
    var losses: Int = 0,
    var draws: Int = 0,
    var previewCellCount: Int = 0,
    var avgMoves: Float = 0f,
    var avgCellsEaten: Float = 0f,
    var avgReproductions: Float = 0f,
    var behaviorX: Float = 0f,
    var behaviorY: Float = 0f,
)

@Serializable
data class EloHistoryEntry(
    val generation: Int,
    val topElo: Float,
    val avgElo: Float,
    val medianElo: Float,
    val matchesPlayed: Int,
    val archiveFillRate: Float = 0f,
)

class EloTournamentRunner(
    val bridge: JepBridge,
    val config: EloTournamentConfig,
    val checkpointDir: String = "data/checkpoints",
    val replayDir: String? = null,
    val onLog: ((String) -> Unit)? = null,
    val onGenerationComplete: ((Int) -> Unit)? = null,
) : JobRunner {
    private val jsonCompact = Json { prettyPrint = false; encodeDefaults = true }
    private val jsonPretty = Json { prettyPrint = true; encodeDefaults = true }
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
    val historyEntries = mutableListOf<EloHistoryEntry>()

    @Volatile
    var population: MutableList<EloGenomeEntry> = mutableListOf()
        private set

    private val previewCache = mutableMapOf<Int, SimulationState>()

    private val diversityArchive = MapElitesArchive(
        binsX = 8,
        binsY = 8,
        rangeX = 0f to 1f,
        rangeY = 0f to 1f,
    )

    @Volatile
    override var running = false
        private set

    override fun run() {
        running = true
        shouldStop = false

        try {
            population = MutableList(config.populationSize) { i ->
                EloGenomeEntry(genome = randomGenome(id = i + 1, rng = rng))
            }

            addLog("Generated ${config.populationSize} random genomes")
            runPreviewBatch(population)
            addLog("Previews complete for initial population")

            for (gen in 0 until config.generations) {
                if (shouldStop) {
                    addLog("Tournament stopped at generation $gen")
                    break
                }

                currentGeneration = gen
                matchesCompletedThisGen = 0

                for (matchIdx in 0 until config.matchesPerGeneration) {
                    if (shouldStop) break

                    val idx1 = rng.nextInt(population.size)
                    var idx2 = rng.nextInt(population.size - 1)
                    if (idx2 >= idx1) idx2++

                    val matchGenome1 = population[idx1]
                    val matchGenome2 = population[idx2]

                    val matchConfig = buildMatchConfig(gen, matchIdx)
                    val genomeDicts = listOf(matchGenome1.genome.toDict(), matchGenome2.genome.toDict())

                    val results = bridge.runEvolutionMatch(matchConfig, genomeDicts)

                    val actualTickLimit = (matchConfig["tick_limit"] as Number).toInt()
                    val compositeScore1 = computeMatchScore(results[0], results[1], actualTickLimit)
                    val compositeScore2 = computeMatchScore(results[1], results[0], actualTickLimit)

                    val score1: Float
                    val score2: Float
                    when {
                        compositeScore1 > compositeScore2 -> {
                            score1 = 1f; score2 = 0f
                            matchGenome1.wins++; matchGenome2.losses++
                        }
                        compositeScore2 > compositeScore1 -> {
                            score1 = 0f; score2 = 1f
                            matchGenome2.wins++; matchGenome1.losses++
                        }
                        else -> {
                            score1 = 0.5f; score2 = 0.5f
                            matchGenome1.draws++; matchGenome2.draws++
                        }
                    }

                    updateElo(matchGenome1, matchGenome2, score1, score2)

                    updateBehaviorStats(matchGenome1, results[0], actualTickLimit)
                    updateBehaviorStats(matchGenome2, results[1], actualTickLimit)

                    diversityArchive.add(matchGenome1.genome, matchGenome1.elo, Pair(matchGenome1.behaviorX, matchGenome1.behaviorY))
                    diversityArchive.add(matchGenome2.genome, matchGenome2.elo, Pair(matchGenome2.behaviorX, matchGenome2.behaviorY))

                    if (config.saveTopMatchReplays > 0
                        && matchIdx < config.saveTopMatchReplays
                        && config.showcaseInterval > 0
                        && gen % config.showcaseInterval == 0
                    ) {
                        saveMatchReplay(gen, matchIdx, matchGenome1, matchGenome2)
                    }

                    matchesCompletedThisGen++
                }

                val sorted = population.sortedByDescending { it.elo }
                val topElo = sorted.first().elo
                val avgElo = population.map { it.elo }.average().toFloat()
                val medianElo = sorted[sorted.size / 2].elo
                val archiveFill = diversityArchive.fillRate()

                historyEntries.add(EloHistoryEntry(
                    generation = gen,
                    topElo = topElo,
                    avgElo = avgElo,
                    medianElo = medianElo,
                    matchesPlayed = config.matchesPerGeneration,
                    archiveFillRate = archiveFill,
                ))

                val top = sorted.first()
                val msg = "Gen $gen | top ELO: ${"%.0f".format(topElo)}" +
                        " | avg: ${"%.0f".format(avgElo)}" +
                        " | median: ${"%.0f".format(medianElo)}" +
                        " | archive: ${"%.0f".format(archiveFill * 64)}/64" +
                        " | #1: genome ${top.genome.id}" +
                        " (${top.wins}W/${top.losses}L" +
                        " moves=${"%.0f".format(top.avgMoves)}" +
                        " eaten=${"%.0f".format(top.avgCellsEaten)})"
                addLog(msg)
                println(msg)

                if (gen < config.generations - 1 && !shouldStop) {
                    population = selectAndReproduce(sorted).toMutableList()

                    val newGenomes = population.filter { !previewCache.containsKey(it.genome.id) }
                    if (newGenomes.isNotEmpty()) {
                        runPreviewBatch(newGenomes)
                        addLog("Previews complete for ${newGenomes.size} new genomes")
                    }
                }

                if (gen % 5 == 0 || gen == config.generations - 1) {
                    saveCheckpoint(gen)
                }

                onGenerationComplete?.invoke(gen)
            }

            if (!shouldStop) {
                addLog("Tournament complete. Top ELO: ${"%.0f".format(population.maxOfOrNull { it.elo } ?: 0f)}")
            }
        } finally {
            running = false
        }
    }

    private fun computeMatchScore(
        result: Map<String, Any>,
        opponentResult: Map<String, Any>,
        tickLimit: Int,
    ): Float {
        val myCells = (result["final_cell_count"] as Number).toFloat()
        val oppCells = (opponentResult["final_cell_count"] as Number).toFloat()
        val cellFraction = myCells / maxOf(1f, myCells + oppCells)

        val moves = (result["total_moves"] as? Number)?.toFloat() ?: 0f
        val moveRate = (moves / tickLimit).coerceAtMost(1f)

        val repro = (result["total_reproductions"] as? Number)?.toFloat() ?: 0f
        val reproRate = (repro / (tickLimit / 50f)).coerceAtMost(1f)

        val eaten = (result["total_cells_eaten"] as? Number)?.toFloat() ?: 0f
        val destroyed = (result["total_cells_destroyed"] as? Number)?.toFloat() ?: 0f
        val interactionRate = ((eaten + destroyed) / (tickLimit / 10f)).coerceAtMost(1f)

        val myAvg = (result["avg_cell_count"] as? Number)?.toFloat() ?: 0f
        val oppAvg = (opponentResult["avg_cell_count"] as? Number)?.toFloat() ?: 0f
        val avgFraction = myAvg / maxOf(1f, myAvg + oppAvg)

        return config.cellCountWeight * cellFraction +
                config.movementWeight * moveRate +
                config.reproductionWeight * reproRate +
                config.interactionWeight * interactionRate +
                config.avgCellsWeight * avgFraction
    }

    private fun updateBehaviorStats(
        entry: EloGenomeEntry,
        result: Map<String, Any>,
        tickLimit: Int,
    ) {
        val alpha = 0.3f
        val moves = (result["total_moves"] as? Number)?.toFloat() ?: 0f
        val eaten = (result["total_cells_eaten"] as? Number)?.toFloat() ?: 0f
        val destroyed = (result["total_cells_destroyed"] as? Number)?.toFloat() ?: 0f
        val repro = (result["total_reproductions"] as? Number)?.toFloat() ?: 0f
        val peakCells = (result["peak_cell_count"] as? Number)?.toFloat() ?: 1f
        val orgCount = (result["organism_count"] as? Number)?.toFloat() ?: 1f

        entry.avgMoves = entry.avgMoves * (1 - alpha) + moves * alpha
        entry.avgCellsEaten = entry.avgCellsEaten * (1 - alpha) + eaten * alpha
        entry.avgReproductions = entry.avgReproductions * (1 - alpha) + repro * alpha

        val interaction = ((eaten + destroyed) / maxOf(1f, peakCells * tickLimit / 100f)).coerceIn(0f, 1f)
        val mobilityRepro = (0.5f * (moves / maxOf(1f, tickLimit.toFloat())) +
                0.5f * (repro / maxOf(1f, orgCount * 2f))).coerceIn(0f, 1f)

        entry.behaviorX = entry.behaviorX * (1 - alpha) + interaction * alpha
        entry.behaviorY = entry.behaviorY * (1 - alpha) + mobilityRepro * alpha
    }

    private fun updateElo(
        entry1: EloGenomeEntry,
        entry2: EloGenomeEntry,
        score1: Float,
        score2: Float,
    ) {
        val k = config.kFactor
        val expected1 = 1f / (1f + Math.pow(10.0, ((entry2.elo - entry1.elo) / 400f).toDouble()).toFloat())
        val expected2 = 1f - expected1
        entry1.elo += k * (score1 - expected1)
        entry2.elo += k * (score2 - expected2)
    }

    private fun selectAndReproduce(sorted: List<EloGenomeEntry>): List<EloGenomeEntry> {
        val eliteCount = (config.populationSize * config.eliteRate).toInt()
        val freshCount = (config.populationSize * config.freshRate).toInt()
        val crossoverCount = (config.populationSize * config.crossoverRate).toInt()
        val diversityCount = (config.populationSize * config.diversityRate).toInt()
        val mutationCount = config.populationSize - eliteCount - crossoverCount - freshCount - diversityCount

        val result = mutableListOf<EloGenomeEntry>()

        val elites = sorted.take(eliteCount)
        result.addAll(elites.map {
            EloGenomeEntry(genome = it.genome, elo = it.elo, previewCellCount = it.previewCellCount)
        })

        fun tournamentSelect(): Genome {
            val candidates = List(3) { sorted[rng.nextInt(sorted.size)] }
            return candidates.maxByOrNull { it.elo }!!.genome
        }

        repeat(mutationCount) {
            val child = mutate(tournamentSelect(), rng).copy(id = nextGenomeId++)
            result.add(EloGenomeEntry(genome = child))
        }

        repeat(crossoverCount) {
            val p1 = tournamentSelect()
            val p2 = tournamentSelect()
            val child = crossover(p1, p2, nextGenomeId++, rng)
            result.add(EloGenomeEntry(genome = child))
        }

        val diverseParents = diversityArchive.sampleDiverse(diversityCount, rng)
        for (parent in diverseParents) {
            val child = mutate(parent, rng).copy(id = nextGenomeId++)
            result.add(EloGenomeEntry(genome = child))
        }
        if (diverseParents.size < diversityCount) {
            repeat(diversityCount - diverseParents.size) {
                result.add(EloGenomeEntry(genome = randomGenome(nextGenomeId++, rng)))
            }
        }

        repeat(freshCount) {
            result.add(EloGenomeEntry(genome = randomGenome(nextGenomeId++, rng)))
        }

        return result
    }

    private fun runPreviewBatch(entries: List<EloGenomeEntry>) {
        for (entry in entries) {
            if (shouldStop) break
            try {
                val previewConfig = mapOf(
                    "width" to config.previewGridSize,
                    "height" to config.previewGridSize,
                    "tick_limit" to config.previewTickLimit,
                    "seed" to config.seed,
                )
                val result = bridge.runGenomePreview(previewConfig, entry.genome.toDict())
                entry.previewCellCount = (result["final_cell_count"] as Number).toInt()

                @Suppress("UNCHECKED_CAST")
                val snapshotRaw = result["final_snapshot"] as? Map<String, Any>
                if (snapshotRaw != null) {
                    val state = bridge.convertPreviewFrame(snapshotRaw)
                    previewCache[entry.genome.id] = state
                }
            } catch (e: Exception) {
                addLog("Preview failed for genome ${entry.genome.id}: ${e.message}")
            }
        }
    }

    private fun saveMatchReplay(gen: Int, matchIdx: Int, entry1: EloGenomeEntry, entry2: EloGenomeEntry) {
        if (replayDir == null) return
        try {
            val matchConfig = mapOf(
                "width" to config.gridWidth,
                "height" to config.gridHeight,
                "tick_limit" to config.matchTickLimit,
                "seed" to config.seed + gen * 10000 + matchIdx,
                "food_count" to config.foodCount,
                "food_respawn_rate" to config.foodRespawnRate,
                "population_size" to config.matchPopulationSize,
            )
            val genomeDicts = listOf(entry1.genome.toDict(), entry2.genome.toDict())
            val frames = bridge.runVisualizableMatch(matchConfig, genomeDicts)

            val genDir = File(replayDir, "gen_$gen")
            genDir.mkdirs()

            val filename = "match_$matchIdx.json"
            val framesJson = jsonCompact.encodeToString(ListSerializer(SimulationState.serializer()), frames)
            File(genDir, filename).writeText(framesJson)

            if (matchIdx == 0) {
                val matchEntries = (0 until config.saveTopMatchReplays).map { i ->
                    ReplayMatchEntry(
                        matchIndex = i,
                        filename = "match_$i.json",
                        genomeIds = listOf(entry1.genome.id, entry2.genome.id),
                        totalTicks = config.matchTickLimit,
                    )
                }
                val index = ReplayIndex(
                    generation = gen,
                    gridWidth = config.gridWidth,
                    gridHeight = config.gridHeight,
                    tickLimit = config.matchTickLimit,
                    matches = matchEntries,
                )
                File(genDir, "index.json").writeText(jsonCompact.encodeToString(ReplayIndex.serializer(), index))
            }
        } catch (e: Exception) {
            addLog("Replay save failed gen $gen match $matchIdx: ${e.message}")
        }
    }

    private fun saveCheckpoint(gen: Int) {
        val sorted = population.sortedByDescending { it.elo }
        val data = EloCheckpointData(
            generation = gen,
            entries = sorted.map { EloCheckpointEntry(it.genome, it.elo, it.wins, it.losses, it.draws) },
        )
        val path = "$checkpointDir/elo_gen_$gen.json"
        val file = File(path)
        file.parentFile?.mkdirs()
        file.writeText(jsonPretty.encodeToString(EloCheckpointData.serializer(), data))
        addLog("Checkpoint saved: $path")
    }

    private fun buildMatchConfig(generation: Int, matchIndex: Int): Map<String, Any> {
        val matchSeed = config.seed + generation * 1000 + matchIndex

        if (!config.varyMatchConditions) {
            return mapOf(
                "width" to config.gridWidth,
                "height" to config.gridHeight,
                "tick_limit" to config.matchTickLimit,
                "seed" to matchSeed,
                "food_count" to config.foodCount,
                "food_respawn_rate" to config.foodRespawnRate,
                "population_size" to config.matchPopulationSize,
            )
        }

        val matchRng = Random(matchSeed.toLong())
        val foodVariation = (config.foodCount * 0.3).toInt()
        val foodCount = (config.foodCount + matchRng.nextInt(foodVariation * 2 + 1) - foodVariation).coerceAtLeast(10)

        val tickVariation = (config.matchTickLimit * 0.2).toInt()
        val tickLimit = (config.matchTickLimit + matchRng.nextInt(tickVariation * 2 + 1) - tickVariation).coerceAtLeast(100)

        val respawnVariation = (config.foodRespawnRate * 0.5).toInt().coerceAtLeast(1)
        val respawnRate = (config.foodRespawnRate + matchRng.nextInt(respawnVariation * 2 + 1) - respawnVariation).coerceAtLeast(1)

        return mapOf(
            "width" to config.gridWidth,
            "height" to config.gridHeight,
            "tick_limit" to tickLimit,
            "seed" to matchSeed,
            "food_count" to foodCount,
            "food_respawn_rate" to respawnRate,
            "population_size" to config.matchPopulationSize,
        )
    }

    private fun addLog(msg: String) {
        synchronized(_log) {
            _log.add(msg)
            if (_log.size > 500) _log.removeFirst()
        }
        onLog?.invoke(msg)
    }

    override fun getLog(): List<String> = synchronized(_log) { _log.toList() }

    override fun bestMetric(): Float = population.maxOfOrNull { it.elo } ?: 0f

    override fun progressMetric(): Float {
        val totalMatches = config.matchesPerGeneration.toFloat()
        return if (totalMatches > 0) matchesCompletedThisGen / totalMatches else 0f
    }

    fun getLeaderboard(): List<EloGenomeEntry> = population.sortedByDescending { it.elo }

    fun getPreview(genomeId: Int): SimulationState? = previewCache[genomeId]

    fun getGenomeById(genomeId: Int): Genome? = population.find { it.genome.id == genomeId }?.genome
}

@Serializable
data class EloCheckpointData(
    val generation: Int,
    val entries: List<EloCheckpointEntry>,
)

@Serializable
data class EloCheckpointEntry(
    val genome: Genome,
    val elo: Float,
    val wins: Int,
    val losses: Int,
    val draws: Int,
)
