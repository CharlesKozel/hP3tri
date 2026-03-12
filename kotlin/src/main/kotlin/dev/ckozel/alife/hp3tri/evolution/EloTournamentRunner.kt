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
    val previewTickLimit: Int = 30,
    val previewGridSize: Int = 48,
    val foodCount: Int = 80,
    val foodRespawnRate: Int = 5,
    val matchPopulationSize: Int = 1,
    val kFactor: Float = 32f,
    val eliteRate: Float = 0.1f,
    val crossoverRate: Float = 0.2f,
    val freshRate: Float = 0.1f,
    val seed: Int = 42,
    val saveTopMatchReplays: Int = 5,
    val showcaseInterval: Int = 1,
)

@Serializable
data class EloGenomeEntry(
    val genome: Genome,
    var elo: Float = 1000f,
    var wins: Int = 0,
    var losses: Int = 0,
    var draws: Int = 0,
    var previewCellCount: Int = 0,
)

@Serializable
data class EloHistoryEntry(
    val generation: Int,
    val topElo: Float,
    val avgElo: Float,
    val medianElo: Float,
    val matchesPlayed: Int,
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
                    if (idx2 >= idx1) idx2++ // prevent same index from playing itself, while keeping selection prob the same

                    val matchGenome1 = population[idx1]
                    val matchGenome2 = population[idx2]

                    val matchConfig = buildMatchConfig(gen, matchIdx)
                    val genomeDicts = listOf(matchGenome1.genome.toDict(), matchGenome2.genome.toDict())

                    val results = bridge.runEvolutionMatch(matchConfig, genomeDicts)

                    val cells1 = (results[0]["final_cell_count"] as Number).toInt()
                    val cells2 = (results[1]["final_cell_count"] as Number).toInt()

                    val score1: Float
                    val score2: Float
                    when {
                        cells1 > cells2 -> {
                            score1 = 1f; score2 = 0f
                            matchGenome1.wins++; matchGenome2.losses++
                        }
                        cells2 > cells1 -> {
                            score1 = 0f; score2 = 1f
                            matchGenome2.wins++; matchGenome1.losses++
                        }
                        else -> {
                            score1 = 0.5f; score2 = 0.5f
                            matchGenome1.draws++; matchGenome2.draws++
                        }
                    }

                    updateElo(matchGenome1, matchGenome2, score1, score2)

                    matchesCompletedThisGen++
                }

                val sorted = population.sortedByDescending { it.elo }

                // Save showcase replays using top-ranked genomes
                if (config.saveTopMatchReplays > 0
                    && config.showcaseInterval > 0
                    && gen % config.showcaseInterval == 0
                ) {
                    saveShowcaseReplays(gen, sorted)
                }
                val topElo = sorted.first().elo
                val avgElo = population.map { it.elo }.average().toFloat()
                val medianElo = sorted[sorted.size / 2].elo

                historyEntries.add(EloHistoryEntry(
                    generation = gen,
                    topElo = topElo,
                    avgElo = avgElo,
                    medianElo = medianElo,
                    matchesPlayed = config.matchesPerGeneration,
                ))

                val msg = "Gen $gen | top ELO: ${"%.0f".format(topElo)}" +
                        " | avg: ${"%.0f".format(avgElo)}" +
                        " | median: ${"%.0f".format(medianElo)}" +
                        " | #1: genome ${sorted.first().genome.id} (${sorted.first().wins}W/${sorted.first().losses}L)"
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
        val mutationCount = config.populationSize - eliteCount - crossoverCount - freshCount

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

        repeat(freshCount) {
            result.add(EloGenomeEntry(genome = randomGenome(nextGenomeId++, rng)))
        }

        return result
    }

    private fun runPreviewBatch(entries: List<EloGenomeEntry>) {
        val startTime = System.currentTimeMillis()
        println("  [Previews] Generating ${entries.size} genome previews...")
        for ((i, entry) in entries.withIndex()) {
            if (shouldStop) break
            try {
                val previewConfig = mapOf(
                    "width" to config.gridWidth,
                    "height" to config.gridHeight,
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
                if ((i + 1) % 10 == 0 || i + 1 == entries.size) {
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    println("  [Previews] ${i + 1}/${entries.size} (${String.format("%.1f", elapsed)}s)")
                }
            } catch (e: Exception) {
                addLog("Preview failed for genome ${entry.genome.id}: ${e.message}")
            }
        }
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        println("  [Previews] Done in ${String.format("%.1f", totalTime)}s")
    }

    private fun saveShowcaseReplays(gen: Int, sorted: List<EloGenomeEntry>) {
        if (replayDir == null) return
        val genDir = File(replayDir, "gen_$gen")
        genDir.mkdirs()

        // Pair top genomes: #1v#2, #1v#3, #2v#3, #1v#4, ...
        val topN = sorted.take((config.saveTopMatchReplays + 1).coerceAtMost(sorted.size))
        val pairs = mutableListOf<Pair<EloGenomeEntry, EloGenomeEntry>>()
        for (i in topN.indices) {
            for (j in i + 1 until topN.size) {
                pairs.add(topN[i] to topN[j])
                if (pairs.size >= config.saveTopMatchReplays) break
            }
            if (pairs.size >= config.saveTopMatchReplays) break
        }

        val matchEntries = mutableListOf<ReplayMatchEntry>()

        for ((matchIdx, pair) in pairs.withIndex()) {
            val (entry1, entry2) = pair
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

                val filename = "match_$matchIdx.json"
                val framesJson = jsonCompact.encodeToString(ListSerializer(SimulationState.serializer()), frames)
                File(genDir, filename).writeText(framesJson)

                matchEntries.add(ReplayMatchEntry(
                    matchIndex = matchIdx,
                    filename = filename,
                    genomeIds = listOf(entry1.genome.id, entry2.genome.id),
                    totalTicks = frames.size,
                ))
            } catch (e: Exception) {
                addLog("Replay save failed gen $gen match $matchIdx: ${e.message}")
            }
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

    private fun buildMatchConfig(generation: Int, matchIndex: Int): Map<String, Any> = mapOf(
        "width" to config.gridWidth,
        "height" to config.gridHeight,
        "tick_limit" to config.matchTickLimit,
        "seed" to config.seed + generation * 1000 + matchIndex,
        "food_count" to config.foodCount,
        "food_respawn_rate" to config.foodRespawnRate,
        "population_size" to config.matchPopulationSize,
    )

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
