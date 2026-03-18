package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.grid.SimulationState
import dev.ckozel.alife.hp3tri.queue.JobRunner
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.Random

@Serializable
data class QLearningConfig(
    val totalMatches: Int = 5000,
    val gridWidth: Int = 64,
    val gridHeight: Int = 64,
    val matchTickLimit: Int = 200,
    val foodCount: Int = 80,
    val foodRespawnRate: Int = 5,
    val foodRespawnInterval: Int = 20,
    val genomesPerMatch: Int = 4,
    val populationSize: Int = 1,
    val trainingStepsPerMatch: Int = 32,
    val batchSize: Int = 64,
    val replayCapacity: Int = 500_000,
    val saveModelInterval: Int = 100,
    val logInterval: Int = 50,
    val seed: Int = 42,
)

private val SEED_CELL_TYPES = listOf(1, 2, 3, 4)

class QLearningRunner(
    private val bridge: JepBridge,
    private val config: QLearningConfig,
    private val checkpointDir: String = "data/checkpoints",
    private val seedCheckpoint: String? = null,
    private val replayDir: String? = null,
    private val onLog: ((String) -> Unit)? = null,
    private val onGenerationComplete: ((Int) -> Unit)? = null,
) : JobRunner {

    override var shouldStop: Boolean = false
    override var running: Boolean = false
        private set
    override var currentGeneration: Int = 0
        private set
    override val totalGenerations: Int
        get() = (config.totalMatches + config.logInterval - 1) / config.logInterval
    override var matchesCompletedThisGen: Int = 0
        private set

    private val log = mutableListOf<String>()
    private val rng = Random(config.seed.toLong())

    private var bestAvgCells: Float = 0f
    private var latestStats: Map<String, Any> = emptyMap()

    override fun run() {
        running = true
        log("Q-Learning started: ${config.totalMatches} matches, " +
            "${config.genomesPerMatch} genomes/match, " +
            "${config.matchTickLimit} ticks/match")

        if (seedCheckpoint != null) {
            try {
                bridge.loadQLearningModel(seedCheckpoint)
                log("Loaded checkpoint: $seedCheckpoint")
            } catch (e: Exception) {
                log("Failed to load checkpoint: ${e.message}")
            }
        }

        try {
            for (matchIdx in 0 until config.totalMatches) {
                if (shouldStop) break

                val genomes = (0 until config.genomesPerMatch).map { i ->
                    val seedType = SEED_CELL_TYPES[rng.nextInt(SEED_CELL_TYPES.size)]
                    mapOf<String, Any>(
                        "id" to (i + 1),
                        "seed_cell_type" to seedType,
                    )
                }

                val matchConfig = mapOf<String, Any>(
                    "width" to config.gridWidth,
                    "height" to config.gridHeight,
                    "tick_limit" to config.matchTickLimit,
                    "seed" to rng.nextInt(1_000_000),
                    "food_count" to config.foodCount,
                    "food_respawn_rate" to config.foodRespawnRate,
                    "food_respawn_interval" to config.foodRespawnInterval,
                    "population_size" to config.populationSize,
                    "training_steps_per_match" to config.trainingStepsPerMatch,
                    "batch_size" to config.batchSize,
                )

                val result = bridge.runQLearningMatch(matchConfig, genomes)

                @Suppress("UNCHECKED_CAST")
                val trainingStats = result["training_stats"] as? Map<String, Any> ?: emptyMap()
                latestStats = trainingStats

                val avgCells = (trainingStats["avg_cells_100"] as? Number)?.toFloat() ?: 0f
                if (avgCells > bestAvgCells) bestAvgCells = avgCells

                matchesCompletedThisGen++

                if ((matchIdx + 1) % config.logInterval == 0) {
                    val epsilon = (trainingStats["epsilon"] as? Number)?.toFloat() ?: 0f
                    val avgReward = (trainingStats["avg_reward_100"] as? Number)?.toFloat() ?: 0f
                    val replaySize = (trainingStats["replay_size"] as? Number)?.toInt() ?: 0
                    val totalSteps = (trainingStats["total_train_steps"] as? Number)?.toInt() ?: 0
                    val loss = (trainingStats["last_loss"] as? Number)?.toFloat() ?: 0f

                    log("Match ${matchIdx + 1}/${config.totalMatches} | " +
                        "ε=${String.format("%.3f", epsilon)} | " +
                        "avg_reward=${String.format("%.3f", avgReward)} | " +
                        "avg_cells=${String.format("%.1f", avgCells)} | " +
                        "replay=$replaySize | " +
                        "train_steps=$totalSteps | " +
                        "loss=${String.format("%.4f", loss)}")

                    currentGeneration = (matchIdx + 1) / config.logInterval
                    matchesCompletedThisGen = 0
                    onGenerationComplete?.invoke(currentGeneration)
                }

                if (config.saveModelInterval > 0 && (matchIdx + 1) % config.saveModelInterval == 0) {
                    bridge.saveQLearningModel("$checkpointDir/qlearning_${matchIdx + 1}.pt")
                    saveReplayMatch(matchIdx + 1)
                }
            }
        } catch (e: Exception) {
            log("Q-Learning error: ${e.message}")
            e.printStackTrace()
        } finally {
            try {
                val finalPath = "$checkpointDir/qlearning_final.pt"
                bridge.saveQLearningModel(finalPath)
                log("Saved final checkpoint: $finalPath")
            } catch (_: Exception) {}
            try {
                saveReplayMatch(config.totalMatches)
                log("Saved final replay")
            } catch (_: Exception) {}
            running = false
            log("Q-Learning finished. Best avg cells: ${String.format("%.1f", bestAvgCells)}")
        }
    }

    override fun getLog(): List<String> = log.toList()

    override fun bestMetric(): Float = bestAvgCells

    override fun progressMetric(): Float {
        val total = config.totalMatches.toFloat()
        val completed = (currentGeneration * config.logInterval + matchesCompletedThisGen).toFloat()
        return (completed / total).coerceIn(0f, 1f)
    }

    fun getTrainingStats(): Map<String, Any> = latestStats

    private fun saveReplayMatch(matchNumber: Int) {
        if (replayDir == null) return
        try {
            val genomes = SEED_CELL_TYPES.mapIndexed { i, seedType ->
                mapOf<String, Any>("id" to (i + 1), "seed_cell_type" to seedType)
            }
            val matchConfig = mapOf<String, Any>(
                "width" to config.gridWidth,
                "height" to config.gridHeight,
                "tick_limit" to config.matchTickLimit,
                "seed" to rng.nextInt(1_000_000),
                "food_count" to config.foodCount,
                "food_respawn_rate" to config.foodRespawnRate,
                "food_respawn_interval" to config.foodRespawnInterval,
                "population_size" to config.populationSize,
                "snapshot_interval" to 3,
            )
            val frames = bridge.runQLearningVisualizableMatch(matchConfig, genomes)
            val genDir = File(replayDir, "gen_${matchNumber}")
            genDir.mkdirs()

            val jsonCompact = Json { prettyPrint = false; encodeDefaults = true }
            val index = ReplayIndex(
                generation = matchNumber,
                gridWidth = config.gridWidth,
                gridHeight = config.gridHeight,
                tickLimit = config.matchTickLimit,
                matches = listOf(ReplayMatchEntry(
                    matchIndex = 0,
                    filename = "match_0.json",
                    genomeIds = SEED_CELL_TYPES,
                    totalTicks = config.matchTickLimit,
                )),
            )
            File(genDir, "index.json").writeText(jsonCompact.encodeToString(index))
            val framesJson = jsonCompact.encodeToString(ListSerializer(SimulationState.serializer()), frames)
            File(genDir, "match_0.json").writeText(framesJson)
        } catch (e: Exception) {
            log("Failed to save replay at match $matchNumber: ${e.message}")
        }
    }

    private fun log(msg: String) {
        log.add(msg)
        onLog?.invoke(msg)
        println("[QLearning] $msg")
    }
}
