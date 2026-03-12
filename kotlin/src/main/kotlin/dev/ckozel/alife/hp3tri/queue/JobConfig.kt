package dev.ckozel.alife.hp3tri.queue

import dev.ckozel.alife.hp3tri.evolution.EloTournamentConfig
import kotlinx.serialization.Serializable

interface JobRunner {
    var shouldStop: Boolean
    val running: Boolean
    val currentGeneration: Int
    val totalGenerations: Int
    val matchesCompletedThisGen: Int
    fun run()
    fun getLog(): List<String>
    fun bestMetric(): Float
    fun progressMetric(): Float
}

@Serializable
data class JobConfig(
    val name: String,
    val description: String = "",
    val priority: Int = 0,
    val tournament: EloTournamentConfig = EloTournamentConfig(),
    val seedCheckpoint: String? = null,
)

@Serializable
data class RunStatus(
    val jobName: String,
    val state: String,
    val generation: Int,
    val totalGenerations: Int,
    val bestFitness: Float,
    val archiveFillRate: Float,
    val matchesCompleted: Int,
    val startedAt: String,
    val updatedAt: String,
    val error: String? = null,
    val hasReplays: Boolean = false,
    val config: JobConfig? = null,
)

data class ActiveRun(
    val runId: String,
    val jobConfig: JobConfig,
    val runner: JobRunner,
    val startedAt: String,
    val logFile: java.io.File,
)

@Serializable
data class PendingJob(
    val filename: String,
    val config: JobConfig,
)

@Serializable
data class RunSummary(
    val runId: String,
    val status: RunStatus,
)

@Serializable
data class CurrentRunResponse(
    val runId: String,
    val jobName: String,
    val state: String,
    val generation: Int,
    val totalGenerations: Int,
    val bestFitness: Float,
    val archiveFillRate: Float,
    val matchesCompleted: Int,
)
