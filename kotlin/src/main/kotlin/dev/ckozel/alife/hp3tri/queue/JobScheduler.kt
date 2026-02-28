package dev.ckozel.alife.hp3tri.queue

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.evolution.CheckpointData
import dev.ckozel.alife.hp3tri.evolution.EvolutionRunner
import dev.ckozel.alife.hp3tri.evolution.loadCheckpoint
import kotlinx.coroutines.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.time.Instant
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

private val json = Json {
    prettyPrint = true
    encodeDefaults = true
    ignoreUnknownKeys = true
}

private val FILENAME_DATE_FMT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")

class JobScheduler(private val bridge: JepBridge) {
    val pendingDir = File("data/queue/pending")
    val runsDir = File("data/queue/runs")
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var runJob: Job? = null

    @Volatile
    var currentRun: ActiveRun? = null
        private set

    init {
        pendingDir.mkdirs()
        runsDir.mkdirs()
    }

    suspend fun start() {
        while (true) {
            delay(5000)
            try {
                tick()
            } catch (e: Exception) {
                println("JobScheduler tick error: ${e.message}")
            }
        }
    }

    private fun tick() {
        val active = currentRun
        if (active != null) {
            if (!active.runner.running) {
                finishRun(active)
                currentRun = null
            } else {
                val higherPriority = findHighestPriorityPending()
                if (higherPriority != null && higherPriority.second.priority > active.jobConfig.priority) {
                    pauseCurrentRun()
                }
                return
            }
        }

        val next = findHighestPriorityPending() ?: return
        startJob(next.first, next.second)
    }

    fun submitJob(config: JobConfig): String {
        val timestamp = LocalDateTime.now().format(FILENAME_DATE_FMT)
        val safeName = config.name.replace(Regex("[^a-zA-Z0-9_-]"), "_")
        val filename = "${timestamp}_${safeName}.json"
        val file = File(pendingDir, filename)
        file.writeText(json.encodeToString(config))
        return filename
    }

    fun pauseCurrentRun() {
        val active = currentRun ?: return
        active.runner.shouldStop = true
        // The polling loop will detect runner stopped and call finishRun with paused state
    }

    fun cancelCurrentRun() {
        val active = currentRun ?: return
        active.runner.shouldStop = true
        val runDir = File(runsDir, active.runId)
        writeStatus(runDir, buildStatus(active, "failed"))
    }

    fun listPending(): List<PendingJob> {
        val files = pendingDir.listFiles { f -> f.extension == "json" } ?: return emptyList()
        return files.mapNotNull { file ->
            try {
                val config = json.decodeFromString<JobConfig>(file.readText())
                PendingJob(file.name, config)
            } catch (_: Exception) {
                null
            }
        }.sortedWith(compareByDescending<PendingJob> { it.config.priority }.thenBy { it.filename })
    }

    fun listRuns(): List<RunSummary> {
        val dirs = runsDir.listFiles { f -> f.isDirectory } ?: return emptyList()
        return dirs.mapNotNull { dir ->
            val statusFile = File(dir, "status.json")
            if (!statusFile.exists()) return@mapNotNull null
            try {
                val status = json.decodeFromString<RunStatus>(statusFile.readText())
                RunSummary(dir.name, status)
            } catch (_: Exception) {
                null
            }
        }.sortedByDescending { it.runId }
    }

    fun getRunStatus(runId: String): RunStatus? {
        val statusFile = File(runsDir, "$runId/status.json")
        if (!statusFile.exists()) return null
        return try {
            json.decodeFromString<RunStatus>(statusFile.readText())
        } catch (_: Exception) {
            null
        }
    }

    fun getRunLog(runId: String): String? {
        val logFile = File(runsDir, "$runId/log.txt")
        if (!logFile.exists()) return null
        return logFile.readText()
    }

    fun removePendingJob(filename: String): Boolean {
        val file = File(pendingDir, filename)
        return file.exists() && file.delete()
    }

    private fun findHighestPriorityPending(): Pair<File, JobConfig>? {
        val files = pendingDir.listFiles { f -> f.extension == "json" } ?: return null
        return files.mapNotNull { file ->
            try {
                val config = json.decodeFromString<JobConfig>(file.readText())
                file to config
            } catch (_: Exception) {
                null
            }
        }.sortedWith(compareByDescending<Pair<File, JobConfig>> { it.second.priority }.thenBy { it.first.name })
            .firstOrNull()
    }

    private fun startJob(pendingFile: File, config: JobConfig) {
        val timestamp = LocalDateTime.now().format(FILENAME_DATE_FMT)
        val safeName = config.name.replace(Regex("[^a-zA-Z0-9_-]"), "_")
        val runId = "${timestamp}_${safeName}"
        val runDir = File(runsDir, runId)
        val checkpointDir = File(runDir, "checkpoints")
        checkpointDir.mkdirs()

        val jobFile = File(runDir, "job.json")
        jobFile.writeText(json.encodeToString(config))

        val logFile = File(runDir, "log.txt")
        logFile.createNewFile()

        val checkpoint: CheckpointData? = config.seedCheckpoint?.let { loadCheckpoint(it) }

        val replayDir = File(runDir, "replays")

        val startedAt = Instant.now().toString()
        val runner = EvolutionRunner(
            bridge = bridge,
            config = config.evolution,
            checkpointDir = checkpointDir.path,
            replayDir = replayDir.path,
            initialCheckpoint = checkpoint,
            onLog = { msg ->
                try {
                    logFile.appendText("$msg\n")
                } catch (_: Exception) {}
            },
            onGenerationComplete = { gen ->
                try {
                    val active = currentRun ?: return@EvolutionRunner
                    writeStatus(runDir, buildStatus(active, "running"))
                } catch (_: Exception) {}
            },
        )

        val active = ActiveRun(
            runId = runId,
            jobConfig = config,
            runner = runner,
            startedAt = startedAt,
            logFile = logFile,
        )
        currentRun = active

        writeStatus(runDir, buildStatus(active, "running"))

        pendingFile.delete()

        runJob = scope.launch {
            runner.run()
        }
    }

    private fun finishRun(active: ActiveRun) {
        val runDir = File(runsDir, active.runId)
        val wasCancelled = runDir.resolve("status.json").let { f ->
            if (f.exists()) {
                try {
                    json.decodeFromString<RunStatus>(f.readText()).state == "failed"
                } catch (_: Exception) { false }
            } else false
        }

        if (wasCancelled) return

        val wasPaused = active.runner.shouldStop &&
                active.runner.currentGeneration < active.jobConfig.evolution.generations - 1

        if (wasPaused) {
            val latestCheckpoint = findLatestCheckpoint(File(runDir, "checkpoints"))
            writeStatus(runDir, buildStatus(active, "paused"))

            val resumeConfig = active.jobConfig.copy(
                seedCheckpoint = latestCheckpoint,
                priority = active.jobConfig.priority - 1,
            )
            submitJob(resumeConfig)
        } else {
            writeStatus(runDir, buildStatus(active, "completed"))
        }
    }

    private fun findLatestCheckpoint(checkpointDir: File): String? {
        val files = checkpointDir.listFiles { f -> f.extension == "json" } ?: return null
        return files.maxByOrNull { it.name }?.path
    }

    fun listReplayGenerations(runId: String): List<Int>? {
        val replaysDir = File(runsDir, "$runId/replays")
        if (!replaysDir.isDirectory) return null
        return replaysDir.listFiles { f -> f.isDirectory && f.name.startsWith("gen_") }
            ?.mapNotNull { it.name.removePrefix("gen_").toIntOrNull() }
            ?.sorted()
            ?: emptyList()
    }

    fun getReplayIndex(runId: String, gen: Int): String? {
        val indexFile = File(runsDir, "$runId/replays/gen_$gen/index.json")
        if (!indexFile.exists()) return null
        return indexFile.readText()
    }

    fun getReplayMatch(runId: String, gen: Int, matchIdx: Int): String? {
        val matchFile = File(runsDir, "$runId/replays/gen_$gen/match_$matchIdx.json")
        if (!matchFile.exists()) return null
        return matchFile.readText()
    }

    private fun buildStatus(active: ActiveRun, state: String): RunStatus {
        val replaysDir = File(runsDir, "${active.runId}/replays")
        return RunStatus(
            jobName = active.jobConfig.name,
            state = state,
            generation = active.runner.currentGeneration,
            totalGenerations = active.jobConfig.evolution.generations,
            bestFitness = active.runner.archive.bestFitness(),
            archiveFillRate = active.runner.archive.fillRate(),
            matchesCompleted = active.runner.matchesCompletedThisGen,
            startedAt = active.startedAt,
            updatedAt = Instant.now().toString(),
            hasReplays = replaysDir.isDirectory && (replaysDir.list()?.isNotEmpty() == true),
        )
    }

    private fun writeStatus(runDir: File, status: RunStatus) {
        runDir.mkdirs()
        File(runDir, "status.json").writeText(json.encodeToString(status))
    }
}
