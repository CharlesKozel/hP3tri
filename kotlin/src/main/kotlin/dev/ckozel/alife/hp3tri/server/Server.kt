package dev.ckozel.alife.hp3tri.server

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.genome.toDict
import dev.ckozel.alife.hp3tri.queue.CurrentRunResponse
import dev.ckozel.alife.hp3tri.queue.JobConfig
import dev.ckozel.alife.hp3tri.queue.JobScheduler
import dev.ckozel.alife.hp3tri.simulation.Simulation
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.http.content.*
import io.ktor.server.routing.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class ReplayInfo(
    val totalTicks: Int,
    val width: Int,
    val height: Int,
)

@Serializable
data class EvolutionStatusResponse(
    val running: Boolean,
    val generation: Int,
    val totalGenerations: Int,
    val archiveFillRate: Float,
    val bestFitness: Float,
    val matchesCompleted: Int,
    val log: List<String>,
)

@Serializable
data class ArchiveEntryResponse(
    val binX: Int,
    val binY: Int,
    val genomeId: Int,
    val fitness: Float,
    val mobility: Float,
    val aggression: Float,
    val symmetryMode: Int,
)

@Serializable
data class HistoryEntryResponse(
    val generation: Int,
    val bestFitness: Float,
    val avgFitness: Float,
    val fillRate: Float,
)

@Serializable
data class RunMatchRequest(
    val genomeIds: List<Int>,
    val gridWidth: Int = 64,
    val gridHeight: Int = 64,
    val tickLimit: Int = 200,
)

private val defaultConfig: Map<String, Any> = mapOf(
    "width" to 32,
    "height" to 32,
    "tick_limit" to 100,
    "seed" to 42,
)

fun startServer(bridge: JepBridge, port: Int = 8080) {
    val cellTypes = bridge.getCellTypes()
    var simulation = Simulation(bridge, defaultConfig)
    val scheduler = JobScheduler(bridge)

    embeddedServer(Netty, port = port) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = false
                encodeDefaults = true
            })
        }
        install(CORS) {
            anyHost()
            allowHeader(HttpHeaders.ContentType)
            allowMethod(HttpMethod.Get)
            allowMethod(HttpMethod.Post)
            allowMethod(HttpMethod.Delete)
        }

        routing {
            get("/api/cell-types") {
                call.respond(cellTypes)
            }
            get("/api/replay/info") {
                val sim = simulation
                call.respond(ReplayInfo(sim.totalTicks, sim.width, sim.height))
            }
            get("/api/replay") {
                call.respond(simulation.replay)
            }
            get("/api/replay/{tick}") {
                val tick = call.parameters["tick"]?.toIntOrNull()
                if (tick == null) {
                    call.respond(HttpStatusCode.BadRequest, "Invalid tick parameter")
                    return@get
                }
                val frame = simulation.getFrame(tick)
                if (frame == null) {
                    call.respond(HttpStatusCode.NotFound, "Tick $tick not found")
                    return@get
                }
                call.respond(frame)
            }
            post("/api/simulation/reset") {
                simulation = Simulation(bridge, defaultConfig)
                call.respond(ReplayInfo(simulation.totalTicks, simulation.width, simulation.height))
            }

            // Evolution routes — delegate to scheduler's current run
            post("/api/evolution/start") {
                val active = scheduler.currentRun
                if (active != null && active.runner.running) {
                    call.respond(HttpStatusCode.Conflict, "Evolution already running")
                    return@post
                }
                call.respond(HttpStatusCode.BadRequest, "Use /api/queue/submit to start evolution runs")
            }

            post("/api/evolution/stop") {
                val active = scheduler.currentRun
                if (active == null || !active.runner.running) {
                    call.respond(HttpStatusCode.OK, mapOf("status" to "not_running"))
                    return@post
                }
                scheduler.pauseCurrentRun()
                call.respond(HttpStatusCode.OK, mapOf("status" to "stopping"))
            }

            get("/api/evolution/status") {
                val runner = scheduler.currentRun?.runner
                val config = scheduler.currentRun?.jobConfig?.evolution
                val status = EvolutionStatusResponse(
                    running = runner?.running ?: false,
                    generation = runner?.currentGeneration ?: 0,
                    totalGenerations = config?.generations ?: 0,
                    archiveFillRate = runner?.archive?.fillRate() ?: 0f,
                    bestFitness = runner?.archive?.bestFitness() ?: 0f,
                    matchesCompleted = runner?.matchesCompletedThisGen ?: 0,
                    log = runner?.getLog() ?: emptyList(),
                )
                call.respond(status)
            }

            get("/api/evolution/archive") {
                val runner = scheduler.currentRun?.runner
                if (runner == null) {
                    call.respond(emptyList<ArchiveEntryResponse>())
                    return@get
                }
                val entries = runner.archive.allEntries().map { entry ->
                    ArchiveEntryResponse(
                        binX = entry.binX,
                        binY = entry.binY,
                        genomeId = entry.genome.id,
                        fitness = entry.fitness,
                        mobility = entry.genome.movementWillingness,
                        aggression = entry.genome.brainParams.getOrElse(0) { 0f },
                        symmetryMode = entry.genome.symmetryMode,
                    )
                }
                call.respond(entries)
            }

            get("/api/evolution/history") {
                val runner = scheduler.currentRun?.runner
                if (runner == null) {
                    call.respond(emptyList<HistoryEntryResponse>())
                    return@get
                }
                val entries = runner.historyEntries.map { h ->
                    HistoryEntryResponse(
                        generation = h.generation,
                        bestFitness = h.bestFitness,
                        avgFitness = h.avgFitness,
                        fillRate = h.fillRate,
                    )
                }
                call.respond(entries)
            }

            post("/api/evolution/run-match") {
                val request = call.receive<RunMatchRequest>()
                val runner = scheduler.currentRun?.runner
                if (runner == null) {
                    call.respond(HttpStatusCode.BadRequest, "No evolution data available. Start evolution first.")
                    return@post
                }
                if (runner.running) {
                    call.respond(HttpStatusCode.Conflict, "Cannot run sample match while evolution is running. Stop evolution first.")
                    return@post
                }

                val genomes = request.genomeIds.mapNotNull { id ->
                    runner.archive.getGenomeById(id)
                }
                if (genomes.isEmpty()) {
                    call.respond(HttpStatusCode.NotFound, "No matching genomes found in archive")
                    return@post
                }

                val matchConfig = mapOf(
                    "width" to request.gridWidth,
                    "height" to request.gridHeight,
                    "tick_limit" to request.tickLimit,
                    "seed" to 42,
                    "food_count" to 40,
                    "food_respawn_rate" to 3,
                )
                val genomeDicts = genomes.map { it.toDict() }

                val frames = bridge.runVisualizableMatch(matchConfig, genomeDicts)
                call.respond(mapOf("frames" to frames))
            }

            // Queue routes
            get("/api/queue/pending") {
                call.respond(scheduler.listPending())
            }

            post("/api/queue/submit") {
                val config = call.receive<JobConfig>()
                val filename = scheduler.submitJob(config)
                call.respond(mapOf("filename" to filename, "status" to "queued"))
            }

            delete("/api/queue/pending/{filename}") {
                val filename = call.parameters["filename"]
                if (filename == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing filename")
                    return@delete
                }
                val removed = scheduler.removePendingJob(filename)
                if (removed) {
                    call.respond(mapOf("status" to "removed"))
                } else {
                    call.respond(HttpStatusCode.NotFound, "Job not found")
                }
            }

            get("/api/queue/runs") {
                call.respond(scheduler.listRuns())
            }

            get("/api/queue/runs/{runId}/status") {
                val runId = call.parameters["runId"]
                if (runId == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing runId")
                    return@get
                }
                val status = scheduler.getRunStatus(runId)
                if (status == null) {
                    call.respond(HttpStatusCode.NotFound, "Run not found")
                    return@get
                }
                call.respond(status)
            }

            get("/api/queue/runs/{runId}/log") {
                val runId = call.parameters["runId"]
                if (runId == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing runId")
                    return@get
                }
                val log = scheduler.getRunLog(runId)
                if (log == null) {
                    call.respond(HttpStatusCode.NotFound, "Run not found")
                    return@get
                }
                call.respondText(log, ContentType.Text.Plain)
            }

            post("/api/queue/pause") {
                val active = scheduler.currentRun
                if (active == null || !active.runner.running) {
                    call.respond(HttpStatusCode.OK, mapOf("status" to "not_running"))
                    return@post
                }
                scheduler.pauseCurrentRun()
                call.respond(mapOf("status" to "pausing"))
            }

            post("/api/queue/cancel") {
                val active = scheduler.currentRun
                if (active == null || !active.runner.running) {
                    call.respond(HttpStatusCode.OK, mapOf("status" to "not_running"))
                    return@post
                }
                scheduler.cancelCurrentRun()
                call.respond(mapOf("status" to "cancelling"))
            }

            get("/api/queue/current") {
                val active = scheduler.currentRun
                if (active == null || !active.runner.running) {
                    call.respond(HttpStatusCode.NoContent)
                    return@get
                }
                call.respond(CurrentRunResponse(
                    runId = active.runId,
                    jobName = active.jobConfig.name,
                    state = "running",
                    generation = active.runner.currentGeneration,
                    totalGenerations = active.jobConfig.evolution.generations,
                    bestFitness = active.runner.archive.bestFitness(),
                    archiveFillRate = active.runner.archive.fillRate(),
                    matchesCompleted = active.runner.matchesCompletedThisGen,
                ))
            }

            // Replay routes
            get("/api/queue/runs/{runId}/replays") {
                val runId = call.parameters["runId"]
                if (runId == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing runId")
                    return@get
                }
                val gens = scheduler.listReplayGenerations(runId)
                if (gens == null) {
                    call.respond(HttpStatusCode.NotFound, "No replays found")
                    return@get
                }
                call.respond(gens)
            }

            get("/api/queue/runs/{runId}/replays/{gen}") {
                val runId = call.parameters["runId"]
                val gen = call.parameters["gen"]?.toIntOrNull()
                if (runId == null || gen == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing parameters")
                    return@get
                }
                val index = scheduler.getReplayIndex(runId, gen)
                if (index == null) {
                    call.respond(HttpStatusCode.NotFound, "Replay index not found")
                    return@get
                }
                call.respondText(index, ContentType.Application.Json)
            }

            get("/api/queue/runs/{runId}/replays/{gen}/{matchIdx}") {
                val runId = call.parameters["runId"]
                val gen = call.parameters["gen"]?.toIntOrNull()
                val matchIdx = call.parameters["matchIdx"]?.toIntOrNull()
                if (runId == null || gen == null || matchIdx == null) {
                    call.respond(HttpStatusCode.BadRequest, "Missing parameters")
                    return@get
                }
                val matchData = scheduler.getReplayMatch(runId, gen, matchIdx)
                if (matchData == null) {
                    call.respond(HttpStatusCode.NotFound, "Replay match not found")
                    return@get
                }
                call.respondText("{\"frames\":$matchData}", ContentType.Application.Json)
            }

            val webDist = File("web/dist")
            if (webDist.isDirectory) {
                staticFiles("/", webDist) {
                    default("index.html")
                }
            }
        }

        launch(Dispatchers.IO) {
            scheduler.start()
        }
    }.start(wait = true)
}
