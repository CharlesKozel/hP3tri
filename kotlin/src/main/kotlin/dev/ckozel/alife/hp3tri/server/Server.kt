package dev.ckozel.alife.hp3tri.server

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.evolution.EloTournamentRunner
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
data class EloLeaderboardEntry(
    val genomeId: Int,
    val elo: Float,
    val wins: Int,
    val losses: Int,
    val draws: Int,
    val previewCellCount: Int,
    val symmetryMode: Int,
)

@Serializable
data class EloHistoryResponse(
    val generation: Int,
    val topElo: Float,
    val avgElo: Float,
    val medianElo: Float,
    val matchesPlayed: Int,
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

            // Evolution status — works for tournament runner
            post("/api/evolution/start") {
                call.respond(HttpStatusCode.BadRequest, "Use /api/queue/submit to start tournament runs")
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
                val status = EvolutionStatusResponse(
                    running = runner?.running ?: false,
                    generation = runner?.currentGeneration ?: 0,
                    totalGenerations = runner?.totalGenerations ?: 0,
                    archiveFillRate = runner?.progressMetric() ?: 0f,
                    bestFitness = runner?.bestMetric() ?: 0f,
                    matchesCompleted = runner?.matchesCompletedThisGen ?: 0,
                    log = runner?.getLog() ?: emptyList(),
                )
                call.respond(status)
            }

            // Tournament leaderboard
            get("/api/tournament/leaderboard") {
                val runner = scheduler.currentRun?.runner
                if (runner == null || runner !is EloTournamentRunner) {
                    call.respond(emptyList<EloLeaderboardEntry>())
                    return@get
                }
                val entries = runner.getLeaderboard().map { entry ->
                    EloLeaderboardEntry(
                        genomeId = entry.genome.id,
                        elo = entry.elo,
                        wins = entry.wins,
                        losses = entry.losses,
                        draws = entry.draws,
                        previewCellCount = entry.previewCellCount,
                        symmetryMode = entry.genome.symmetryMode,
                    )
                }
                call.respond(entries)
            }

            // Tournament ELO history
            get("/api/tournament/history") {
                val runner = scheduler.currentRun?.runner
                if (runner == null || runner !is EloTournamentRunner) {
                    call.respond(emptyList<EloHistoryResponse>())
                    return@get
                }
                val entries = runner.historyEntries.map { h ->
                    EloHistoryResponse(
                        generation = h.generation,
                        topElo = h.topElo,
                        avgElo = h.avgElo,
                        medianElo = h.medianElo,
                        matchesPlayed = h.matchesPlayed,
                    )
                }
                call.respond(entries)
            }

            // Preview snapshot for a genome in the current tournament
            get("/api/tournament/preview/{genomeId}") {
                val genomeId = call.parameters["genomeId"]?.toIntOrNull()
                if (genomeId == null) {
                    call.respond(HttpStatusCode.BadRequest, "Invalid genomeId")
                    return@get
                }
                val runner = scheduler.currentRun?.runner
                if (runner == null || runner !is EloTournamentRunner) {
                    call.respond(HttpStatusCode.NotFound, "No tournament running")
                    return@get
                }
                val preview = runner.getPreview(genomeId)
                if (preview == null) {
                    call.respond(HttpStatusCode.NotFound, "No preview for genome $genomeId")
                    return@get
                }
                call.respond(preview)
            }

            // Run a 1v1 match between two genomes from the tournament
            post("/api/tournament/run-match") {
                val request = call.receive<RunMatchRequest>()
                val runner = scheduler.currentRun?.runner
                if (runner == null || runner !is EloTournamentRunner) {
                    call.respond(HttpStatusCode.BadRequest, "No tournament data available")
                    return@post
                }
                if (runner.running) {
                    call.respond(HttpStatusCode.Conflict, "Cannot run sample match while tournament is running")
                    return@post
                }

                val genomes = request.genomeIds.mapNotNull { id ->
                    runner.getGenomeById(id)
                }
                if (genomes.isEmpty()) {
                    call.respond(HttpStatusCode.NotFound, "No matching genomes found")
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
                    totalGenerations = active.runner.totalGenerations,
                    bestFitness = active.runner.bestMetric(),
                    archiveFillRate = active.runner.progressMetric(),
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
