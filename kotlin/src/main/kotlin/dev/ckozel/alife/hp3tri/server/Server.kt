package dev.ckozel.alife.hp3tri.server

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.evolution.EvolutionConfig
import dev.ckozel.alife.hp3tri.evolution.EvolutionRunner
import dev.ckozel.alife.hp3tri.genome.toDict
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

    var evolutionRunner: EvolutionRunner? = null
    val config = EvolutionConfig()

    embeddedServer(Netty, port = port) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = false
                encodeDefaults = true
            })
        }
        install(CORS) {
            allowHost("localhost:5173")
            allowHost("127.0.0.1:5173")
            allowHeader(HttpHeaders.ContentType)
            allowMethod(HttpMethod.Get)
            allowMethod(HttpMethod.Post)
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

            post("/api/evolution/start") {
                val runner = evolutionRunner
                if (runner != null && runner.running) {
                    call.respond(HttpStatusCode.Conflict, "Evolution already running")
                    return@post
                }
                val newRunner = EvolutionRunner(bridge, config)
                evolutionRunner = newRunner
                launch(Dispatchers.IO) {
                    newRunner.run()
                }
                call.respond(HttpStatusCode.OK, mapOf("status" to "started"))
            }

            post("/api/evolution/stop") {
                val runner = evolutionRunner
                if (runner == null || !runner.running) {
                    call.respond(HttpStatusCode.OK, mapOf("status" to "not_running"))
                    return@post
                }
                runner.shouldStop = true
                call.respond(HttpStatusCode.OK, mapOf("status" to "stopping"))
            }

            get("/api/evolution/status") {
                val runner = evolutionRunner
                val status = EvolutionStatusResponse(
                    running = runner?.running ?: false,
                    generation = runner?.currentGeneration ?: 0,
                    totalGenerations = config.generations,
                    archiveFillRate = runner?.archive?.fillRate() ?: 0f,
                    bestFitness = runner?.archive?.bestFitness() ?: 0f,
                    matchesCompleted = runner?.matchesCompletedThisGen ?: 0,
                    log = runner?.getLog() ?: emptyList(),
                )
                call.respond(status)
            }

            get("/api/evolution/archive") {
                val runner = evolutionRunner
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
                val runner = evolutionRunner
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
                val runner = evolutionRunner
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

            val webDist = File("web/dist")
            if (webDist.isDirectory) {
                staticFiles("/", webDist) {
                    default("index.html")
                }
            }
        }
    }.start(wait = true)
}
