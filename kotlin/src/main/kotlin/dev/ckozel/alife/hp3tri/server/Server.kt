package dev.ckozel.alife.hp3tri.server

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.simulation.Simulation
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class ReplayInfo(
    val totalTicks: Int,
    val width: Int,
    val height: Int,
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
        }
    }.start(wait = true)
}
