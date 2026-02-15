package dev.ckozel.alife.hp3tri.server

import dev.ckozel.alife.hp3tri.grid.createDemoGrid
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.json.Json

fun startServer(port: Int = 8080) {
    val demoGrid = createDemoGrid()

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
        }
        routing {
            get("/api/grid") {
                call.respond(demoGrid)
            }
        }
    }.start(wait = true)
}
