package dev.ckozel.alife.hp3tri

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.server.startServer
import java.io.File

fun main() {
    println("hP3tri - Artificial Life Evolution Simulator")

    val pythonDir = File("python").absolutePath
    println("Initializing Jep bridge (Python: $pythonDir)...")
    val bridge = JepBridge(pythonDir)

    println("Starting server on port 8080...")
    try {
        startServer(bridge, port = 8080)
    } finally {
        bridge.close()
    }
}
