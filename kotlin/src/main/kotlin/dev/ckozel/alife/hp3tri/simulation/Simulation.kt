package dev.ckozel.alife.hp3tri.simulation

import dev.ckozel.alife.hp3tri.bridge.JepBridge
import dev.ckozel.alife.hp3tri.grid.SimulationState

class Simulation(bridge: JepBridge, config: Map<String, Any>) {
    val replay: List<SimulationState> = bridge.runSimulation(config)
    val totalTicks: Int = replay.size - 1
    val width: Int = replay.firstOrNull()?.grid?.width ?: 0
    val height: Int = replay.firstOrNull()?.grid?.height ?: 0

    fun getFrame(tick: Int): SimulationState? {
        if (tick < 0 || tick >= replay.size) return null
        return replay[tick]
    }
}
