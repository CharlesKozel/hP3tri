package dev.ckozel.alife.hp3tri.bridge

import dev.ckozel.alife.hp3tri.grid.*
import jep.SharedInterpreter

class JepBridge(pythonSourceDir: String) {
    private val interpreter: SharedInterpreter = SharedInterpreter().apply {
        exec("import sys")
        exec("sys.path.insert(0, '$pythonSourceDir')")
        exec("from simulator.sim_runner import run_simulation")
        exec("from simulator.cell_types import get_cell_type_metadata")
    }

    fun runSimulation(config: Map<String, Any>): List<SimulationState> {
        interpreter.set("_config", config)
        interpreter.exec("_result = run_simulation(dict(_config))")
        @Suppress("UNCHECKED_CAST")
        val result = interpreter.getValue("_result") as List<Map<String, Any>>
        return result.map { convertFrame(it) }
    }

    @Suppress("UNCHECKED_CAST")
    private fun convertFrame(frame: Map<String, Any>): SimulationState {
        val gridMap = frame["grid"] as Map<String, Any>
        val tilesList = gridMap["tiles"] as List<Map<String, Any>>
        val tiles = tilesList.map { t ->
            TileState(
                q = (t["q"] as Number).toInt(),
                r = (t["r"] as Number).toInt(),
                terrainType = (t["terrainType"] as Number).toInt(),
                cellType = (t["cellType"] as Number).toInt(),
                organismId = (t["organismId"] as Number).toInt(),
            )
        }
        val grid = GridState(
            width = (gridMap["width"] as Number).toInt(),
            height = (gridMap["height"] as Number).toInt(),
            tiles = tiles,
        )

        val orgList = frame["organisms"] as List<Map<String, Any>>
        val organisms = orgList.map { o ->
            OrganismState(
                id = (o["id"] as Number).toInt(),
                energy = (o["energy"] as Number).toInt(),
                alive = o["alive"] as Boolean,
                cellCount = (o["cellCount"] as Number).toInt(),
            )
        }

        return SimulationState(
            tick = (frame["tick"] as Number).toInt(),
            status = frame["status"] as String,
            grid = grid,
            organisms = organisms,
        )
    }

    fun getCellTypes(): List<CellTypeInfo> {
        interpreter.exec("_cell_types = get_cell_type_metadata()")
        @Suppress("UNCHECKED_CAST")
        val result = interpreter.getValue("_cell_types") as List<Map<String, Any>>
        return result.map { m ->
            CellTypeInfo(
                id = (m["id"] as Number).toInt(),
                name = m["name"] as String,
                color = m["color"] as String,
            )
        }
    }

    fun close() {
        interpreter.close()
    }
}
