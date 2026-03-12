package dev.ckozel.alife.hp3tri.bridge

import dev.ckozel.alife.hp3tri.grid.*
import jep.SharedInterpreter
import java.util.concurrent.Callable
import java.util.concurrent.Executors

class JepBridge(pythonSourceDir: String) {
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "jep-thread").apply { isDaemon = true }
    }

    private val interpreter: SharedInterpreter = runOnJepThread {
        println("  [Jep] Creating Python interpreter...")
        SharedInterpreter().apply {
            exec("import sys")
            exec("sys.stdout.reconfigure(line_buffering=True)")
            exec("sys.path.insert(0, '${pythonSourceDir.replace("\\", "/")}')")
            println("  [Jep] Importing simulator (triggers Taichi init)...")
            exec("from simulator.sim_runner import run_simulation")
            println("  [Jep] Importing cell types...")
            exec("from simulator.cell_types import get_cell_type_metadata")
            println("  [Jep] Importing match runner...")
            exec("from evolution.match_runner import run_evolution_match, run_visualizable_match, run_genome_preview")
            println("  [Jep] Python bridge ready.")
        }
    }

    private fun <T> runOnJepThread(block: () -> T): T =
        executor.submit(Callable { block() }).get()

    fun runSimulation(config: Map<String, Any>): List<SimulationState> = runOnJepThread {
        interpreter.set("_config", config)
        interpreter.exec("_result = run_simulation(dict(_config))")
        @Suppress("UNCHECKED_CAST")
        val result = interpreter.getValue("_result") as List<Map<String, Any>>
        result.map { convertFrame(it) }
    }

    @Suppress("UNCHECKED_CAST")
    fun convertPreviewFrame(frame: Map<String, Any>): SimulationState = convertFrame(frame)

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
                genomeId = (o["genomeId"] as Number).toInt(),
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

    fun getCellTypes(): List<CellTypeInfo> = runOnJepThread {
        interpreter.exec("_cell_types = get_cell_type_metadata()")
        @Suppress("UNCHECKED_CAST")
        val result = interpreter.getValue("_cell_types") as List<Map<String, Any>>
        result.map { m ->
            CellTypeInfo(
                id = (m["id"] as Number).toInt(),
                name = m["name"] as String,
                color = m["color"] as String,
            )
        }
    }

    fun runEvolutionMatch(config: Map<String, Any>, genomes: List<Map<String, Any>>): List<Map<String, Any>> = runOnJepThread {
        interpreter.set("_match_config", config)
        interpreter.set("_genomes", genomes)
        interpreter.exec("_match_result = run_evolution_match(dict(_match_config), [dict(g) for g in _genomes])")
        @Suppress("UNCHECKED_CAST")
        interpreter.getValue("_match_result") as List<Map<String, Any>>
    }

    fun runVisualizableMatch(config: Map<String, Any>, genomes: List<Map<String, Any>>): List<SimulationState> = runOnJepThread {
        interpreter.set("_vis_config", config)
        interpreter.set("_vis_genomes", genomes)
        interpreter.exec("_vis_result = run_visualizable_match(dict(_vis_config), [dict(g) for g in _vis_genomes])")
        @Suppress("UNCHECKED_CAST")
        val result = interpreter.getValue("_vis_result") as List<Map<String, Any>>
        result.map { convertFrame(it) }
    }

    fun runGenomePreview(config: Map<String, Any>, genome: Map<String, Any>): Map<String, Any> = runOnJepThread {
        interpreter.set("_preview_config", config)
        interpreter.set("_preview_genome", genome)
        interpreter.exec("_preview_result = run_genome_preview(dict(_preview_config), dict(_preview_genome))")
        @Suppress("UNCHECKED_CAST")
        interpreter.getValue("_preview_result") as Map<String, Any>
    }

    fun close() {
        runOnJepThread { interpreter.close() }
        executor.shutdown()
    }
}
