package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.genome.Genome
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

private val json = Json {
    prettyPrint = true
    encodeDefaults = true
}

@Serializable
data class CheckpointData(
    val generation: Int,
    val genomes: List<Genome>,
    val fitnesses: List<Float>,
    val behaviorDescriptors: List<BehaviorDescriptor>,
)

@Serializable
data class BehaviorDescriptor(
    val mobility: Float,
    val aggression: Float,
)

fun saveCheckpoint(path: String, data: CheckpointData) {
    val file = File(path)
    file.parentFile?.mkdirs()
    file.writeText(json.encodeToString(data))
    println("Checkpoint saved to $path")
}

fun loadCheckpoint(path: String): CheckpointData? {
    val file = File(path)
    if (!file.exists()) return null
    return json.decodeFromString<CheckpointData>(file.readText())
}
