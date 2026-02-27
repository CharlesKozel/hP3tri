package dev.ckozel.alife.hp3tri.genome

import java.util.Random
import kotlin.math.sqrt

fun randomGenome(id: Int, rng: Random): Genome {
    val cppnWeights = buildList {
        addAll(xavierWeights(fanIn = 5, count = 5 * 12, rng = rng))
        addAll(xavierWeights(fanIn = 12, count = 12 * 12, rng = rng))
        addAll(xavierWeights(fanIn = 12, count = 12 * 10, rng = rng))
    }

    val cppnActivations = List(24) { rng.nextInt(8) }
    val symmetryMode = rng.nextInt(7)
    val brainParams = List(30) { rng.nextFloat() }
    val reproductionThreshold = 0.3f + rng.nextFloat() * 0.4f
    val offspringEnergy = 0.2f + rng.nextFloat() * 0.3f
    val growthDesire = 0.3f + rng.nextFloat() * 0.5f
    val movementWillingness = 0.2f + rng.nextFloat() * 0.6f

    return Genome(
        id = id,
        cppnWeights = cppnWeights,
        cppnActivations = cppnActivations,
        symmetryMode = symmetryMode,
        brainParams = brainParams,
        reproductionThreshold = reproductionThreshold,
        offspringEnergy = offspringEnergy,
        growthDesire = growthDesire,
        movementWillingness = movementWillingness,
    )
}

private fun xavierWeights(fanIn: Int, count: Int, rng: Random): List<Float> {
    val stddev = sqrt(2.0 / fanIn).toFloat()
    return List(count) { (rng.nextGaussian() * stddev).toFloat() }
}
