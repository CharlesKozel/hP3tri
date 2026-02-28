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

    // Brain params with viability-biased initialization.
    // Indices: 0-7 core thresholds, 8-13 wander weights,
    // 14-21 cell type growth weights, 22-29 metabolic tweaks.
    val brainParams = MutableList(30) { rng.nextFloat() }
    // P_GROWTH_THRESHOLD (idx 6): low so organisms grow early
    brainParams[6] = 0.02f + rng.nextFloat() * 0.13f
    // P_HUNGER_THRESHOLD (idx 2): moderate so organisms seek food
    brainParams[2] = 0.3f + rng.nextFloat() * 0.4f
    // P_REPRODUCE_THRESHOLD (idx 4): high so organisms grow large before reproducing
    brainParams[4] = 0.7f + rng.nextFloat() * 0.25f
    // P_MIN_REPRODUCE_AGE (idx 5): require maturity before reproducing
    brainParams[5] = 0.15f + rng.nextFloat() * 0.15f
    // Cell type weights (idx 14-21): mapping = weight_idx+1 but skip FOOD(6)
    // idx 14=SOFT_TISSUE, 15=MOUTH, 16=FLAGELLA, 17=EYE, 18=SPIKE,
    // 19=PHOTOSYNTHETIC, 20=ARMOR, 21=SKIN
    // Bias PHOTOSYNTHETIC high; keep non-productive types low for viability.
    brainParams[14] = 0.2f + rng.nextFloat() * 0.3f  // SOFT_TISSUE: moderate
    brainParams[15] = rng.nextFloat() * 0.15f         // MOUTH: low
    brainParams[16] = rng.nextFloat() * 0.15f         // FLAGELLA: low
    brainParams[17] = rng.nextFloat() * 0.1f          // EYE: very low
    brainParams[18] = rng.nextFloat() * 0.1f          // SPIKE: very low
    brainParams[19] = 0.7f + rng.nextFloat() * 0.3f   // PHOTOSYNTHETIC: dominant
    brainParams[20] = rng.nextFloat() * 0.1f          // ARMOR: very low
    brainParams[21] = rng.nextFloat() * 0.15f         // SKIN: low
    // P_REPRODUCE_ENERGY_FRAC (idx 22): don't drain parent too much
    brainParams[22] = 0.1f + rng.nextFloat() * 0.2f

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
