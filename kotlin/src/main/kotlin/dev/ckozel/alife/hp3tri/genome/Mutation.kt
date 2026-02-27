package dev.ckozel.alife.hp3tri.genome

import java.util.Random

fun mutate(genome: Genome, rng: Random): Genome {
    val weights = genome.cppnWeights.map { w ->
        if (rng.nextFloat() < 0.8f) w + (rng.nextGaussian() * 0.1).toFloat()
        else w
    }

    val brainParams = genome.brainParams.map { p ->
        (p + (rng.nextGaussian() * 0.05).toFloat()).coerceIn(0f, 1f)
    }

    val activations = genome.cppnActivations.map { a ->
        if (rng.nextFloat() < 0.05f) rng.nextInt(8) else a
    }

    val symmetryMode = if (rng.nextFloat() < 0.02f) rng.nextInt(7) else genome.symmetryMode

    return genome.copy(
        cppnWeights = weights,
        cppnActivations = activations,
        symmetryMode = symmetryMode,
        brainParams = brainParams,
        reproductionThreshold = mutateMetabolic(genome.reproductionThreshold, rng),
        offspringEnergy = mutateMetabolic(genome.offspringEnergy, rng),
        growthDesire = mutateMetabolic(genome.growthDesire, rng),
        movementWillingness = mutateMetabolic(genome.movementWillingness, rng),
    )
}

fun crossover(a: Genome, b: Genome, nextId: Int, rng: Random): Genome {
    val weights = a.cppnWeights.zip(b.cppnWeights).map { (wa, wb) ->
        if (rng.nextBoolean()) wa else wb
    }

    val activations = a.cppnActivations.zip(b.cppnActivations).map { (aa, ab) ->
        if (rng.nextBoolean()) aa else ab
    }

    val brainParams = a.brainParams.zip(b.brainParams).map { (pa, pb) ->
        if (rng.nextBoolean()) pa else pb
    }

    return Genome(
        id = nextId,
        cppnWeights = weights,
        cppnActivations = activations,
        symmetryMode = if (rng.nextBoolean()) a.symmetryMode else b.symmetryMode,
        brainParams = brainParams,
        reproductionThreshold = (a.reproductionThreshold + b.reproductionThreshold) / 2f,
        offspringEnergy = (a.offspringEnergy + b.offspringEnergy) / 2f,
        growthDesire = (a.growthDesire + b.growthDesire) / 2f,
        movementWillingness = (a.movementWillingness + b.movementWillingness) / 2f,
    )
}

private fun mutateMetabolic(value: Float, rng: Random): Float =
    (value + (rng.nextGaussian() * 0.05).toFloat()).coerceIn(0f, 1f)
