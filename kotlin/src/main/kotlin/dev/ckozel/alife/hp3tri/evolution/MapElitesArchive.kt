package dev.ckozel.alife.hp3tri.evolution

import dev.ckozel.alife.hp3tri.genome.Genome
import java.util.Random

data class ArchiveEntryData(
    val binX: Int,
    val binY: Int,
    val genome: Genome,
    val fitness: Float,
)

class MapElitesArchive(
    val binsX: Int,
    val binsY: Int,
    val rangeX: Pair<Float, Float>,
    val rangeY: Pair<Float, Float>,
) {
    private val archive: Array<Array<Pair<Genome, Float>?>> = Array(binsX) { arrayOfNulls(binsY) }

    fun add(genome: Genome, fitness: Float, bd: Pair<Float, Float>): Boolean {
        val binX = toBinIndex(bd.first, rangeX, binsX)
        val binY = toBinIndex(bd.second, rangeY, binsY)
        val existing = archive[binX][binY]
        if (existing == null || fitness > existing.second) {
            archive[binX][binY] = Pair(genome, fitness)
            return true
        }
        return false
    }

    fun sample(n: Int, rng: Random): List<Genome> {
        val occupied = allGenomes()
        if (occupied.isEmpty()) return emptyList()
        return List(n) { occupied[rng.nextInt(occupied.size)].first }
    }

    fun sampleDiverse(n: Int, rng: Random): List<Genome> {
        val occupiedCells = mutableListOf<Pair<Int, Int>>()
        for (x in 0 until binsX) {
            for (y in 0 until binsY) {
                if (archive[x][y] != null) occupiedCells.add(Pair(x, y))
            }
        }
        if (occupiedCells.isEmpty()) return emptyList()

        val selected = mutableListOf<Genome>()
        val usedCells = mutableSetOf<Pair<Int, Int>>()
        while (selected.size < n) {
            val available = occupiedCells.filter { it !in usedCells }
            val pool = available.ifEmpty { occupiedCells }
            val cell = pool[rng.nextInt(pool.size)]
            usedCells.add(cell)
            selected.add(archive[cell.first][cell.second]!!.first)
        }
        return selected
    }

    fun fillRate(): Float {
        var filled = 0
        for (x in 0 until binsX) {
            for (y in 0 until binsY) {
                if (archive[x][y] != null) filled++
            }
        }
        return filled.toFloat() / (binsX * binsY)
    }

    fun allGenomes(): List<Pair<Genome, Float>> {
        val result = mutableListOf<Pair<Genome, Float>>()
        for (x in 0 until binsX) {
            for (y in 0 until binsY) {
                archive[x][y]?.let { result.add(it) }
            }
        }
        return result
    }

    fun bestFitness(): Float =
        allGenomes().maxOfOrNull { it.second } ?: 0f

    fun allEntries(): List<ArchiveEntryData> {
        val result = mutableListOf<ArchiveEntryData>()
        for (x in 0 until binsX) {
            for (y in 0 until binsY) {
                archive[x][y]?.let { (genome, fitness) ->
                    result.add(ArchiveEntryData(x, y, genome, fitness))
                }
            }
        }
        return result
    }

    fun getGenomeById(id: Int): Genome? {
        for (x in 0 until binsX) {
            for (y in 0 until binsY) {
                archive[x][y]?.let { (genome, _) ->
                    if (genome.id == id) return genome
                }
            }
        }
        return null
    }

    private fun toBinIndex(value: Float, range: Pair<Float, Float>, bins: Int): Int {
        val normalized = ((value - range.first) / (range.second - range.first)).coerceIn(0f, 0.9999f)
        return (normalized * bins).toInt()
    }
}
