package dev.ckozel.alife.hp3tri.grid

import kotlinx.serialization.Serializable

@Serializable
data class TileState(
    val q: Int,
    val r: Int,
    val terrainType: Int,
    val cellType: Int,
    val organismId: Int,
)

@Serializable
data class GridState(
    val width: Int,
    val height: Int,
    val tiles: List<TileState>,
)

fun createDemoGrid(): GridState {
    val width = 32
    val height = 32
    val tiles = mutableListOf<TileState>()

    // Organism 1: a small plant-like organism near center
    val org1Center = Pair(10, 10)
    val org1Cells = listOf(
        Triple(10, 10, 5), // photosynthetic (center)
        Triple(11, 10, 5), // photosynthetic
        Triple(10, 9, 5),  // photosynthetic
        Triple(9, 10, 1),  // skin
        Triple(11, 9, 1),  // skin
        Triple(10, 11, 9), // root
        Triple(9, 11, 9),  // root
    )
    for ((q, r, cellType) in org1Cells) {
        tiles.add(TileState(q, r, terrainType = 0, cellType = cellType, organismId = 1))
    }

    // Organism 2: a predator-like organism
    val org2Cells = listOf(
        Triple(20, 15, 1), // skin (center)
        Triple(21, 15, 3), // mouth (front)
        Triple(21, 14, 4), // spike
        Triple(20, 16, 7), // flagella
        Triple(19, 15, 2), // armor (back)
        Triple(19, 16, 1), // skin
    )
    for ((q, r, cellType) in org2Cells) {
        tiles.add(TileState(q, r, terrainType = 0, cellType = cellType, organismId = 2))
    }

    // Organism 3: armored defender
    val org3Cells = listOf(
        Triple(25, 5, 5),  // photosynthetic (center)
        Triple(26, 5, 2),  // armor
        Triple(25, 4, 2),  // armor
        Triple(24, 5, 2),  // armor
        Triple(24, 6, 2),  // armor
        Triple(25, 6, 2),  // armor
        Triple(26, 4, 2),  // armor
    )
    for ((q, r, cellType) in org3Cells) {
        tiles.add(TileState(q, r, terrainType = 0, cellType = cellType, organismId = 3))
    }

    // Some terrain variation
    for (q in 14..18) {
        for (r in 5..9) {
            tiles.add(TileState(q, r, terrainType = 1, cellType = 0, organismId = 0)) // water
        }
    }
    for (q in 5..8) {
        for (r in 20..23) {
            tiles.add(TileState(q, r, terrainType = 3, cellType = 0, organismId = 0)) // fertile
        }
    }
    tiles.add(TileState(12, 12, terrainType = 2, cellType = 0, organismId = 0)) // rock
    tiles.add(TileState(13, 12, terrainType = 2, cellType = 0, organismId = 0)) // rock
    tiles.add(TileState(12, 13, terrainType = 2, cellType = 0, organismId = 0)) // rock

    return GridState(width, height, tiles)
}
