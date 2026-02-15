export interface TileState {
  q: number;
  r: number;
  terrainType: number;
  cellType: number;
  organismId: number;
  energy: number;
}

export interface GridState {
  width: number;
  height: number;
  tiles: TileState[];
}

export const TerrainType = {
  GROUND: 0,
  WATER: 1,
  ROCK: 2,
  FERTILE: 3,
  TOXIC: 4,
} as const;

export const CellType = {
  EMPTY: 0,
  SKIN: 1,
  ARMOR: 2,
  MOUTH: 3,
  SPIKE: 4,
  PHOTOSYNTHETIC: 5,
  EYE: 6,
  FLAGELLA: 7,
  MEMBRANE: 8,
  ROOT: 9,
  TEETH: 10,
  CHEMICAL_SENSOR: 11,
  TOUCH_SENSOR: 12,
  CILIA: 13,
  PSEUDOPOD: 14,
  STORAGE_VACUOLE: 15,
  REPRODUCTIVE: 16,
  SIGNAL_EMITTER: 17,
  PIGMENT: 18,
} as const;

export const TERRAIN_COLORS: Record<number, string> = {
  [TerrainType.GROUND]: '#2d5a1e',
  [TerrainType.WATER]: '#1a4a7a',
  [TerrainType.ROCK]: '#4a4a4a',
  [TerrainType.FERTILE]: '#5a3a1a',
  [TerrainType.TOXIC]: '#4a1a4a',
};

export const CELL_COLORS: Record<number, string> = {
  [CellType.SKIN]: '#d4a574',
  [CellType.ARMOR]: '#8a8a8a',
  [CellType.MOUTH]: '#cc3333',
  [CellType.SPIKE]: '#ff6644',
  [CellType.PHOTOSYNTHETIC]: '#44cc44',
  [CellType.EYE]: '#ffffff',
  [CellType.FLAGELLA]: '#cc88dd',
  [CellType.MEMBRANE]: '#88ccdd',
  [CellType.ROOT]: '#886633',
  [CellType.TEETH]: '#ffdddd',
  [CellType.CHEMICAL_SENSOR]: '#dddd44',
  [CellType.TOUCH_SENSOR]: '#dd88aa',
  [CellType.CILIA]: '#bb99cc',
  [CellType.PSEUDOPOD]: '#99bbcc',
  [CellType.STORAGE_VACUOLE]: '#ddaa55',
  [CellType.REPRODUCTIVE]: '#ff88ff',
  [CellType.SIGNAL_EMITTER]: '#ffff88',
  [CellType.PIGMENT]: '#ff44ff',
};

export const CELL_TYPE_NAMES: Record<number, string> = {
  [CellType.EMPTY]: 'Empty',
  [CellType.SKIN]: 'Skin',
  [CellType.ARMOR]: 'Armor',
  [CellType.MOUTH]: 'Mouth',
  [CellType.SPIKE]: 'Spike',
  [CellType.PHOTOSYNTHETIC]: 'Photosynthetic',
  [CellType.EYE]: 'Eye',
  [CellType.FLAGELLA]: 'Flagella',
  [CellType.MEMBRANE]: 'Membrane',
  [CellType.ROOT]: 'Root',
  [CellType.TEETH]: 'Teeth',
  [CellType.CHEMICAL_SENSOR]: 'Chemical Sensor',
  [CellType.TOUCH_SENSOR]: 'Touch Sensor',
  [CellType.CILIA]: 'Cilia',
  [CellType.PSEUDOPOD]: 'Pseudopod',
  [CellType.STORAGE_VACUOLE]: 'Storage Vacuole',
  [CellType.REPRODUCTIVE]: 'Reproductive',
  [CellType.SIGNAL_EMITTER]: 'Signal Emitter',
  [CellType.PIGMENT]: 'Pigment',
};

export const TERRAIN_TYPE_NAMES: Record<number, string> = {
  [TerrainType.GROUND]: 'Ground',
  [TerrainType.WATER]: 'Water',
  [TerrainType.ROCK]: 'Rock',
  [TerrainType.FERTILE]: 'Fertile Soil',
  [TerrainType.TOXIC]: 'Toxic',
};
