export interface TileState {
  q: number;
  r: number;
  terrainType: number;
  cellType: number;
  organismId: number;
}

export interface GridState {
  width: number;
  height: number;
  tiles: TileState[];
}

export interface OrganismState {
  id: number;
  energy: number;
  alive: boolean;
  cellCount: number;
}

export interface SimulationState {
  tick: number;
  status: string;
  grid: GridState;
  organisms: OrganismState[];
}

export interface ReplayInfo {
  totalTicks: number;
  width: number;
  height: number;
}

export interface CellTypeInfo {
  id: number;
  name: string;
  color: string;
}

export const TerrainType = {
  GROUND: 0,
  WATER: 1,
  ROCK: 2,
  FERTILE: 3,
  TOXIC: 4,
} as const;

export const TERRAIN_COLORS: Record<number, string> = {
  [TerrainType.GROUND]: '#2d5a1e',
  [TerrainType.WATER]: '#1a4a7a',
  [TerrainType.ROCK]: '#4a4a4a',
  [TerrainType.FERTILE]: '#5a3a1a',
  [TerrainType.TOXIC]: '#4a1a4a',
};

export const TERRAIN_TYPE_NAMES: Record<number, string> = {
  [TerrainType.GROUND]: 'Ground',
  [TerrainType.WATER]: 'Water',
  [TerrainType.ROCK]: 'Rock',
  [TerrainType.FERTILE]: 'Fertile Soil',
  [TerrainType.TOXIC]: 'Toxic',
};