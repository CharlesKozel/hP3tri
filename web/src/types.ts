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
  genomeId: number;
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

export interface GenomeIdentity {
    tint: string;
    patternId: number;
    label: string;
}

export const GENOME_IDENTITIES: GenomeIdentity[] = [
    { tint: '#4488ff', patternId: 0, label: '★' },
    { tint: '#44cc44', patternId: 1, label: '☽' },
    { tint: '#ff6644', patternId: 2, label: '◆' },
    { tint: '#cc44ff', patternId: 3, label: '●' },
    { tint: '#ffcc22', patternId: 4, label: '✚' },
];

export const TERRAIN_TYPE_NAMES: Record<number, string> = {
  [TerrainType.GROUND]: 'Ground',
  [TerrainType.WATER]: 'Water',
  [TerrainType.ROCK]: 'Rock',
  [TerrainType.FERTILE]: 'Fertile Soil',
  [TerrainType.TOXIC]: 'Toxic',
};

export interface EvolutionStatus {
    running: boolean;
    generation: number;
    totalGenerations: number;
    archiveFillRate: number;
    bestFitness: number;
    matchesCompleted: number;
    log: string[];
}

export interface ArchiveEntry {
    binX: number;
    binY: number;
    genomeId: number;
    fitness: number;
    mobility: number;
    aggression: number;
    symmetryMode: number;
}

export interface HistoryEntry {
    generation: number;
    bestFitness: number;
    avgFitness: number;
    fillRate: number;
}