import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import type {CellTypeInfo, GridState, OrganismState, TileState} from '../types';
import {
    GENOME_IDENTITIES,
    TERRAIN_COLORS,
    TERRAIN_TYPE_NAMES,
    TerrainType,
} from '../types';

const HEX_SIZE = 14;
const SQRT3 = Math.sqrt(3);
const CELL_ICON_OPACITY = 0.45;

const CELL_TYPE_ICON_PATHS: Record<number, string> = {
    1: '/icons/soft_tissue.svg',
    2: '/icons/mouth.svg',
    3: '/icons/flagella.svg',
    4: '/icons/eye.svg',
    5: '/icons/spike.svg',
    6: '/icons/food.svg',
    7: '/icons/photosynthetic.svg',
    8: '/icons/armor.svg',
    9: '/icons/skin.svg',
};

// Convert axial (q, r) to odd-r offset (col, row) for rectangular display.
// Data model stays axial — this is purely a rendering transform.
// Convert axial (q, r) to odd-r offset (col, row) for rectangular display.
// Wraps col with modulo so the toroidal grid renders as a rectangle.
function axialToOffset(q: number, r: number, width: number): [number, number] {
    const col = ((q + Math.floor(r / 2)) % width + width) % width;
    return [col, r];
}

function offsetToAxial(col: number, row: number, width: number): [number, number] {
    const q = ((col - Math.floor(row / 2)) % width + width) % width;
    return [q, row];
}

// Pointy-top hex layout using offset coordinates for rectangular grid.
function axialToPixel(q: number, r: number, size: number, gridWidth: number): [number, number] {
    const [col, row] = axialToOffset(q, r, gridWidth);
    const hexW = SQRT3 * size;
    const hexH = size * (3 / 2);
    const xShift = (row % 2 !== 0) ? hexW / 2 : 0;
    const x = col * hexW + xShift;
    const y = row * hexH;
    return [x, y];
}

function pixelToAxial(
    px: number,
    py: number,
    size: number,
    gridWidth: number,
): [number, number] {
    const hexW = SQRT3 * size;
    const hexH = size * (3 / 2);
    const row = Math.round(py / hexH);
    const xShift = (row % 2 !== 0) ? hexW / 2 : 0;
    const col = Math.round((px - xShift) / hexW);
    return offsetToAxial(col, row, gridWidth);
}

function drawHex(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    size: number,
    fillColor: string,
    strokeColor: string,
) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const x = cx + size * Math.cos(angle);
        const y = cy + size * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 0.5;
    ctx.stroke();
}

const AXIAL_NEIGHBORS: [number, number][] = [
    [+1, 0], [+1, -1], [0, -1], [-1, 0], [-1, +1], [0, +1],
];

const NEIGHBOR_TO_EDGE = [0, 5, 4, 3, 2, 1];

const HEX_VERTICES: [number, number][] = Array.from({length: 6}, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return [Math.cos(angle), Math.sin(angle)];
});

function wrapAxial(q: number, r: number, width: number, height: number): [number, number] {
    return [((q % width) + width) % width, ((r % height) + height) % height];
}

function drawHexEdge(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    size: number,
    edgeIndex: number,
    color: string,
    lineWidth: number,
) {
    const v0 = HEX_VERTICES[edgeIndex];
    const v1 = HEX_VERTICES[(edgeIndex + 1) % 6];
    ctx.beginPath();
    ctx.moveTo(cx + size * v0[0], cy + size * v0[1]);
    ctx.lineTo(cx + size * v1[0], cy + size * v1[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
}

function addHexToPath(path: Path2D, cx: number, cy: number, size: number) {
    for (let i = 0; i < 6; i++) {
        const x = cx + size * HEX_VERTICES[i][0];
        const y = cy + size * HEX_VERTICES[i][1];
        if (i === 0) path.moveTo(x, y);
        else path.lineTo(x, y);
    }
    path.closePath();
}

function createPatternCanvases(): HTMLCanvasElement[] {
    const S = 24;
    const canvases: HTMLCanvasElement[] = [];

    for (let p = 0; p < 5; p++) {
        const c = document.createElement('canvas');
        c.width = S;
        c.height = S;
        const ctx = c.getContext('2d')!;
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        const mid = S / 2;

        if (p === 0) {
            // Stars — 6-pointed star
            ctx.beginPath();
            for (let i = 0; i < 12; i++) {
                const angle = (Math.PI / 6) * i - Math.PI / 2;
                const r = i % 2 === 0 ? 7 : 3.5;
                const x = mid + r * Math.cos(angle);
                const y = mid + r * Math.sin(angle);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fill();
        } else if (p === 1) {
            // Moons — crescent
            ctx.beginPath();
            ctx.arc(mid, mid, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.arc(mid + 3, mid - 1, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalCompositeOperation = 'source-over';
        } else if (p === 2) {
            // Diamonds — rotated square
            ctx.beginPath();
            ctx.moveTo(mid, mid - 7);
            ctx.lineTo(mid + 5, mid);
            ctx.lineTo(mid, mid + 7);
            ctx.lineTo(mid - 5, mid);
            ctx.closePath();
            ctx.fill();
        } else if (p === 3) {
            // Dots — filled circle
            ctx.beginPath();
            ctx.arc(mid, mid, 4, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Crosses — plus shape
            ctx.fillRect(mid - 2, mid - 6, 4, 12);
            ctx.fillRect(mid - 6, mid - 2, 12, 4);
        }

        canvases.push(c);
    }
    return canvases;
}

interface Props {
    grid: GridState;
    organisms?: OrganismState[];
    cellTypes: CellTypeInfo[];
}

export default function HexGridCanvas({grid, organisms, cellTypes}: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [offset, setOffset] = useState({x: 40, y: 40});
    const [zoom, setZoom] = useState(1);
    const [dragging, setDragging] = useState(false);
    const [dragStart, setDragStart] = useState({x: 0, y: 0});
    const [hoverTile, setHoverTile] = useState<TileState | null>(null);
    const [hoverCoord, setHoverCoord] = useState<{ q: number; r: number } | null>(null);

    const cellColorMap = useMemo(() => {
        const map = new Map<number, string>();
        for (const ct of cellTypes) {
            map.set(ct.id, ct.color);
        }
        return map;
    }, [cellTypes]);

    const cellNameMap = useMemo(() => {
        const map = new Map<number, string>();
        for (const ct of cellTypes) {
            map.set(ct.id, ct.name);
        }
        return map;
    }, [cellTypes]);

    const tileMap = useRef(new Map<string, TileState>());

    useEffect(() => {
        const map = new Map<string, TileState>();
        for (const tile of grid.tiles) {
            map.set(`${tile.q},${tile.r}`, tile);
        }
        tileMap.current = map;
    }, [grid]);

    const organismMap = useMemo(() => {
        const map = new Map<number, OrganismState>();
        if (organisms) {
            for (const org of organisms) {
                map.set(org.id, org);
            }
        }
        return map;
    }, [organisms]);

    const patternCanvases = useMemo(() => createPatternCanvases(), []);

    const [cellIcons, setCellIcons] = useState<Map<number, HTMLImageElement>>(new Map());
    useEffect(() => {
        const map = new Map<number, HTMLImageElement>();
        let loaded = 0;
        const entries = Object.entries(CELL_TYPE_ICON_PATHS);
        for (const [idStr, path] of entries) {
            const img = new Image();
            img.src = path;
            img.onload = () => {
                map.set(Number(idStr), img);
                loaded++;
                if (loaded === entries.length) setCellIcons(new Map(map));
            };
        }
    }, []);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.save();
        ctx.translate(offset.x, offset.y);
        ctx.scale(zoom, zoom);

        const size = HEX_SIZE;
        const hexW = size * 2;

        const viewLeft = -offset.x / zoom;
        const viewTop = -offset.y / zoom;
        const viewRight = (rect.width - offset.x) / zoom;
        const viewBottom = (rect.height - offset.y) / zoom;

        const margin = hexW * 2;

        // Pass 1: Terrain hexes
        for (let r = 0; r < grid.height; r++) {
            for (let q = 0; q < grid.width; q++) {
                const [px, py] = axialToPixel(q, r, size, grid.width);
                if (
                    px < viewLeft - margin || px > viewRight + margin ||
                    py < viewTop - margin || py > viewBottom + margin
                ) continue;

                const key = `${q},${r}`;
                const tile = tileMap.current.get(key);
                const terrain = tile?.terrainType ?? TerrainType.GROUND;
                const terrainColor = TERRAIN_COLORS[terrain] ?? TERRAIN_COLORS[TerrainType.GROUND];
                drawHex(ctx, px, py, size, terrainColor, '#1a1a1a');
            }
        }

        // Pass 2: Cell fills + build compound paths per genome for passes 3-4
        const genomePaths = new Map<number, Path2D>();
        const genomeIdByOrgId = new Map<number, number>();

        for (const tile of grid.tiles) {
            if (tile.cellType === 0) continue;
            const [px, py] = axialToPixel(tile.q, tile.r, size, grid.width);
            if (
                px < viewLeft - margin || px > viewRight + margin ||
                py < viewTop - margin || py > viewBottom + margin
            ) continue;

            const cellColor = cellColorMap.get(tile.cellType) ?? '#ffffff';

            if (tile.organismId !== 0) {
                const org = organismMap.get(tile.organismId);
                if (org) {
                    drawHex(ctx, px, py, size, cellColor, '#1a1a1a');
                    ctx.lineWidth = 0.5;

                    genomeIdByOrgId.set(tile.organismId, org.genomeId);
                    let path = genomePaths.get(org.genomeId);
                    if (!path) {
                        path = new Path2D();
                        genomePaths.set(org.genomeId, path);
                    }
                    addHexToPath(path, px, py, size);
                }
            } else {
                drawHex(ctx, px, py, size * 0.7, cellColor, cellColor);
            }

            const icon = cellIcons.get(tile.cellType);
            if (icon) {
                const iconSize = tile.organismId !== 0 ? size : size * 0.7;
                ctx.save();
                ctx.globalAlpha = CELL_ICON_OPACITY;
                ctx.drawImage(icon, px - iconSize, py - iconSize, iconSize * 2, iconSize * 2);
                ctx.restore();
            }
        }

        // Pass 3: Genome tint overlay
        for (const [genomeId, path] of genomePaths) {
            const identity = GENOME_IDENTITIES[genomeId % GENOME_IDENTITIES.length];
            ctx.save();
            ctx.globalAlpha = 0.2;
            ctx.fillStyle = identity.tint;
            ctx.fill(path);
            ctx.restore();
        }

        // Pass 4: Genome pattern overlay (screen-space, zoom-independent)
        for (const [genomeId, path] of genomePaths) {
            const identity = GENOME_IDENTITIES[genomeId % GENOME_IDENTITIES.length];
            const patternCanvas = patternCanvases[identity.patternId];
            if (!patternCanvas) continue;

            const pattern = ctx.createPattern(patternCanvas, 'repeat');
            if (!pattern) continue;

            pattern.setTransform(new DOMMatrix().scale(1 / zoom, 1 / zoom));
            ctx.save();
            ctx.clip(path);
            ctx.globalAlpha = 0.18;
            ctx.fillStyle = pattern;
            ctx.fillRect(viewLeft - margin, viewTop - margin,
                viewRight - viewLeft + margin * 2, viewBottom - viewTop + margin * 2);
            ctx.restore();
        }

        // Pass 5: Organism boundary outlines
        ctx.lineCap = 'round';
        for (const tile of grid.tiles) {
            if (tile.cellType === 0 || tile.organismId === 0) continue;
            const [px, py] = axialToPixel(tile.q, tile.r, size, grid.width);
            if (
                px < viewLeft - margin || px > viewRight + margin ||
                py < viewTop - margin || py > viewBottom + margin
            ) continue;

            const genomeId = genomeIdByOrgId.get(tile.organismId);
            if (genomeId === undefined) continue;
            const identity = GENOME_IDENTITIES[genomeId % GENOME_IDENTITIES.length];

            for (let i = 0; i < 6; i++) {
                const [nq, nr] = wrapAxial(
                    tile.q + AXIAL_NEIGHBORS[i][0],
                    tile.r + AXIAL_NEIGHBORS[i][1],
                    grid.width, grid.height,
                );
                const neighborTile = tileMap.current.get(`${nq},${nr}`);
                if (!neighborTile || neighborTile.organismId !== tile.organismId) {
                    drawHexEdge(ctx, px, py, size, NEIGHBOR_TO_EDGE[i], identity.tint, 2.5);
                }
            }
        }

        ctx.restore();
    }, [grid, offset, zoom, cellColorMap, organismMap, patternCanvases, cellIcons]);

    useEffect(() => {
        draw();
    }, [draw]);

    useEffect(() => {
        const handleResize = () => draw();
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [draw]);

    const handleMouseDown = (e: React.MouseEvent) => {
        setDragging(true);
        setDragStart({x: e.clientX - offset.x, y: e.clientY - offset.y});
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragging) {
            setOffset({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y,
            });
        }

        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - offset.x) / zoom;
        const my = (e.clientY - rect.top - offset.y) / zoom;
        const [q, r] = pixelToAxial(mx, my, HEX_SIZE, grid.width);

        if (q >= 0 && q < grid.width && r >= 0 && r < grid.height) {
            setHoverCoord({q, r});
            const tile = tileMap.current.get(`${q},${r}`) ?? null;
            setHoverTile(tile);
        } else {
            setHoverCoord(null);
            setHoverTile(null);
        }
    };

    const handleMouseUp = () => {
        setDragging(false);
    };

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();

        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
        const newZoom = Math.max(0.1, Math.min(10, zoom * factor));

        setOffset({
            x: mx - (mx - offset.x) * (newZoom / zoom),
            y: my - (my - offset.y) * (newZoom / zoom),
        });
        setZoom(newZoom);
    };

    const infoText = hoverCoord
        ? `(${hoverCoord.q}, ${hoverCoord.r})` +
        (hoverTile
            ? ` | Terrain: ${TERRAIN_TYPE_NAMES[hoverTile.terrainType] ?? 'Unknown'}` +
            (hoverTile.cellType !== 0
                ? ` | Cell: ${cellNameMap.get(hoverTile.cellType) ?? 'Unknown'}` +
                ` | Organism: ${hoverTile.organismId}` +
                (() => {
                    const org = organismMap.get(hoverTile.organismId);
                    return org ? ` | Energy: ${org.energy}` : '';
                })()
                : '')
            : ' | Ground')
        : '';

    return (
        <div style={{position: 'relative', width: '100%', height: '100%'}}>
            <canvas
                ref={canvasRef}
                style={{width: '100%', height: '100%', cursor: dragging ? 'grabbing' : 'grab'}}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
            />
            <div
                style={{
                    position: 'absolute',
                    bottom: 8,
                    left: 8,
                    background: 'rgba(0,0,0,0.7)',
                    color: '#ccc',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 13,
                    fontFamily: 'monospace',
                    pointerEvents: 'none',
                }}
            >
                {infoText || `${grid.width}x${grid.height} grid | Scroll to zoom, drag to pan`}
            </div>
        </div>
    );
}
