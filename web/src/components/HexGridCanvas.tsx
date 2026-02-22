import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import type {CellTypeInfo, GridState, OrganismState, TileState} from '../types';
import {
    TERRAIN_COLORS,
    TERRAIN_TYPE_NAMES,
    TerrainType,
} from '../types';

const HEX_SIZE = 14;
const SQRT3 = Math.sqrt(3);

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

        for (let r = 0; r < grid.height; r++) {
            for (let q = 0; q < grid.width; q++) {
                const [px, py] = axialToPixel(q, r, size, grid.width);

                if (
                    px < viewLeft - margin ||
                    px > viewRight + margin ||
                    py < viewTop - margin ||
                    py > viewBottom + margin
                ) {
                    continue;
                }

                const key = `${q},${r}`;
                const tile = tileMap.current.get(key);
                const terrain = tile?.terrainType ?? TerrainType.GROUND;
                const terrainColor = TERRAIN_COLORS[terrain] ?? TERRAIN_COLORS[TerrainType.GROUND];

                drawHex(ctx, px, py, size, terrainColor, '#1a1a1a');

                if (tile && tile.cellType !== 0) {
                    const cellColor = cellColorMap.get(tile.cellType) ?? '#ffffff';
                    const innerSize = size * 0.7;
                    drawHex(ctx, px, py, innerSize, cellColor, cellColor);
                }
            }
        }

        ctx.restore();
    }, [grid, offset, zoom, cellColorMap]);

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
