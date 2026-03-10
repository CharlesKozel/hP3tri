import {useEffect, useMemo, useRef} from 'react';
import type {CellTypeInfo, SimulationState} from '../types';

const SQRT3 = Math.sqrt(3);

interface Props {
    snapshot: SimulationState | null;
    cellTypes: CellTypeInfo[];
    size: number;
    selected: boolean;
    elo: number;
    rank: number;
    genomeId: number;
    onClick?: () => void;
}

export default function OrganismThumbnail({snapshot, cellTypes, size, selected, elo, rank, genomeId, onClick}: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const cellColorMap = useMemo(() => {
        const map = new Map<number, string>();
        for (const ct of cellTypes) {
            map.set(ct.id, ct.color);
        }
        return map;
    }, [cellTypes]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !snapshot) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, size, size);

        const tiles = snapshot.grid.tiles.filter(t => t.cellType !== 0);
        if (tiles.length === 0) {
            ctx.fillStyle = '#333';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('no cells', size / 2, size / 2);
            return;
        }

        let minQ = Infinity, maxQ = -Infinity, minR = Infinity, maxR = -Infinity;
        for (const t of tiles) {
            if (t.q < minQ) minQ = t.q;
            if (t.q > maxQ) maxQ = t.q;
            if (t.r < minR) minR = t.r;
            if (t.r > maxR) maxR = t.r;
        }

        const spanQ = maxQ - minQ + 1;
        const spanR = maxR - minR + 1;
        const hexH = SQRT3;
        const hexW = 2;
        const totalW = spanQ * hexW * 0.75 + 0.5;
        const totalH = spanR * hexH + hexH * 0.5;
        const padding = 4;
        const available = size - padding * 2;
        const hexSize = Math.min(available / totalW, available / totalH) * 0.9;

        const centerQ = (minQ + maxQ) / 2;
        const centerR = (minR + maxR) / 2;

        for (const tile of tiles) {
            const q = tile.q - centerQ;
            const r = tile.r - centerR;
            const px = size / 2 + (q * 1.5) * hexSize;
            const py = size / 2 + (r * SQRT3 + (q % 2 !== 0 ? SQRT3 / 2 : 0)) * hexSize;

            ctx.beginPath();
            for (let i = 0; i < 6; i++) {
                const angle = (Math.PI / 3) * i - Math.PI / 6;
                const x = px + hexSize * 0.9 * Math.cos(angle);
                const y = py + hexSize * 0.9 * Math.sin(angle);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fillStyle = cellColorMap.get(tile.cellType) ?? '#ffffff';
            ctx.fill();
        }
    }, [snapshot, cellTypes, size, cellColorMap]);

    const borderColor = selected ? '#4c8' : '#333';
    const bg = selected ? '#1a2a1a' : '#1a1a1a';

    return (
        <div
            onClick={onClick}
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: 4,
                background: bg,
                border: `2px solid ${borderColor}`,
                borderRadius: 6,
                cursor: onClick ? 'pointer' : 'default',
                width: size + 8,
                flexShrink: 0,
            }}
        >
            <canvas
                ref={canvasRef}
                style={{width: size, height: size, borderRadius: 4}}
            />
            <div style={{
                fontSize: 10,
                fontFamily: 'monospace',
                color: '#ccc',
                textAlign: 'center',
                marginTop: 2,
                lineHeight: '14px',
            }}>
                <div style={{color: '#888'}}>#{rank} g{genomeId}</div>
                <div style={{color: elo >= 1100 ? '#4c8' : elo <= 900 ? '#c44' : '#cc8'}}>
                    {Math.round(elo)} ELO
                </div>
            </div>
        </div>
    );
}
