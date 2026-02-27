import {useEffect, useRef} from 'react';
import type {ArchiveEntry} from '../types';

interface MapElitesGridProps {
    entries: ArchiveEntry[];
    binsX: number;
    binsY: number;
    selectedBin: {x: number; y: number} | null;
    onSelectBin: (x: number, y: number) => void;
}

export default function MapElitesGrid({entries, binsX, binsY, selectedBin, onSelectBin}: MapElitesGridProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const size = 240;
    const cellW = size / binsX;
    const cellH = size / binsY;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, size, size);

        const maxFitness = entries.reduce((m, e) => Math.max(m, e.fitness), 1);

        const grid = new Map<string, ArchiveEntry>();
        for (const e of entries) {
            grid.set(`${e.binX},${e.binY}`, e);
        }

        for (let x = 0; x < binsX; x++) {
            for (let y = 0; y < binsY; y++) {
                const entry = grid.get(`${x},${y}`);
                const px = x * cellW;
                const py = (binsY - 1 - y) * cellH;

                if (entry) {
                    const t = Math.min(entry.fitness / maxFitness, 1);
                    const r = Math.round(26 + t * 8);
                    const g = Math.round(26 + t * 144);
                    const b = Math.round(26 + t * 42);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                } else {
                    ctx.fillStyle = '#1a1a1a';
                }
                ctx.fillRect(px, py, cellW - 1, cellH - 1);

                if (selectedBin && selectedBin.x === x && selectedBin.y === y) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(px + 1, py + 1, cellW - 3, cellH - 3);
                }
            }
        }
    }, [entries, binsX, binsY, selectedBin, cellW, cellH]);

    const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const x = Math.floor((e.clientX - rect.left) / cellW);
        const y = binsY - 1 - Math.floor((e.clientY - rect.top) / cellH);
        if (x >= 0 && x < binsX && y >= 0 && y < binsY) {
            onSelectBin(x, y);
        }
    };

    return (
        <div>
            <div style={{color: '#888', fontSize: 11, marginBottom: 4}}>
                MAP-Elites Archive ({entries.length}/{binsX * binsY} filled)
            </div>
            <canvas
                ref={canvasRef}
                width={size}
                height={size}
                onClick={handleClick}
                style={{cursor: 'pointer', borderRadius: 4, border: '1px solid #333'}}
            />
            <div style={{display: 'flex', justifyContent: 'space-between', color: '#666', fontSize: 10, marginTop: 2}}>
                <span>Mobility &rarr;</span>
                <span>&uarr; Aggression</span>
            </div>
        </div>
    );
}
