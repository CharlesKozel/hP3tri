import {useMemo} from 'react';
import type {OrganismState} from '../types';
import {GENOME_IDENTITIES} from '../types';

interface StatsPanelProps {
    organisms: OrganismState[];
    currentTick: number;
    totalTicks: number;
    open: boolean;
    onClose: () => void;
}

interface GenomeStats {
    genomeId: number;
    count: number;
    totalCells: number;
    avgEnergy: number;
    minEnergy: number;
    maxEnergy: number;
    avgSize: number;
    minSize: number;
    maxSize: number;
}

export default function StatsPanel({organisms, currentTick, totalTicks, open, onClose}: StatsPanelProps) {
    const aliveCount = useMemo(() => organisms.filter(o => o.alive).length, [organisms]);

    const genomeStats = useMemo((): GenomeStats[] => {
        const byGenome = new Map<number, OrganismState[]>();
        for (const o of organisms) {
            if (!o.alive) continue;
            const list = byGenome.get(o.genomeId);
            if (list) list.push(o);
            else byGenome.set(o.genomeId, [o]);
        }

        const stats: GenomeStats[] = [];
        for (const [genomeId, orgs] of byGenome) {
            const energies = orgs.map(o => o.energy);
            const sizes = orgs.map(o => o.cellCount);
            const totalCells = sizes.reduce((a, b) => a + b, 0);
            stats.push({
                genomeId,
                count: orgs.length,
                totalCells,
                avgEnergy: Math.round(energies.reduce((a, b) => a + b, 0) / orgs.length),
                minEnergy: Math.min(...energies),
                maxEnergy: Math.max(...energies),
                avgSize: Math.round(sizes.reduce((a, b) => a + b, 0) / orgs.length),
                minSize: Math.min(...sizes),
                maxSize: Math.max(...sizes),
            });
        }

        stats.sort((a, b) => b.totalCells - a.totalCells);
        return stats;
    }, [organisms]);

    const genomeIndexMap = useMemo(() => {
        const map = new Map<number, number>();
        genomeStats.forEach((g, i) => map.set(g.genomeId, i));
        return map;
    }, [genomeStats]);

    return (
        <div style={{
            width: open ? 360 : 0,
            minWidth: open ? 360 : 0,
            height: '100vh',
            background: '#1a1a1a',
            borderLeft: open ? '1px solid #333' : 'none',
            transition: 'width 0.3s ease, min-width 0.3s ease',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            fontFamily: 'monospace',
            fontSize: 13,
            color: '#ccc',
            flexShrink: 0,
        }}>
            <div style={{minWidth: 360}}>
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                borderBottom: '1px solid #333',
            }}>
                <span style={{fontSize: 15, fontWeight: 'bold'}}>Stats</span>
                <button onClick={onClose} style={closeBtnStyle}>X</button>
            </div>

            <div style={{overflowY: 'auto', flex: 1, padding: '12px 16px'}}>
                <div style={{marginBottom: 16, color: '#999'}}>
                    <div>Tick: {currentTick} / {totalTicks}</div>
                    <div>Organisms alive: {aliveCount}</div>
                </div>

                {genomeStats.map(g => (
                    <div key={g.genomeId} style={{
                        marginBottom: 12,
                        padding: '10px 12px',
                        background: '#222',
                        borderRadius: 4,
                        border: '1px solid #333',
                    }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: 6,
                            fontWeight: 'bold',
                            color: '#eee',
                        }}>
                            <span style={{display: 'flex', alignItems: 'center', gap: 6}}>
                                <span style={{
                                    display: 'inline-block',
                                    width: 12,
                                    height: 12,
                                    borderRadius: 2,
                                    background: GENOME_IDENTITIES[(genomeIndexMap.get(g.genomeId) ?? 0) % GENOME_IDENTITIES.length].tint,
                                }} />
                                {GENOME_IDENTITIES[(genomeIndexMap.get(g.genomeId) ?? 0) % GENOME_IDENTITIES.length].label} Genome {g.genomeId}
                            </span>
                            <span style={{color: '#999', fontWeight: 'normal'}}>
                                {g.count} organism{g.count !== 1 ? 's' : ''}
                            </span>
                        </div>
                        <StatRow label="Total cells" value={g.totalCells} />
                        <StatRow label="Energy" avg={g.avgEnergy} min={g.minEnergy} max={g.maxEnergy} />
                        <StatRow label="Size" avg={g.avgSize} min={g.minSize} max={g.maxSize} />
                    </div>
                ))}

                {genomeStats.length === 0 && (
                    <div style={{color: '#666', fontStyle: 'italic'}}>No living organisms</div>
                )}
            </div>
            </div>
        </div>
    );
}

function StatRow({label, value, avg, min, max}: {
    label: string;
    value?: number;
    avg?: number;
    min?: number;
    max?: number;
}) {
    return (
        <div style={{display: 'flex', justifyContent: 'space-between', padding: '2px 0', color: '#aaa'}}>
            <span>{label}</span>
            {value !== undefined ? (
                <span style={{color: '#ccc'}}>{value}</span>
            ) : (
                <span style={{color: '#ccc'}}>
                    avg:{avg} min:{min} max:{max}
                </span>
            )}
        </div>
    );
}

const closeBtnStyle: React.CSSProperties = {
    padding: '2px 8px',
    background: '#333',
    color: '#ccc',
    border: '1px solid #555',
    borderRadius: 4,
    cursor: 'pointer',
    fontFamily: 'monospace',
    fontSize: 13,
};
