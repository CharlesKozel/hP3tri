import type {HistoryEntry} from '../types';

interface FitnessChartProps {
    history: HistoryEntry[];
}

export default function FitnessChart({history}: FitnessChartProps) {
    if (history.length === 0) {
        return (
            <div style={{color: '#666', fontSize: 12, padding: '20px 0', textAlign: 'center'}}>
                No history yet
            </div>
        );
    }

    const w = 400;
    const h = 180;
    const pad = {top: 10, right: 10, bottom: 24, left: 45};
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const maxGen = Math.max(history[history.length - 1].generation, 1);
    const maxFit = history.reduce((m, e) => Math.max(m, e.bestFitness, e.avgFitness), 1);

    const scaleX = (gen: number) => pad.left + (gen / maxGen) * plotW;
    const scaleY = (val: number) => pad.top + plotH - (val / maxFit) * plotH;

    const bestLine = history.map(e => `${scaleX(e.generation)},${scaleY(e.bestFitness)}`).join(' ');
    const avgLine = history.map(e => `${scaleX(e.generation)},${scaleY(e.avgFitness)}`).join(' ');

    const yTicks = 4;
    const yTickMarks = Array.from({length: yTicks + 1}, (_, i) => (maxFit / yTicks) * i);

    return (
        <div>
            <div style={{color: '#888', fontSize: 11, marginBottom: 4}}>Fitness Over Generations</div>
            <svg width={w} height={h} style={{display: 'block'}}>
                <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="#1a1a1a" rx={2}/>

                {yTickMarks.map((val, i) => (
                    <g key={i}>
                        <line
                            x1={pad.left} x2={pad.left + plotW}
                            y1={scaleY(val)} y2={scaleY(val)}
                            stroke="#2a2a2a" strokeWidth={1}
                        />
                        <text x={pad.left - 4} y={scaleY(val) + 3} textAnchor="end" fill="#666" fontSize={9}>
                            {Math.round(val)}
                        </text>
                    </g>
                ))}

                <polyline points={avgLine} fill="none" stroke="#2a6a2a" strokeWidth={1.5} opacity={0.7}/>
                <polyline points={bestLine} fill="none" stroke="#22aa44" strokeWidth={2}/>

                <text x={pad.left + plotW / 2} y={h - 4} textAnchor="middle" fill="#666" fontSize={10}>
                    Generation
                </text>

                <rect x={pad.left + plotW - 100} y={pad.top + 4} width={8} height={2} fill="#22aa44"/>
                <text x={pad.left + plotW - 88} y={pad.top + 8} fill="#888" fontSize={9}>Best</text>
                <rect x={pad.left + plotW - 50} y={pad.top + 4} width={8} height={2} fill="#2a6a2a" opacity={0.7}/>
                <text x={pad.left + plotW - 38} y={pad.top + 8} fill="#888" fontSize={9}>Avg</text>
            </svg>
        </div>
    );
}
