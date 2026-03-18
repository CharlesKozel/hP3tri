import {useCallback, useEffect, useRef, useState} from 'react';
import {useNavigate, useSearchParams} from 'react-router-dom';
import {getApiBase} from '../api';

const POLL_INTERVAL = 2000;

interface TrainingStats {
    total_matches: number;
    epsilon: number;
    replay_size: number;
    total_train_steps: number;
    last_loss: number;
    avg_reward_100: number;
    avg_cells_100: number;
}

interface EvolutionStatus {
    running: boolean;
    generation: number;
    totalGenerations: number;
    matchesCompleted: number;
    bestFitness: number;
    log: string[];
}

export default function QLearningDashboard() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const [stats, setStats] = useState<TrainingStats | null>(null);
    const [status, setStatus] = useState<EvolutionStatus | null>(null);
    const [history, setHistory] = useState<TrainingStats[]>([]);
    const [error, setError] = useState<string | null>(null);
    const pollRef = useRef<number | null>(null);
    const lastMatchCount = useRef(0);

    const base = getApiBase();
    const host = searchParams.get('host');
    const hostQuery = host ? `?host=${encodeURIComponent(host)}` : '?';

    const fetchData = useCallback(async () => {
        try {
            const [statusRes, statsRes] = await Promise.all([
                fetch(`${base}/api/evolution/status`),
                fetch(`${base}/api/qlearning/status`),
            ]);

            if (statusRes.ok) setStatus(await statusRes.json());

            if (statsRes.ok) {
                const s: TrainingStats = await statsRes.json();
                setStats(s);
                if (s.total_matches > lastMatchCount.current) {
                    lastMatchCount.current = s.total_matches;
                    setHistory(prev => {
                        const next = [...prev, s];
                        if (next.length > 500) next.shift();
                        return next;
                    });
                }
                setError(null);
            } else if (statsRes.status === 404) {
                setStats(null);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    }, [base]);

    useEffect(() => {
        fetchData();
        pollRef.current = window.setInterval(fetchData, POLL_INTERVAL);
        return () => {
            if (pollRef.current !== null) clearInterval(pollRef.current);
        };
    }, [fetchData]);

    const handleRunMatch = () => {
        const ids = [1, 2, 3, 4].join(',');
        navigate(`/match${hostQuery.replace('?', '?qlearning_match=' + ids + '&')}`);
    };

    const running = status?.running ?? false;

    return (
        <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            fontFamily: 'monospace',
            fontSize: 13,
            color: '#ccc',
            overflow: 'hidden',
        }}>
            <div style={{flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 16}}>
                {error && (
                    <div style={{color: '#c44', padding: '8px 12px', background: '#2a1111', border: '1px solid #522', borderRadius: 4}}>
                        Error: {error}
                    </div>
                )}

                {/* Header */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 16,
                    padding: '12px 14px',
                    background: '#1a1a1a',
                    borderRadius: 4,
                    border: '1px solid #333',
                    flexWrap: 'wrap',
                }}>
                    <span style={{
                        color: running ? '#4c4' : '#888',
                        fontWeight: 'bold',
                    }}>
                        {running ? 'Training' : 'Idle'}
                    </span>
                    {stats && (
                        <>
                            <Stat label="Matches" value={stats.total_matches}/>
                            <Stat label="Epsilon" value={stats.epsilon.toFixed(3)}/>
                            <Stat label="Replay" value={stats.replay_size.toLocaleString()}/>
                            <Stat label="Train Steps" value={stats.total_train_steps.toLocaleString()}/>
                        </>
                    )}
                    <button
                        onClick={handleRunMatch}
                        style={btnStyle('#1a3a2a', '#2a6a4a')}
                    >
                        Watch Match
                    </button>
                </div>

                {/* Stats Cards */}
                {stats && (
                    <div style={{display: 'flex', gap: 12, flexWrap: 'wrap'}}>
                        <StatCard label="Avg Reward (100)" value={stats.avg_reward_100.toFixed(3)} color="#4c8"/>
                        <StatCard label="Avg Cells (100)" value={stats.avg_cells_100.toFixed(1)} color="#48f"/>
                        <StatCard label="Loss" value={stats.last_loss.toFixed(6)} color="#cc8"/>
                        <StatCard label="Epsilon" value={stats.epsilon.toFixed(4)} color="#c8c"/>
                    </div>
                )}

                {/* Charts */}
                {history.length > 1 && (
                    <div style={{
                        padding: '12px 14px',
                        background: '#1a1a1a',
                        borderRadius: 4,
                        border: '1px solid #333',
                    }}>
                        <div style={{color: '#888', fontSize: 11, textTransform: 'uppercase', marginBottom: 8, letterSpacing: 1}}>
                            Training Progress
                        </div>
                        <div style={{display: 'flex', gap: 16, flexWrap: 'wrap'}}>
                            <div style={{flex: 1, minWidth: 300}}>
                                <TrainingChart
                                    data={history}
                                    lines={[
                                        {key: 'avg_reward_100', color: '#4c8', label: 'Avg Reward'},
                                    ]}
                                    height={140}
                                />
                            </div>
                            <div style={{flex: 1, minWidth: 300}}>
                                <TrainingChart
                                    data={history}
                                    lines={[
                                        {key: 'avg_cells_100', color: '#48f', label: 'Avg Cells'},
                                    ]}
                                    height={140}
                                />
                            </div>
                            <div style={{flex: 1, minWidth: 300}}>
                                <TrainingChart
                                    data={history}
                                    lines={[
                                        {key: 'last_loss', color: '#cc8', label: 'Loss'},
                                    ]}
                                    height={140}
                                />
                            </div>
                            <div style={{flex: 1, minWidth: 300}}>
                                <TrainingChart
                                    data={history}
                                    lines={[
                                        {key: 'epsilon', color: '#c8c', label: 'Epsilon'},
                                    ]}
                                    height={140}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* Log */}
                {status && status.log.length > 0 && (
                    <div style={{
                        padding: '12px 14px',
                        background: '#1a1a1a',
                        borderRadius: 4,
                        border: '1px solid #333',
                    }}>
                        <div style={{color: '#888', fontSize: 11, textTransform: 'uppercase', marginBottom: 8, letterSpacing: 1}}>
                            Log
                        </div>
                        <pre style={{
                            margin: 0,
                            padding: 10,
                            background: '#0a0a0a',
                            border: '1px solid #282828',
                            borderRadius: 4,
                            maxHeight: 200,
                            overflowY: 'auto',
                            fontSize: 11,
                            color: '#999',
                            whiteSpace: 'pre-wrap',
                        }}>
                            {status.log.slice(-50).join('\n')}
                        </pre>
                    </div>
                )}

                {!stats && !error && (
                    <div style={{color: '#666', textAlign: 'center', padding: 40}}>
                        No Q-learning job running. Submit one from the Queue page.
                    </div>
                )}
            </div>
        </div>
    );
}


function Stat({label, value}: {label: string; value: string | number}) {
    return (
        <span style={{color: '#999', fontSize: 12}}>
            {label}: <span style={{color: '#ccc'}}>{String(value)}</span>
        </span>
    );
}

function StatCard({label, value, color}: {label: string; value: string; color: string}) {
    return (
        <div style={{
            padding: '10px 16px',
            background: '#1a1a1a',
            borderRadius: 4,
            border: '1px solid #333',
            minWidth: 140,
        }}>
            <div style={{color: '#888', fontSize: 10, textTransform: 'uppercase', marginBottom: 4}}>{label}</div>
            <div style={{color, fontSize: 20, fontWeight: 'bold'}}>{value}</div>
        </div>
    );
}

interface LineConfig {
    key: string;
    color: string;
    label: string;
}

function TrainingChart({data, lines, height = 120}: {
    data: TrainingStats[];
    lines: LineConfig[];
    height?: number;
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || data.length < 2) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const pad = {top: 10, right: 10, bottom: 16, left: 50};
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);

        const allVals = lines.flatMap(l => data.map(d => (d as unknown as Record<string, number>)[l.key]));
        const minV = Math.min(...allVals);
        const maxV = Math.max(...allVals);
        const rangeV = maxV - minV || 1;

        const toX = (i: number) => pad.left + (i / Math.max(data.length - 1, 1)) * plotW;
        const toY = (v: number) => pad.top + (1 - (v - minV) / rangeV) * plotH;

        // Axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, h - pad.bottom);
        ctx.lineTo(w - pad.right, h - pad.bottom);
        ctx.stroke();

        // Y labels
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        const topLabel = maxV >= 100 ? Math.round(maxV).toString() : maxV.toFixed(2);
        const botLabel = minV >= 100 ? Math.round(minV).toString() : minV.toFixed(2);
        ctx.fillText(topLabel, pad.left - 4, pad.top + 4);
        ctx.fillText(botLabel, pad.left - 4, h - pad.bottom);

        // Lines
        for (const line of lines) {
            const values = data.map(d => (d as unknown as Record<string, number>)[line.key]);
            ctx.strokeStyle = line.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < values.length; i++) {
                const x = toX(i);
                const y = toY(values[i]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Legend
        let legendX = pad.left + 8;
        for (const line of lines) {
            ctx.fillStyle = line.color;
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(line.label, legendX, h - 2);
            legendX += ctx.measureText(line.label).width + 12;
        }
    }, [data, lines]);

    return <canvas ref={canvasRef} style={{width: '100%', height}}/>;
}

function btnStyle(bg: string, border: string): React.CSSProperties {
    return {
        padding: '4px 14px',
        background: bg,
        color: '#ccc',
        border: `1px solid ${border}`,
        borderRadius: 4,
        cursor: 'pointer',
        fontFamily: 'monospace',
        fontSize: 12,
    };
}
