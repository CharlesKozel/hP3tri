import {useCallback, useEffect, useRef, useState} from 'react';
import {useNavigate, useSearchParams} from 'react-router-dom';
import OrganismThumbnail from '../components/OrganismThumbnail';
import {getApiBase} from '../api';
import type {CellTypeInfo, EloHistoryEntry, EloLeaderboardEntry, EvolutionStatus, SimulationState} from '../types';

const POLL_INTERVAL = 2000;

export default function TournamentDashboard() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const [status, setStatus] = useState<EvolutionStatus | null>(null);
    const [leaderboard, setLeaderboard] = useState<EloLeaderboardEntry[]>([]);
    const [history, setHistory] = useState<EloHistoryEntry[]>([]);
    const [cellTypes, setCellTypes] = useState<CellTypeInfo[]>([]);
    const [previews, setPreviews] = useState<Map<number, SimulationState>>(new Map());
    const [selectedIds, setSelectedIds] = useState<number[]>([]);
    const [error, setError] = useState<string | null>(null);
    const pollRef = useRef<number | null>(null);
    const lastGenRef = useRef(-1);

    const base = getApiBase();
    const host = searchParams.get('host');
    const hostQuery = host ? `?host=${encodeURIComponent(host)}` : '';

    useEffect(() => {
        fetch(`${base}/api/cell-types`)
            .then(r => r.json())
            .then(setCellTypes)
            .catch(() => {});
    }, [base]);

    const fetchData = useCallback(async () => {
        try {
            const [statusRes, lbRes] = await Promise.all([
                fetch(`${base}/api/evolution/status`),
                fetch(`${base}/api/tournament/leaderboard`),
            ]);

            if (statusRes.ok) {
                const s: EvolutionStatus = await statusRes.json();
                setStatus(s);
                setError(null);

                if (s.generation !== lastGenRef.current) {
                    lastGenRef.current = s.generation;
                    const histRes = await fetch(`${base}/api/tournament/history`);
                    if (histRes.ok) setHistory(await histRes.json());
                }
            }

            if (lbRes.ok) {
                const entries: EloLeaderboardEntry[] = await lbRes.json();
                setLeaderboard(entries);

                const newPreviews = new Map(previews);
                const toFetch = entries.filter(e => !newPreviews.has(e.genomeId)).slice(0, 10);
                for (const entry of toFetch) {
                    try {
                        const res = await fetch(`${base}/api/tournament/preview/${entry.genomeId}`);
                        if (res.ok) {
                            const snap: SimulationState = await res.json();
                            newPreviews.set(entry.genomeId, snap);
                        }
                    } catch {
                        // preview not ready yet
                    }
                }
                if (toFetch.length > 0) setPreviews(newPreviews);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    }, [base, previews]);

    useEffect(() => {
        fetchData();
        pollRef.current = window.setInterval(fetchData, POLL_INTERVAL);
        return () => {
            if (pollRef.current !== null) clearInterval(pollRef.current);
        };
    }, [fetchData]);

    const toggleSelected = (id: number) => {
        setSelectedIds(prev => {
            if (prev.includes(id)) return prev.filter(x => x !== id);
            if (prev.length >= 2) return [prev[1], id];
            return [...prev, id];
        });
    };

    const handleFight = () => {
        if (selectedIds.length !== 2) return;
        navigate(`/match${hostQuery}${hostQuery ? '&' : '?'}tournament_match=${selectedIds.join(',')}`);
    };

    const running = status?.running ?? false;

    const SYMMETRY_NAMES: Record<number, string> = {
        0: 'Bilateral',
        1: 'Rad-2',
        2: 'Rad-3',
        3: 'Rad-4',
        4: 'Rad-5',
        5: 'Rad-6',
        6: 'Asymmetric',
    };

    return (
        <div style={{
            flex: 1,
            display: 'flex',
            fontFamily: 'monospace',
            fontSize: 13,
            color: '#ccc',
            overflow: 'hidden',
        }}>
            {/* Main content */}
            <div style={{flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden'}}>
                {/* Status bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '10px 14px',
                    background: '#1a1a1a',
                    borderBottom: '1px solid #333',
                    flexWrap: 'wrap',
                    flexShrink: 0,
                }}>
                    <span style={{color: running ? '#4c4' : '#888'}}>
                        {running ? 'Running' : 'Idle'}
                    </span>
                    {status && (
                        <>
                            <Stat label="Gen" value={`${status.generation}/${status.totalGenerations}`}/>
                            <Stat label="Top ELO" value={Math.round(status.bestFitness)}/>
                            <Stat label="Matches" value={status.matchesCompleted}/>
                            <Stat label="Population" value={leaderboard.length}/>
                        </>
                    )}
                    {selectedIds.length === 2 && (
                        <button onClick={handleFight} style={btnStyle('#1a3a2a', '#2a6a4a')}>
                            Fight: {selectedIds[0]} vs {selectedIds[1]}
                        </button>
                    )}
                    {selectedIds.length > 0 && (
                        <button onClick={() => setSelectedIds([])} style={btnStyle('#3a1a1a', '#5a2a2a')}>
                            Clear
                        </button>
                    )}
                    {error && <span style={{color: '#c44', fontSize: 11}}>Error: {error}</span>}
                </div>

                {/* Organism gallery */}
                <div style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: 12,
                }}>
                    <div style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: 6,
                        justifyContent: 'flex-start',
                    }}>
                        {leaderboard.map((entry, i) => (
                            <OrganismThumbnail
                                key={entry.genomeId}
                                snapshot={previews.get(entry.genomeId) ?? null}
                                cellTypes={cellTypes}
                                size={80}
                                selected={selectedIds.includes(entry.genomeId)}
                                elo={entry.elo}
                                rank={i + 1}
                                genomeId={entry.genomeId}
                                onClick={() => toggleSelected(entry.genomeId)}
                            />
                        ))}
                        {leaderboard.length === 0 && (
                            <div style={{color: '#666', padding: 20}}>
                                No tournament data. Submit a tournament job from the Queue page.
                            </div>
                        )}
                    </div>

                    {/* ELO History Chart */}
                    {history.length > 0 && (
                        <div style={{marginTop: 16, padding: '12px 14px', background: '#1a1a1a', borderRadius: 4, border: '1px solid #333'}}>
                            <div style={{color: '#888', fontSize: 11, textTransform: 'uppercase', marginBottom: 8, letterSpacing: 1}}>
                                ELO History
                            </div>
                            <EloChart history={history}/>
                        </div>
                    )}
                </div>
            </div>

            {/* Sidebar: leaderboard table */}
            <div style={{
                width: 320,
                minWidth: 320,
                borderLeft: '1px solid #333',
                background: '#151515',
                overflowY: 'auto',
                padding: 12,
            }}>
                <div style={{color: '#888', fontSize: 11, textTransform: 'uppercase', marginBottom: 10, letterSpacing: 1}}>
                    Leaderboard
                </div>
                <table style={{width: '100%', borderCollapse: 'collapse'}}>
                    <thead>
                        <tr style={{borderBottom: '1px solid #333', color: '#888', textAlign: 'left'}}>
                            <th style={thStyle}>#</th>
                            <th style={thStyle}>ID</th>
                            <th style={thStyle}>ELO</th>
                            <th style={thStyle}>W/L/D</th>
                            <th style={thStyle}>Cells</th>
                            <th style={thStyle}>Sym</th>
                        </tr>
                    </thead>
                    <tbody>
                        {leaderboard.map((entry, i) => (
                            <tr
                                key={entry.genomeId}
                                onClick={() => toggleSelected(entry.genomeId)}
                                style={{
                                    borderBottom: '1px solid #222',
                                    cursor: 'pointer',
                                    background: selectedIds.includes(entry.genomeId) ? '#1a2a1a' : 'transparent',
                                }}
                            >
                                <td style={tdStyle}>{i + 1}</td>
                                <td style={tdStyle}>{entry.genomeId}</td>
                                <td style={{...tdStyle, color: entry.elo >= 1100 ? '#4c8' : entry.elo <= 900 ? '#c44' : '#cc8'}}>
                                    {Math.round(entry.elo)}
                                </td>
                                <td style={tdStyle}>{entry.wins}/{entry.losses}/{entry.draws}</td>
                                <td style={tdStyle}>{entry.previewCellCount}</td>
                                <td style={{...tdStyle, fontSize: 10}}>{SYMMETRY_NAMES[entry.symmetryMode] ?? '?'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function EloChart({history}: {history: EloHistoryEntry[]}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || history.length === 0) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const pad = {top: 10, right: 10, bottom: 20, left: 40};
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);

        const allVals = history.flatMap(e => [e.topElo, e.avgElo, e.medianElo]);
        const minV = Math.min(...allVals) - 20;
        const maxV = Math.max(...allVals) + 20;
        const rangeV = maxV - minV || 1;

        const toX = (i: number) => pad.left + (i / Math.max(history.length - 1, 1)) * plotW;
        const toY = (v: number) => pad.top + (1 - (v - minV) / rangeV) * plotH;

        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, h - pad.bottom);
        ctx.lineTo(w - pad.right, h - pad.bottom);
        ctx.stroke();

        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(maxV).toString(), pad.left - 4, pad.top + 4);
        ctx.fillText(Math.round(minV).toString(), pad.left - 4, h - pad.bottom);

        const lines: {data: number[]; color: string; label: string}[] = [
            {data: history.map(e => e.topElo), color: '#4c8', label: 'Top'},
            {data: history.map(e => e.avgElo), color: '#cc8', label: 'Avg'},
            {data: history.map(e => e.medianElo), color: '#888', label: 'Median'},
        ];

        for (const line of lines) {
            ctx.strokeStyle = line.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < line.data.length; i++) {
                const x = toX(i);
                const y = toY(line.data[i]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        let legendX = pad.left + 8;
        for (const line of lines) {
            ctx.fillStyle = line.color;
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(line.label, legendX, h - 4);
            legendX += ctx.measureText(line.label).width + 12;
        }
    }, [history]);

    return (
        <canvas
            ref={canvasRef}
            style={{width: '100%', height: 120}}
        />
    );
}

function Stat({label, value}: {label: string; value: string | number}) {
    return (
        <span style={{color: '#999', fontSize: 12}}>
            {label}: <span style={{color: '#ccc'}}>{String(value)}</span>
        </span>
    );
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

const thStyle: React.CSSProperties = {padding: '4px 6px', fontSize: 11};
const tdStyle: React.CSSProperties = {padding: '4px 6px', fontSize: 11};
