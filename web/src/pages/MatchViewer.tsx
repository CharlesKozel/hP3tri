import {useCallback, useEffect, useRef, useState} from 'react';
import {useSearchParams} from 'react-router-dom';
import HexGridCanvas from '../components/HexGridCanvas';
import StatsPanel from '../components/StatsPanel';
import {getApiBase} from '../api';
import type {CellTypeInfo, ReplayInfo, SimulationState} from '../types';

export default function MatchViewer() {
    const [searchParams, setSearchParams] = useSearchParams();
    const [replay, setReplay] = useState<SimulationState[] | null>(null);
    const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
    const [cellTypes, setCellTypes] = useState<CellTypeInfo[]>([]);
    const [currentTick, setCurrentTick] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(200);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [statsOpen, setStatsOpen] = useState(true);
    const intervalRef = useRef<number | null>(null);
    const [replayList, setReplayList] = useState<{gen: number; matches: {matchIndex: number; genomeIds: number[]; totalTicks: number}[]}[]>([]);
    const [replayListOpen, setReplayListOpen] = useState(true);

    const fetchReplay = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const base = getApiBase();
            const runId = searchParams.get('run');
            const gen = searchParams.get('gen');
            const matchIdx = searchParams.get('match');
            const genomes = searchParams.get('genomes');
            const tournamentMatch = searchParams.get('tournament_match');
            const qlearningMatch = searchParams.get('qlearning_match');

            let replayPromise: Promise<Response>;
            let infoPromise: Promise<Response>;

            if (qlearningMatch) {
                const genomeIds = qlearningMatch.split(',').map(Number).filter(n => !isNaN(n));
                replayPromise = fetch(`${base}/api/qlearning/run-match`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({genomeIds, gridWidth: 64, gridHeight: 64, tickLimit: 200}),
                });
                infoPromise = replayPromise.then(res => res.clone());
            } else if (runId && gen !== null && matchIdx !== null) {
                const url = `${base}/api/queue/runs/${encodeURIComponent(runId)}/replays/${gen}/${matchIdx}`;
                replayPromise = fetch(url);
                infoPromise = replayPromise.then(res => res.clone());
            } else if (tournamentMatch) {
                const genomeIds = tournamentMatch.split(',').map(Number).filter(n => !isNaN(n));
                replayPromise = fetch(`${base}/api/tournament/run-match`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({genomeIds, gridWidth: 64, gridHeight: 64, tickLimit: 200}),
                });
                infoPromise = replayPromise.then(res => res.clone());
            } else if (genomes) {
                const genomeIds = genomes.split(',').map(Number).filter(n => !isNaN(n));
                replayPromise = fetch(`${base}/api/evolution/run-match`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({genomeIds, gridWidth: 64, gridHeight: 64, tickLimit: 200}),
                });
                infoPromise = replayPromise.then(res => res.clone());
            } else {
                replayPromise = fetch(`${base}/api/replay`);
                infoPromise = fetch(`${base}/api/replay/info`);
            }

            const cellTypesRes = await fetch(`${base}/api/cell-types`);
            if (!cellTypesRes.ok) throw new Error(`HTTP ${cellTypesRes.status}`);
            const types: CellTypeInfo[] = await cellTypesRes.json();
            setCellTypes(types);

            if (runId || genomes || tournamentMatch || qlearningMatch) {
                const replayRes = await replayPromise;
                if (!replayRes.ok) throw new Error(`HTTP ${replayRes.status}`);
                const data = await replayRes.json();
                const frames: SimulationState[] = data.frames;
                setReplay(frames);
                setReplayInfo({
                    totalTicks: frames[frames.length - 1]?.tick ?? frames.length - 1,
                    width: frames[0]?.grid?.width ?? 64,
                    height: frames[0]?.grid?.height ?? 64,
                });
            } else {
                const [infoRes, replayRes] = await Promise.all([infoPromise, replayPromise]);
                if (!infoRes.ok) throw new Error(`HTTP ${infoRes.status}`);
                if (!replayRes.ok) throw new Error(`HTTP ${replayRes.status}`);
                const info: ReplayInfo = await infoRes.json();
                setReplayInfo(info);
                const frames: SimulationState[] = await replayRes.json();
                setReplay(frames);
            }

            setCurrentTick(0);
            setPlaying(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    }, [searchParams]);

    useEffect(() => {
        fetchReplay();
    }, [fetchReplay]);

    useEffect(() => {
        if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (playing && replay) {
            intervalRef.current = window.setInterval(() => {
                setCurrentTick((prev) => {
                    if (prev >= replay.length - 1) {
                        setPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, playbackSpeed);
        }
        return () => {
            if (intervalRef.current !== null) {
                clearInterval(intervalRef.current);
            }
        };
    }, [playing, playbackSpeed, replay]);

    useEffect(() => {
        const runId = searchParams.get('run');
        if (!runId) return;
        const base = getApiBase();
        (async () => {
            try {
                const gensRes = await fetch(`${base}/api/queue/runs/${encodeURIComponent(runId)}/replays`);
                if (!gensRes.ok) return;
                const gens: number[] = await gensRes.json();
                const allGens: typeof replayList = [];
                for (const gen of gens) {
                    const indexRes = await fetch(`${base}/api/queue/runs/${encodeURIComponent(runId)}/replays/${gen}`);
                    if (indexRes.ok) {
                        const index = await indexRes.json();
                        allGens.push({ gen, matches: index.matches });
                    }
                }
                setReplayList(allGens);
            } catch { /* ignore */ }
        })();
    }, [searchParams]);

    const navigateToReplay = (gen: number, matchIdx: number) => {
        const params = new URLSearchParams(searchParams);
        params.set('gen', String(gen));
        params.set('match', String(matchIdx));
        setSearchParams(params);
    };

    const handleReset = async () => {
        setPlaying(false);
        setLoading(true);
        try {
            const res = await fetch(`${getApiBase()}/api/simulation/reset`, {method: 'POST'});
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            await fetchReplay();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
            setLoading(false);
        }
    };

    const handleStepBack = () => {
        setPlaying(false);
        setCurrentTick((prev) => Math.max(0, prev - 1));
    };

    const handleStepForward = () => {
        if (!replay) return;
        setPlaying(false);
        setCurrentTick((prev) => Math.min(replay.length - 1, prev + 1));
    };

    if (error && !replay) {
        return (
            <div style={{padding: 20, color: '#ff4444', fontFamily: 'monospace'}}>
                Failed to load replay: {error}
                <br/>
                Make sure the Kotlin server is running on port 8080.
            </div>
        );
    }

    if (loading && !replay) {
        return (
            <div style={{padding: 20, color: '#888', fontFamily: 'monospace'}}>
                Loading simulation...
            </div>
        );
    }

    if (!replay || !replayInfo) return null;

    const frame = replay[currentTick];

    return (
        <div style={{flex: 1, display: 'flex', minHeight: 0}}>
            <div style={{flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column'}}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 16,
                    padding: '8px 16px',
                    background: '#1a1a1a',
                    borderBottom: '1px solid #333',
                    fontFamily: 'monospace',
                    fontSize: 13,
                    color: '#ccc',
                    flexShrink: 0,
                    flexWrap: 'wrap',
                }}>
                    <span>Tick: {frame?.tick ?? currentTick}/{replayInfo.totalTicks}</span>
                    <span style={{color: frame?.status === 'RUNNING' ? '#4c4' : '#c44'}}>
                        {frame?.status}
                    </span>
                    <div style={{display: 'flex', gap: 4}}>
                        <button onClick={handleStepBack} style={btnStyle} title="Step back">
                            {'<'}
                        </button>
                        <button
                            onClick={() => setPlaying(!playing)}
                            style={{...btnStyle, width: 48}}
                        >
                            {playing ? 'II' : '>'}
                        </button>
                        <button onClick={handleStepForward} style={btnStyle} title="Step forward">
                            {'>'}
                        </button>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={replay.length - 1}
                        value={currentTick}
                        onChange={(e) => {
                            setPlaying(false);
                            setCurrentTick(Number(e.target.value));
                        }}
                        style={{flex: 1, minWidth: 100}}
                    />
                    <label style={{display: 'flex', alignItems: 'center', gap: 8}}>
                        Speed:
                        <input
                            type="range"
                            min={50}
                            max={1000}
                            step={50}
                            value={playbackSpeed}
                            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                            style={{width: 80}}
                        />
                        <span style={{width: 45}}>{playbackSpeed}ms</span>
                    </label>
                    <button onClick={handleReset} style={btnStyle}>
                        Reset
                    </button>
                    <button onClick={() => setStatsOpen(!statsOpen)} style={btnStyle} title="Stats">
                        {'\u2630'}
                    </button>
                </div>
                <div style={{flex: 1, minHeight: 0}}>
                    {frame && <HexGridCanvas grid={frame.grid} organisms={frame.organisms} cellTypes={cellTypes}/>}
                </div>
            </div>
            {/* Sidebar */}
            <div style={{
                width: statsOpen ? 360 : 0,
                minWidth: statsOpen ? 360 : 0,
                height: '100vh',
                background: '#1a1a1a',
                borderLeft: statsOpen ? '1px solid #333' : 'none',
                transition: 'width 0.3s ease, min-width 0.3s ease',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                flexShrink: 0,
            }}>
                <div style={{minWidth: 360, display: 'flex', flexDirection: 'column', height: '100%'}}>
                    {/* Replay Browser */}
                    {replayList.length > 0 && (
                        <div style={{
                            borderBottom: '1px solid #333',
                            maxHeight: replayListOpen ? 300 : 36,
                            overflow: 'hidden',
                            transition: 'max-height 0.2s ease',
                            flexShrink: 0,
                        }}>
                            <div
                                onClick={() => setReplayListOpen(!replayListOpen)}
                                style={{
                                    padding: '8px 16px',
                                    cursor: 'pointer',
                                    fontFamily: 'monospace',
                                    fontSize: 12,
                                    color: '#888',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                }}
                            >
                                <span>Replays ({replayList.reduce((sum, g) => sum + g.matches.length, 0)})</span>
                                <span>{replayListOpen ? '\u25BC' : '\u25B6'}</span>
                            </div>
                            <div style={{overflowY: 'auto', maxHeight: 260, padding: '0 16px 8px'}}>
                                {replayList.map(genGroup => (
                                    <div key={genGroup.gen} style={{marginBottom: 8}}>
                                        <div style={{color: '#666', fontSize: 11, marginBottom: 4, fontFamily: 'monospace'}}>
                                            Gen {genGroup.gen}
                                        </div>
                                        {genGroup.matches.map(m => {
                                            const currentGen = searchParams.get('gen');
                                            const currentMatch = searchParams.get('match');
                                            const isActive = currentGen === String(genGroup.gen) && currentMatch === String(m.matchIndex);
                                            return (
                                                <div
                                                    key={m.matchIndex}
                                                    onClick={() => navigateToReplay(genGroup.gen, m.matchIndex)}
                                                    style={{
                                                        padding: '4px 8px',
                                                        cursor: 'pointer',
                                                        borderRadius: 3,
                                                        background: isActive ? '#2a4a6a' : 'transparent',
                                                        border: isActive ? '1px solid #4a7aaa' : '1px solid transparent',
                                                        fontFamily: 'monospace',
                                                        fontSize: 12,
                                                        color: isActive ? '#eee' : '#aaa',
                                                        marginBottom: 2,
                                                    }}
                                                >
                                                    {m.matchIndex >= 10000 ? `Top #${m.matchIndex - 10000}` : `Match ${m.matchIndex}`} — {m.genomeIds.join(' vs ')} ({m.totalTicks}t)
                                                </div>
                                            );
                                        })}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    {/* Stats content */}
                    <div style={{flex: 1, overflow: 'hidden'}}>
                        <StatsPanel
                            organisms={frame?.organisms ?? []}
                            currentTick={frame?.tick ?? currentTick}
                            totalTicks={replayInfo.totalTicks}
                            open={true}
                            onClose={() => setStatsOpen(false)}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

const btnStyle: React.CSSProperties = {
    padding: '4px 12px',
    background: '#333',
    color: '#ccc',
    border: '1px solid #555',
    borderRadius: 4,
    cursor: 'pointer',
    fontFamily: 'monospace',
    fontSize: 13,
};
