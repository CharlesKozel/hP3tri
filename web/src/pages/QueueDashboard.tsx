import {useCallback, useEffect, useRef, useState} from 'react';
import {useSearchParams} from 'react-router-dom';
import {getApiBase} from '../api';
import type {ReplayIndex} from '../types';

const POLL_INTERVAL = 3000;

interface PendingJob {
    filename: string;
    config: {
        name: string;
        description: string;
        priority: number;
        evolution: Record<string, unknown>;
    };
}

interface RunStatus {
    jobName: string;
    state: string;
    generation: number;
    totalGenerations: number;
    bestFitness: number;
    archiveFillRate: number;
    matchesCompleted: number;
    startedAt: string;
    updatedAt: string;
    error: string | null;
    hasReplays: boolean;
}

interface RunSummary {
    runId: string;
    status: RunStatus;
}

interface CurrentRun {
    runId: string;
    jobName: string;
    state: string;
    generation: number;
    totalGenerations: number;
    bestFitness: number;
    archiveFillRate: number;
    matchesCompleted: number;
}

interface JobForm {
    name: string;
    description: string;
    priority: number;
    populationSize: number;
    generations: number;
    matchesPerGeneration: number;
    genomesPerMatch: number;
    gridWidth: number;
    gridHeight: number;
    tickLimit: number;
    foodCount: number;
    foodRespawnRate: number;
    seed: number;
    showcaseInterval: number;
}

const DEFAULT_FORM: JobForm = {
    name: '',
    description: '',
    priority: 0,
    populationSize: 100,
    generations: 50,
    matchesPerGeneration: 200,
    genomesPerMatch: 3,
    gridWidth: 64,
    gridHeight: 64,
    tickLimit: 500,
    foodCount: 80,
    foodRespawnRate: 5,
    seed: 42,
    showcaseInterval: 5,
};

export default function QueueDashboard() {
    const [searchParams] = useSearchParams();
    const [pending, setPending] = useState<PendingJob[]>([]);
    const [runs, setRuns] = useState<RunSummary[]>([]);
    const [current, setCurrent] = useState<CurrentRun | null>(null);
    const [form, setForm] = useState<JobForm>({...DEFAULT_FORM});
    const [error, setError] = useState<string | null>(null);
    const [expandedLog, setExpandedLog] = useState<string | null>(null);
    const [logText, setLogText] = useState('');
    const [expandedReplays, setExpandedReplays] = useState<string | null>(null);
    const [replayGens, setReplayGens] = useState<number[]>([]);
    const [selectedGen, setSelectedGen] = useState<number | null>(null);
    const [replayIndex, setReplayIndex] = useState<ReplayIndex | null>(null);
    const pollRef = useRef<number | null>(null);

    const base = getApiBase();
    const host = searchParams.get('host');
    const hostQuery = host ? `&host=${encodeURIComponent(host)}` : '';

    const fetchAll = useCallback(async () => {
        try {
            const [pendingRes, runsRes, currentRes] = await Promise.all([
                fetch(`${base}/api/queue/pending`),
                fetch(`${base}/api/queue/runs`),
                fetch(`${base}/api/queue/current`),
            ]);
            if (pendingRes.ok) setPending(await pendingRes.json());
            if (runsRes.ok) setRuns(await runsRes.json());
            if (currentRes.ok && currentRes.status !== 204) {
                setCurrent(await currentRes.json());
            } else {
                setCurrent(null);
            }
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    }, [base]);

    useEffect(() => {
        fetchAll();
        pollRef.current = window.setInterval(fetchAll, POLL_INTERVAL);
        return () => {
            if (pollRef.current !== null) clearInterval(pollRef.current);
        };
    }, [fetchAll]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!form.name.trim()) {
            setError('Job name is required');
            return;
        }
        try {
            const body = {
                name: form.name.trim(),
                description: form.description,
                priority: form.priority,
                evolution: {
                    populationSize: form.populationSize,
                    generations: form.generations,
                    matchesPerGeneration: form.matchesPerGeneration,
                    genomesPerMatch: form.genomesPerMatch,
                    gridWidth: form.gridWidth,
                    gridHeight: form.gridHeight,
                    tickLimit: form.tickLimit,
                    foodCount: form.foodCount,
                    foodRespawnRate: form.foodRespawnRate,
                    seed: form.seed,
                    showcaseInterval: form.showcaseInterval,
                },
            };
            const res = await fetch(`${base}/api/queue/submit`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            setForm({...DEFAULT_FORM});
            setError(null);
            await fetchAll();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const handleDeletePending = async (filename: string) => {
        try {
            await fetch(`${base}/api/queue/pending/${encodeURIComponent(filename)}`, {method: 'DELETE'});
            await fetchAll();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const handlePause = async () => {
        try {
            await fetch(`${base}/api/queue/pause`, {method: 'POST'});
            await fetchAll();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const handleCancel = async () => {
        try {
            await fetch(`${base}/api/queue/cancel`, {method: 'POST'});
            await fetchAll();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const handleViewLog = async (runId: string) => {
        if (expandedLog === runId) {
            setExpandedLog(null);
            return;
        }
        try {
            const res = await fetch(`${base}/api/queue/runs/${encodeURIComponent(runId)}/log`);
            if (res.ok) setLogText(await res.text());
            else setLogText('(no log available)');
            setExpandedLog(runId);
        } catch {
            setLogText('(error loading log)');
            setExpandedLog(runId);
        }
    };

    const handleViewReplays = async (runId: string) => {
        if (expandedReplays === runId) {
            setExpandedReplays(null);
            setSelectedGen(null);
            setReplayIndex(null);
            return;
        }
        try {
            const res = await fetch(`${base}/api/queue/runs/${encodeURIComponent(runId)}/replays`);
            if (res.ok) {
                const gens: number[] = await res.json();
                setReplayGens(gens);
                setExpandedReplays(runId);
                setSelectedGen(null);
                setReplayIndex(null);
            }
        } catch {
            setReplayGens([]);
            setExpandedReplays(runId);
        }
    };

    const handleSelectGen = async (runId: string, gen: number) => {
        if (selectedGen === gen) {
            setSelectedGen(null);
            setReplayIndex(null);
            return;
        }
        try {
            const res = await fetch(`${base}/api/queue/runs/${encodeURIComponent(runId)}/replays/${gen}`);
            if (res.ok) {
                const index: ReplayIndex = await res.json();
                setReplayIndex(index);
                setSelectedGen(gen);
            }
        } catch {
            setReplayIndex(null);
            setSelectedGen(gen);
        }
    };

    const updateField = (field: keyof JobForm, value: string | number) => {
        setForm(prev => ({...prev, [field]: value}));
    };

    return (
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', fontFamily: 'monospace', fontSize: 13, color: '#ccc', overflow: 'hidden'}}>
            <div style={{flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 20}}>
                {error && <div style={{color: '#c44', padding: '8px 12px', background: '#2a1111', border: '1px solid #522', borderRadius: 4}}>Error: {error}</div>}

                {/* Current Run */}
                {current && (
                    <Section title="Currently Running">
                        <div style={{display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap'}}>
                            <span style={{color: '#4c4'}}>Running</span>
                            <Stat label="Job" value={current.jobName}/>
                            <Stat label="Gen" value={`${current.generation}/${current.totalGenerations}`}/>
                            <Stat label="Best" value={current.bestFitness.toFixed(1)}/>
                            <Stat label="Archive" value={`${(current.archiveFillRate * 100).toFixed(0)}%`}/>
                            <Stat label="Matches" value={current.matchesCompleted}/>
                            <button onClick={handlePause} style={btnStyle('#5a4a1a', '#8a7a2a')}>Pause</button>
                            <button onClick={handleCancel} style={btnStyle('#5a1a1a', '#8a2a2a')}>Cancel</button>
                        </div>
                    </Section>
                )}

                {/* Submit New Run */}
                <Section title="Submit New Run">
                    <form onSubmit={handleSubmit} style={{display: 'flex', flexDirection: 'column', gap: 10}}>
                        <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                            <Field label="Name" value={form.name} onChange={v => updateField('name', v)} width={200}/>
                            <Field label="Priority" value={form.priority} onChange={v => updateField('priority', parseInt(v) || 0)} width={60} type="number"/>
                            <Field label="Description" value={form.description} onChange={v => updateField('description', v)} width={300}/>
                        </div>
                        <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                            <Field label="Pop Size" value={form.populationSize} onChange={v => updateField('populationSize', parseInt(v) || 0)} width={70} type="number"/>
                            <Field label="Generations" value={form.generations} onChange={v => updateField('generations', parseInt(v) || 0)} width={70} type="number"/>
                            <Field label="Matches/Gen" value={form.matchesPerGeneration} onChange={v => updateField('matchesPerGeneration', parseInt(v) || 0)} width={80} type="number"/>
                            <Field label="Genomes/Match" value={form.genomesPerMatch} onChange={v => updateField('genomesPerMatch', parseInt(v) || 0)} width={80} type="number"/>
                        </div>
                        <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                            <Field label="Grid W" value={form.gridWidth} onChange={v => updateField('gridWidth', parseInt(v) || 0)} width={60} type="number"/>
                            <Field label="Grid H" value={form.gridHeight} onChange={v => updateField('gridHeight', parseInt(v) || 0)} width={60} type="number"/>
                            <Field label="Tick Limit" value={form.tickLimit} onChange={v => updateField('tickLimit', parseInt(v) || 0)} width={70} type="number"/>
                            <Field label="Food" value={form.foodCount} onChange={v => updateField('foodCount', parseInt(v) || 0)} width={60} type="number"/>
                            <Field label="Food Respawn" value={form.foodRespawnRate} onChange={v => updateField('foodRespawnRate', parseInt(v) || 0)} width={80} type="number"/>
                            <Field label="Seed" value={form.seed} onChange={v => updateField('seed', parseInt(v) || 0)} width={70} type="number"/>
                            <Field label="Showcase Interval" value={form.showcaseInterval} onChange={v => updateField('showcaseInterval', parseInt(v) || 0)} width={100} type="number"/>
                        </div>
                        <div>
                            <button type="submit" style={btnStyle('#1a5a2a', '#2a8a4a')}>Submit Job</button>
                        </div>
                    </form>
                </Section>

                {/* Pending Jobs */}
                <Section title={`Pending Jobs (${pending.length})`}>
                    {pending.length === 0 ? (
                        <span style={{color: '#666'}}>No pending jobs</span>
                    ) : (
                        <table style={{width: '100%', borderCollapse: 'collapse'}}>
                            <thead>
                                <tr style={{borderBottom: '1px solid #333', color: '#888', textAlign: 'left'}}>
                                    <th style={thStyle}>Name</th>
                                    <th style={thStyle}>Priority</th>
                                    <th style={thStyle}>Generations</th>
                                    <th style={thStyle}>Grid</th>
                                    <th style={thStyle}>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {pending.map(job => (
                                    <tr key={job.filename} style={{borderBottom: '1px solid #222'}}>
                                        <td style={tdStyle}>{job.config.name}</td>
                                        <td style={tdStyle}>{job.config.priority}</td>
                                        <td style={tdStyle}>{(job.config.evolution as Record<string, unknown>).generations as number ?? '?'}</td>
                                        <td style={tdStyle}>{`${(job.config.evolution as Record<string, unknown>).gridWidth ?? '?'}x${(job.config.evolution as Record<string, unknown>).gridHeight ?? '?'}`}</td>
                                        <td style={tdStyle}>
                                            <button onClick={() => handleDeletePending(job.filename)} style={btnStyle('#3a1a1a', '#5a2a2a')}>Delete</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </Section>

                {/* Run History */}
                <Section title={`Run History (${runs.length})`}>
                    {runs.length === 0 ? (
                        <span style={{color: '#666'}}>No runs yet</span>
                    ) : (
                        <div style={{display: 'flex', flexDirection: 'column', gap: 4}}>
                            {runs.map(run => (
                                <div key={run.runId}>
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 12,
                                        padding: '6px 10px',
                                        background: '#1a1a1a',
                                        borderRadius: 4,
                                        border: '1px solid #282828',
                                        flexWrap: 'wrap',
                                    }}>
                                        <StatusBadge state={run.status.state}/>
                                        <span style={{color: '#ddd', minWidth: 120}}>{run.status.jobName}</span>
                                        <Stat label="Gen" value={`${run.status.generation}/${run.status.totalGenerations}`}/>
                                        <Stat label="Best" value={run.status.bestFitness.toFixed(1)}/>
                                        <Stat label="Archive" value={`${(run.status.archiveFillRate * 100).toFixed(0)}%`}/>
                                        <button onClick={() => handleViewLog(run.runId)} style={btnStyle('#1a2a3a', '#2a4a6a')}>
                                            {expandedLog === run.runId ? 'Hide Log' : 'Log'}
                                        </button>
                                        {run.status.hasReplays && (
                                            <button onClick={() => handleViewReplays(run.runId)} style={btnStyle('#1a3a2a', '#2a6a4a')}>
                                                {expandedReplays === run.runId ? 'Hide Replays' : 'Replays'}
                                            </button>
                                        )}
                                    </div>
                                    {expandedLog === run.runId && (
                                        <pre style={{
                                            margin: '4px 0 0 0',
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
                                            {logText || '(empty)'}
                                        </pre>
                                    )}
                                    {expandedReplays === run.runId && (
                                        <div style={{
                                            margin: '4px 0 0 0',
                                            padding: 10,
                                            background: '#0a0a0a',
                                            border: '1px solid #282828',
                                            borderRadius: 4,
                                        }}>
                                            <div style={{display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8}}>
                                                {replayGens.length === 0 ? (
                                                    <span style={{color: '#666', fontSize: 11}}>No replays saved yet</span>
                                                ) : replayGens.map(gen => (
                                                    <button
                                                        key={gen}
                                                        onClick={() => handleSelectGen(run.runId, gen)}
                                                        style={{
                                                            ...btnStyle(selectedGen === gen ? '#2a4a6a' : '#1a2a3a', selectedGen === gen ? '#4a7aaa' : '#2a4a6a'),
                                                            fontSize: 11,
                                                            padding: '3px 10px',
                                                        }}
                                                    >
                                                        Gen {gen}
                                                    </button>
                                                ))}
                                            </div>
                                            {selectedGen !== null && replayIndex && (
                                                <div>
                                                    <div style={{color: '#888', fontSize: 11, marginBottom: 6}}>
                                                        {replayIndex.gridWidth}x{replayIndex.gridHeight} grid, {replayIndex.tickLimit} tick limit
                                                    </div>
                                                    <table style={{width: '100%', borderCollapse: 'collapse'}}>
                                                        <thead>
                                                            <tr style={{borderBottom: '1px solid #333', color: '#888', textAlign: 'left'}}>
                                                                <th style={thStyle}>Match</th>
                                                                <th style={thStyle}>Genomes</th>
                                                                <th style={thStyle}>Ticks</th>
                                                                <th style={thStyle}>Action</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {replayIndex.matches.map(match => (
                                                                <tr key={match.matchIndex} style={{borderBottom: '1px solid #222'}}>
                                                                    <td style={tdStyle}>#{match.matchIndex}</td>
                                                                    <td style={tdStyle}>{match.genomeIds.join(', ')}</td>
                                                                    <td style={tdStyle}>{match.totalTicks}</td>
                                                                    <td style={tdStyle}>
                                                                        <a
                                                                            href={`/match?run=${encodeURIComponent(run.runId)}&gen=${selectedGen}&match=${match.matchIndex}${hostQuery}`}
                                                                            style={{color: '#4c8', textDecoration: 'none', fontSize: 12}}
                                                                        >
                                                                            Play
                                                                        </a>
                                                                    </td>
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </Section>
            </div>
        </div>
    );
}

function Section({title, children}: {title: string; children: React.ReactNode}) {
    return (
        <div style={{
            padding: '12px 14px',
            background: '#1a1a1a',
            borderRadius: 4,
            border: '1px solid #333',
        }}>
            <div style={{color: '#888', fontSize: 11, textTransform: 'uppercase', marginBottom: 10, letterSpacing: 1}}>{title}</div>
            {children}
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

function Field({label, value, onChange, width, type = 'text'}: {
    label: string;
    value: string | number;
    onChange: (v: string) => void;
    width: number;
    type?: string;
}) {
    return (
        <label style={{display: 'flex', flexDirection: 'column', gap: 2}}>
            <span style={{color: '#888', fontSize: 11}}>{label}</span>
            <input
                type={type}
                value={value}
                onChange={e => onChange(e.target.value)}
                style={{
                    width,
                    padding: '4px 6px',
                    background: '#222',
                    color: '#ccc',
                    border: '1px solid #444',
                    borderRadius: 3,
                    fontFamily: 'monospace',
                    fontSize: 12,
                }}
            />
        </label>
    );
}

function StatusBadge({state}: {state: string}) {
    const colors: Record<string, string> = {
        running: '#4c4',
        completed: '#48f',
        paused: '#cc4',
        failed: '#c44',
    };
    return (
        <span style={{
            color: colors[state] ?? '#888',
            fontSize: 11,
            padding: '2px 8px',
            border: `1px solid ${colors[state] ?? '#555'}`,
            borderRadius: 3,
            minWidth: 60,
            textAlign: 'center',
            display: 'inline-block',
        }}>
            {state}
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

const thStyle: React.CSSProperties = {padding: '4px 8px', fontSize: 11};
const tdStyle: React.CSSProperties = {padding: '4px 8px'};
