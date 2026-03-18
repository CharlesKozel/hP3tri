import {useCallback, useEffect, useRef, useState} from 'react';
import {useSearchParams} from 'react-router-dom';
import {getApiBase} from '../api';
import type {ReplayIndex} from '../types';

const POLL_INTERVAL = 3000;

interface QLCheckpoint {
    runId: string;
    filename: string;
    path: string;
}

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
    seedCheckpoint: string | null;
    jobType: string;
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

type JobType = 'tournament' | 'qlearning';

interface JobForm {
    jobType: JobType;
    name: string;
    description: string;
    priority: number;
    // Tournament fields
    populationSize: number;
    matchPopulationSize: number;
    generations: number;
    matchesPerGeneration: number;
    gridWidth: number;
    gridHeight: number;
    matchTickLimit: number;
    previewTickLimit: number;
    previewGridSize: number;
    seed: number;
    // Q-Learning fields
    qlTotalMatches: number;
    qlGenomesPerMatch: number;
    qlTrainingSteps: number;
    qlBatchSize: number;
    qlFoodCount: number;
    seedCheckpoint: string;
}

const DEFAULT_FORM: JobForm = {
    jobType: 'tournament',
    name: '',
    description: '',
    priority: 0,
    populationSize: 100,
    matchPopulationSize: 1,
    generations: 10,
    matchesPerGeneration: 500,
    gridWidth: 64,
    gridHeight: 64,
    matchTickLimit: 500,
    previewTickLimit: 100,
    previewGridSize: 128,
    seed: 42,
    qlTotalMatches: 5000,
    qlGenomesPerMatch: 4,
    qlTrainingSteps: 32,
    qlBatchSize: 64,
    qlFoodCount: 80,
    seedCheckpoint: '',
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
    const [checkpoints, setCheckpoints] = useState<QLCheckpoint[]>([]);
    const pollRef = useRef<number | null>(null);

    const base = getApiBase();
    const host = searchParams.get('host');
    const hostQuery = host ? `&host=${encodeURIComponent(host)}` : '';

    const fetchAll = useCallback(async () => {
        try {
            const [pendingRes, runsRes, currentRes, cpRes] = await Promise.all([
                fetch(`${base}/api/queue/pending`),
                fetch(`${base}/api/queue/runs`),
                fetch(`${base}/api/queue/current`),
                fetch(`${base}/api/qlearning/checkpoints`),
            ]);
            if (pendingRes.ok) setPending(await pendingRes.json());
            if (runsRes.ok) setRuns(await runsRes.json());
            if (currentRes.ok && currentRes.status !== 204) {
                setCurrent(await currentRes.json());
            } else {
                setCurrent(null);
            }
            if (cpRes.ok) setCheckpoints(await cpRes.json());
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
            const body: Record<string, unknown> = {
                name: form.name.trim(),
                description: form.description,
                priority: form.priority,
            };
            if (form.seedCheckpoint) {
                body.seedCheckpoint = form.seedCheckpoint;
            }
            if (form.jobType === 'qlearning') {
                body.qlearning = {
                    totalMatches: form.qlTotalMatches,
                    gridWidth: form.gridWidth,
                    gridHeight: form.gridHeight,
                    matchTickLimit: form.matchTickLimit,
                    foodCount: form.qlFoodCount,
                    genomesPerMatch: form.qlGenomesPerMatch,
                    trainingStepsPerMatch: form.qlTrainingSteps,
                    batchSize: form.qlBatchSize,
                    seed: form.seed,
                };
            } else {
                body.tournament = {
                    populationSize: form.populationSize,
                    matchPopulationSize: form.matchPopulationSize,
                    generations: form.generations,
                    matchesPerGeneration: form.matchesPerGeneration,
                    gridWidth: form.gridWidth,
                    gridHeight: form.gridHeight,
                    matchTickLimit: form.matchTickLimit,
                    previewTickLimit: form.previewTickLimit,
                    previewGridSize: form.previewGridSize,
                    seed: form.seed,
                };
            }
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
                        <div style={{display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end'}}>
                            <label style={{display: 'flex', flexDirection: 'column', gap: 2}}>
                                <span style={{color: '#888', fontSize: 11}}>Type</span>
                                <select
                                    value={form.jobType}
                                    onChange={e => updateField('jobType', e.target.value)}
                                    style={{
                                        padding: '4px 6px',
                                        background: '#222',
                                        color: '#ccc',
                                        border: '1px solid #444',
                                        borderRadius: 3,
                                        fontFamily: 'monospace',
                                        fontSize: 12,
                                    }}
                                >
                                    <option value="tournament">Tournament</option>
                                    <option value="qlearning">Q-Learning</option>
                                </select>
                            </label>
                            <Field label="Name" value={form.name} onChange={v => updateField('name', v)} width={200}/>
                            <Field label="Priority" value={form.priority} onChange={v => updateField('priority', parseInt(v) || 0)} width={60} type="number"/>
                            <Field label="Description" value={form.description} onChange={v => updateField('description', v)} width={300}/>
                        </div>
                        {form.jobType === 'tournament' ? (
                            <>
                                <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                                    <Field label="Pop Size" value={form.populationSize} onChange={v => updateField('populationSize', parseInt(v) || 0)} width={70} type="number"/>
                                    <Field label="Seeds/Genome" value={form.matchPopulationSize} onChange={v => updateField('matchPopulationSize', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Generations" value={form.generations} onChange={v => updateField('generations', parseInt(v) || 0)} width={70} type="number"/>
                                    <Field label="Matches/Gen" value={form.matchesPerGeneration} onChange={v => updateField('matchesPerGeneration', parseInt(v) || 0)} width={80} type="number"/>
                                </div>
                                <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                                    <Field label="Grid W" value={form.gridWidth} onChange={v => updateField('gridWidth', parseInt(v) || 0)} width={60} type="number"/>
                                    <Field label="Grid H" value={form.gridHeight} onChange={v => updateField('gridHeight', parseInt(v) || 0)} width={60} type="number"/>
                                    <Field label="Match Ticks" value={form.matchTickLimit} onChange={v => updateField('matchTickLimit', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Preview Ticks" value={form.previewTickLimit} onChange={v => updateField('previewTickLimit', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Preview Grid" value={form.previewGridSize} onChange={v => updateField('previewGridSize', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Seed" value={form.seed} onChange={v => updateField('seed', parseInt(v) || 0)} width={70} type="number"/>
                                </div>
                            </>
                        ) : (
                            <>
                                <div style={{display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end'}}>
                                    <Field label="Total Matches" value={form.qlTotalMatches} onChange={v => updateField('qlTotalMatches', parseInt(v) || 0)} width={90} type="number"/>
                                    <Field label="Genomes/Match" value={form.qlGenomesPerMatch} onChange={v => updateField('qlGenomesPerMatch', parseInt(v) || 0)} width={90} type="number"/>
                                    <Field label="Food Count" value={form.qlFoodCount} onChange={v => updateField('qlFoodCount', parseInt(v) || 0)} width={80} type="number"/>
                                    <label style={{display: 'flex', flexDirection: 'column', gap: 2}}>
                                        <span style={{color: '#888', fontSize: 11}}>Resume From</span>
                                        <select
                                            value={form.seedCheckpoint}
                                            onChange={e => updateField('seedCheckpoint', e.target.value)}
                                            style={{
                                                padding: '4px 6px',
                                                background: '#222',
                                                color: '#ccc',
                                                border: '1px solid #444',
                                                borderRadius: 3,
                                                fontFamily: 'monospace',
                                                fontSize: 12,
                                                maxWidth: 280,
                                            }}
                                        >
                                            <option value="">Fresh start</option>
                                            {checkpoints.map(cp => (
                                                <option key={cp.path} value={cp.path}>
                                                    {cp.runId.slice(0, 20)}... / {cp.filename}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                </div>
                                <div style={{display: 'flex', gap: 10, flexWrap: 'wrap'}}>
                                    <Field label="Grid W" value={form.gridWidth} onChange={v => updateField('gridWidth', parseInt(v) || 0)} width={60} type="number"/>
                                    <Field label="Grid H" value={form.gridHeight} onChange={v => updateField('gridHeight', parseInt(v) || 0)} width={60} type="number"/>
                                    <Field label="Match Ticks" value={form.matchTickLimit} onChange={v => updateField('matchTickLimit', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Train Steps" value={form.qlTrainingSteps} onChange={v => updateField('qlTrainingSteps', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Batch Size" value={form.qlBatchSize} onChange={v => updateField('qlBatchSize', parseInt(v) || 0)} width={80} type="number"/>
                                    <Field label="Seed" value={form.seed} onChange={v => updateField('seed', parseInt(v) || 0)} width={70} type="number"/>
                                </div>
                            </>
                        )}
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
                                        <td style={tdStyle}>{((job.config as Record<string, unknown>).tournament as Record<string, unknown>)?.generations as number ?? '?'}</td>
                                        <td style={tdStyle}>{`${((job.config as Record<string, unknown>).tournament as Record<string, unknown>)?.gridWidth ?? '?'}x${((job.config as Record<string, unknown>).tournament as Record<string, unknown>)?.gridHeight ?? '?'}`}</td>
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
                                        <span style={{
                                            color: run.status.jobType === 'qlearning' ? '#c8c' : '#ddd',
                                            minWidth: 120,
                                        }}>
                                            {run.status.jobType === 'qlearning' ? '[QL] ' : ''}{run.status.jobName}
                                        </span>
                                        <Stat label="Gen" value={`${run.status.generation}/${run.status.totalGenerations}`}/>
                                        <Stat label="Best" value={run.status.bestFitness.toFixed(1)}/>
                                        <Stat label="Archive" value={`${(run.status.archiveFillRate * 100).toFixed(0)}%`}/>
                                        {run.status.seedCheckpoint && (
                                            <span style={{color: '#886', fontSize: 11}} title={run.status.seedCheckpoint}>
                                                resumed
                                            </span>
                                        )}
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
