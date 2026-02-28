import {useCallback, useEffect, useRef, useState} from 'react';
import MapElitesGrid from '../components/MapElitesGrid';
import FitnessChart from '../components/FitnessChart';
import EvolutionLog from '../components/EvolutionLog';
import GenomeInfo from '../components/GenomeInfo';
import {getApiBase} from '../api';
import type {ArchiveEntry, EvolutionStatus, HistoryEntry} from '../types';

const POLL_INTERVAL = 2000;

export default function EvolutionDashboard() {
    const [status, setStatus] = useState<EvolutionStatus | null>(null);
    const [archive, setArchive] = useState<ArchiveEntry[]>([]);
    const [history, setHistory] = useState<HistoryEntry[]>([]);
    const [selectedBin, setSelectedBin] = useState<{x: number; y: number} | null>(null);
    const [error, setError] = useState<string | null>(null);
    const lastGenRef = useRef(-1);
    const pollRef = useRef<number | null>(null);

    const fetchStatus = useCallback(async () => {
        const base = getApiBase();
        try {
            const res = await fetch(`${base}/api/evolution/status`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data: EvolutionStatus = await res.json();
            setStatus(data);
            setError(null);

            if (data.generation !== lastGenRef.current) {
                lastGenRef.current = data.generation;
                const [archiveRes, historyRes] = await Promise.all([
                    fetch(`${base}/api/evolution/archive`),
                    fetch(`${base}/api/evolution/history`),
                ]);
                if (archiveRes.ok) setArchive(await archiveRes.json());
                if (historyRes.ok) setHistory(await historyRes.json());
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        pollRef.current = window.setInterval(fetchStatus, POLL_INTERVAL);
        return () => {
            if (pollRef.current !== null) clearInterval(pollRef.current);
        };
    }, [fetchStatus]);

    const handleStart = async () => {
        try {
            const res = await fetch(`${getApiBase()}/api/evolution/start`, {method: 'POST'});
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            await fetchStatus();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const handleStop = async () => {
        try {
            const res = await fetch(`${getApiBase()}/api/evolution/stop`, {method: 'POST'});
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            await fetchStatus();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    const selectedEntry = selectedBin
        ? archive.find(e => e.binX === selectedBin.x && e.binY === selectedBin.y) ?? null
        : null;

    const running = status?.running ?? false;

    return (
        <div style={{
            flex: 1,
            display: 'flex',
            fontFamily: 'monospace',
            fontSize: 13,
            color: '#ccc',
            overflow: 'hidden',
        }}>
            <div style={{flex: 1, display: 'flex', flexDirection: 'column', padding: 16, gap: 16, overflowY: 'auto'}}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '10px 14px',
                    background: '#1a1a1a',
                    borderRadius: 4,
                    border: '1px solid #333',
                    flexWrap: 'wrap',
                }}>
                    <button onClick={handleStart} disabled={running} style={btnStyle(running ? '#222' : '#1a5a2a', running ? '#555' : '#2a8a4a')}>
                        Start
                    </button>
                    <button onClick={handleStop} disabled={!running} style={btnStyle(!running ? '#222' : '#5a1a1a', !running ? '#555' : '#8a2a2a')}>
                        Stop
                    </button>
                    <span style={{color: running ? '#4c4' : '#888'}}>
                        {running ? 'Running' : 'Stopped'}
                    </span>
                    {status && (
                        <>
                            <Stat label="Generation" value={`${status.generation}/${status.totalGenerations}`}/>
                            <Stat label="Best Fitness" value={status.bestFitness.toFixed(1)}/>
                            <Stat label="Archive" value={`${(status.archiveFillRate * 100).toFixed(0)}%`}/>
                            <Stat label="Matches" value={status.matchesCompleted}/>
                        </>
                    )}
                    {error && <span style={{color: '#c44', fontSize: 11}}>Error: {error}</span>}
                </div>

                <div style={{display: 'flex', gap: 16, flexWrap: 'wrap'}}>
                    <MapElitesGrid
                        entries={archive}
                        binsX={8}
                        binsY={8}
                        selectedBin={selectedBin}
                        onSelectBin={(x, y) => setSelectedBin({x, y})}
                    />
                    <div style={{flex: 1, minWidth: 300}}>
                        <FitnessChart history={history}/>
                    </div>
                </div>

                <EvolutionLog entries={status?.log ?? []}/>
            </div>

            <div style={{
                width: 280,
                minWidth: 280,
                borderLeft: '1px solid #333',
                padding: 16,
                background: '#151515',
                overflowY: 'auto',
            }}>
                <GenomeInfo entry={selectedEntry}/>
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

function btnStyle(bg: string, border: string): React.CSSProperties {
    return {
        padding: '6px 20px',
        background: bg,
        color: '#ccc',
        border: `1px solid ${border}`,
        borderRadius: 4,
        cursor: 'pointer',
        fontFamily: 'monospace',
        fontSize: 13,
    };
}
