import {useNavigate} from 'react-router-dom';
import type {ArchiveEntry} from '../types';

interface GenomeInfoProps {
    entry: ArchiveEntry | null;
}

const SYMMETRY_NAMES: Record<number, string> = {
    0: 'Asymmetric',
    1: 'Bilateral',
    2: 'Radial-2',
    3: 'Radial-3',
    4: 'Radial-4',
    5: 'Radial-5',
    6: 'Radial-6',
};

export default function GenomeInfo({entry}: GenomeInfoProps) {
    const navigate = useNavigate();

    if (!entry) {
        return (
            <div style={{color: '#555', fontSize: 12, padding: '12px 0'}}>
                Click a cell in the MAP-Elites grid to view genome details
            </div>
        );
    }

    const handleRunMatch = () => {
        navigate(`/match?genomes=${entry.genomeId}`);
    };

    return (
        <div>
            <div style={{color: '#888', fontSize: 11, marginBottom: 4}}>Selected Genome</div>
            <div style={{
                background: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: 4,
                padding: '10px 12px',
                fontSize: 12,
            }}>
                <Row label="Genome ID" value={entry.genomeId}/>
                <Row label="Fitness" value={entry.fitness.toFixed(1)}/>
                <Row label="Bin" value={`(${entry.binX}, ${entry.binY})`}/>
                <Row label="Mobility" value={entry.mobility.toFixed(3)}/>
                <Row label="Aggression" value={entry.aggression.toFixed(3)}/>
                <Row label="Symmetry" value={SYMMETRY_NAMES[entry.symmetryMode] ?? `Mode ${entry.symmetryMode}`}/>
                <button
                    onClick={handleRunMatch}
                    style={{
                        marginTop: 8,
                        padding: '6px 16px',
                        background: '#1a5a2a',
                        color: '#ccc',
                        border: '1px solid #2a8a4a',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontFamily: 'monospace',
                        fontSize: 12,
                        width: '100%',
                    }}
                >
                    Run Sample Match
                </button>
            </div>
        </div>
    );
}

function Row({label, value}: {label: string; value: string | number}) {
    return (
        <div style={{display: 'flex', justifyContent: 'space-between', padding: '2px 0', color: '#aaa'}}>
            <span>{label}</span>
            <span style={{color: '#ccc'}}>{String(value)}</span>
        </div>
    );
}
