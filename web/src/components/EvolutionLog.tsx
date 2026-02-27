import {useEffect, useRef} from 'react';

interface EvolutionLogProps {
    entries: string[];
}

export default function EvolutionLog({entries}: EvolutionLogProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [entries]);

    return (
        <div>
            <div style={{color: '#888', fontSize: 11, marginBottom: 4}}>Log</div>
            <div
                ref={containerRef}
                style={{
                    height: 160,
                    overflowY: 'auto',
                    background: '#0d0d0d',
                    border: '1px solid #333',
                    borderRadius: 4,
                    padding: '6px 8px',
                    fontSize: 11,
                    color: '#888',
                    lineHeight: 1.5,
                }}
            >
                {entries.length === 0 ? (
                    <span style={{color: '#555'}}>No log entries</span>
                ) : (
                    entries.map((entry, i) => <div key={i}>{entry}</div>)
                )}
            </div>
        </div>
    );
}
