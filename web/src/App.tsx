import {useEffect, useState} from 'react';
import HexGridCanvas from './components/HexGridCanvas';
import type {GridState} from './types';

export default function App() {
    const [grid, setGrid] = useState<GridState | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch('/api/grid')
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then((data: GridState) => setGrid(data))
            .catch((err) => setError(err.message));
    }, []);

    if (error) {
        return (
            <div style={{padding: 20, color: '#ff4444', fontFamily: 'monospace'}}>
                Failed to load grid: {error}
                <br/>
                Make sure the Kotlin server is running on port 8080.
            </div>
        );
    }

    if (!grid) {
        return (
            <div style={{padding: 20, color: '#888', fontFamily: 'monospace'}}>
                Loading grid...
            </div>
        );
    }

    return (
        <div style={{width: '100vw', height: '100vh', background: '#111'}}>
            <HexGridCanvas grid={grid}/>
        </div>
    );
}
