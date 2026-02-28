import {BrowserRouter, Routes, Route} from 'react-router-dom';
import NavBar from './components/NavBar';
import EvolutionDashboard from './pages/EvolutionDashboard';
import MatchViewer from './pages/MatchViewer';
import QueueDashboard from './pages/QueueDashboard';

export default function App() {
    return (
        <BrowserRouter>
            <div style={{width: '100vw', height: '100vh', background: '#111', display: 'flex', flexDirection: 'column'}}>
                <NavBar/>
                <Routes>
                    <Route path="/" element={<EvolutionDashboard/>}/>
                    <Route path="/match" element={<MatchViewer/>}/>
                    <Route path="/queue" element={<QueueDashboard/>}/>
                </Routes>
            </div>
        </BrowserRouter>
    );
}
