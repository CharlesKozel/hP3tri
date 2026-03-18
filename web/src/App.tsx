import {BrowserRouter, Routes, Route} from 'react-router-dom';
import NavBar from './components/NavBar';
import EvolutionDashboard from './pages/EvolutionDashboard';
import MatchViewer from './pages/MatchViewer';
import QLearningDashboard from './pages/QLearningDashboard';
import QueueDashboard from './pages/QueueDashboard';
import TournamentDashboard from './pages/TournamentDashboard';

export default function App() {
    return (
        <BrowserRouter>
            <div style={{width: '100vw', height: '100vh', background: '#111', display: 'flex', flexDirection: 'column'}}>
                <NavBar/>
                <Routes>
                    <Route path="/" element={<TournamentDashboard/>}/>
                    <Route path="/evolution" element={<EvolutionDashboard/>}/>
                    <Route path="/qlearning" element={<QLearningDashboard/>}/>
                    <Route path="/match" element={<MatchViewer/>}/>
                    <Route path="/queue" element={<QueueDashboard/>}/>
                </Routes>
            </div>
        </BrowserRouter>
    );
}
