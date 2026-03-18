import {NavLink, useSearchParams} from 'react-router-dom';

const linkStyle: React.CSSProperties = {
    padding: '6px 16px',
    color: '#888',
    textDecoration: 'none',
    borderRadius: 4,
    fontSize: 13,
};

const activeLinkStyle: React.CSSProperties = {
    ...linkStyle,
    color: '#eee',
    background: '#333',
};

export default function NavBar() {
    const [searchParams] = useSearchParams();
    const host = searchParams.get('host');
    const hostQuery = host ? `?host=${encodeURIComponent(host)}` : '';

    return (
        <nav style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: '6px 12px',
            background: '#0d0d0d',
            borderBottom: '1px solid #333',
            fontFamily: 'monospace',
            flexShrink: 0,
        }}>
            <span style={{color: '#555', marginRight: 8, fontSize: 14, fontWeight: 'bold'}}>hP3tri</span>
            <NavLink to={`/${hostQuery}`} end style={({isActive}) => isActive ? activeLinkStyle : linkStyle}>
                Tournament
            </NavLink>
            <NavLink to={`/match${hostQuery}`} style={({isActive}) => isActive ? activeLinkStyle : linkStyle}>
                Match Viewer
            </NavLink>
            <NavLink to={`/qlearning${hostQuery}`} style={({isActive}) => isActive ? activeLinkStyle : linkStyle}>
                Q-Learning
            </NavLink>
            <NavLink to={`/queue${hostQuery}`} style={({isActive}) => isActive ? activeLinkStyle : linkStyle}>
                Queue
            </NavLink>
            <NavLink to={`/evolution${hostQuery}`} style={({isActive}) => isActive ? activeLinkStyle : linkStyle}>
                Evolution
            </NavLink>
            {host && <span style={{color: '#666', fontSize: 11, marginLeft: 'auto'}}>remote: {host}</span>}
        </nav>
    );
}
