import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useWebSocket } from '../contexts/WebSocketContext';
import './Header.css';

const Header: React.FC = () => {
  const location = useLocation();
  const { isConnected } = useWebSocket();

  const navItems = [
    { path: '/', label: 'Chat', icon: '💬' },
    { path: '/editor', label: 'Editor', icon: '📝' },
    { path: '/canvas', label: 'Canvas', icon: '🎨' },
    { path: '/integrated', label: 'Dev Canvas', icon: '🔧' },
  ];

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <Link to="/" className="logo">
            <span className="logo-icon">🕉️</span>
            <span className="logo-text">Sanskrit Engine</span>
          </Link>
        </div>

        <nav className="nav">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </Link>
          ))}
        </nav>

        <div className="header-right">
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-indicator"></span>
            <span className="status-text">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;