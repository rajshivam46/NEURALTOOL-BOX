import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Brain, Zap, Network, GitBranch, Activity, ScanFace } from 'lucide-react';
import './Sidebar.css';

const Sidebar = () => {
  const menus = [
    { path: '/', label: 'Overview', icon: LayoutDashboard },
    { path: '/perceptron', label: 'Perceptron', icon: Zap },
    { path: '/backprop', label: 'Backpropagation', icon: Network },
    { path: '/rnn', label: 'RNN Forecast', icon: GitBranch },
    { path: '/cnn', label: 'CNN Face Auth', icon: ScanFace },
    { path: '/loss', label: 'Loss Analysis', icon: Activity },
  ];

  return (
    <div className="sidebar-container">
      <div className="sidebar-header">
        <div className="logo-icon"><Brain size={28} /></div>
        <h1>Neural Toolbox</h1>
      </div>
      
      <div className="sidebar-nav">
        <div className="nav-group-title">MODELS</div>
        {menus.map(m => {
          const Icon = m.icon;
          return (
            <NavLink 
              key={m.path} 
              to={m.path} 
              className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} />
              <span>{m.label}</span>
            </NavLink>
          );
        })}
      </div>
    </div>
  );
};

export default Sidebar;
