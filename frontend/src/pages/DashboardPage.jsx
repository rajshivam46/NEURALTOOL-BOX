import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, CartesianGrid, ComposedChart } from 'recharts';
import { Database, Brain, Activity, Target, Download, Printer, Maximize2, RefreshCw, ChevronDown, MessageSquare, ShoppingCart, Users, Package, Zap } from 'lucide-react';
import './Dashboard.css';

// Reusable Components
const SparkLine = ({ data, color, type = 'bar' }) => (
  <div className="ds-sparkline-container">
    <ResponsiveContainer width="100%" height={28}>
      {type === 'bar' ? (
        <BarChart data={data?.length ? data : [{value: 0}]}>
          <Bar dataKey="value" fill={color} radius={[1, 1, 0, 0]}>
            {(data || []).map((entry, index) => (
              <Cell key={`cell-${index}`} fill={color} fillOpacity={0.4 + (entry.value / 100) * 0.6} />
            ))}
          </Bar>
        </BarChart>
      ) : (
        <BarChart data={data?.length ? data : [{value: 0}]}>
          <Bar dataKey="value" fill={color} radius={[1, 1, 0, 0]} />
        </BarChart>
      )}
    </ResponsiveContainer>
    <div className="ds-spark-progress">
      <div className="ds-spark-fill" style={{ width: '100%', background: color }}></div>
      <div className="ds-spark-thumb" style={{ left: '100%' }}></div>
    </div>
  </div>
);

const GradientCard = ({ title, value, icon: Icon, gradientClass, sparkData = [] }) => {
  const data = sparkData.length > 0 ? sparkData : Array(35).fill(0).map(() => ({ value: 0 }));
  return (
    <div className={`ds-card ds-gradient-card ${gradientClass}`}>
      <div className="ds-card-bg-icon"><Icon size={120} strokeWidth={1} /></div>
      <div className="ds-card-content">
        <div className="ds-card-top">
          <div className="ds-card-value">{value}</div>
          <Icon size={20} className="ds-icon-right" />
        </div>
        <div className="ds-card-title">{title}</div>
        <div className="ds-card-bottom">
          <span className="ds-card-subtitle">Monthly progress</span>
          <SparkLine data={data} color="#ffffff" />
        </div>
      </div>
    </div>
  );
};

const StatusCard = ({ title, value, status, icon: Icon, valuePrefix = '', sparkData = [] }) => {
  const data = sparkData.length > 0 ? sparkData : Array(35).fill(0).map(() => ({ value: 0 }));
  return (
    <div className="ds-card ds-status-card">
      <div className="ds-card-content">
        <div className="ds-card-top">
          <div className="ds-status-value-group">
            <span className="ds-card-value">
              {valuePrefix && <span className="ds-prefix">{valuePrefix}</span>}
              {value}
            </span>
            {status === 'up' && <span className="ds-indicator ds-up"><ChevronDown size={14} style={{transform: 'rotate(180deg)'}}/></span>}
            {status === 'down' && <span className="ds-indicator ds-down"><ChevronDown size={14}/></span>}
          </div>
          <Icon size={20} className="ds-icon-muted" />
        </div>
        <div className="ds-card-title ds-title-muted">{title}</div>
        <div className="ds-card-bottom">
          <span className="ds-card-subtitle ds-subtitle-muted">Monthly progress</span>
          <SparkLine data={data} color="#4c5680" type="simple" />
        </div>
      </div>
    </div>
  );
};

const DashboardPage = () => {
  const [stats, setStats] = React.useState({
    total_datasets: 0,
    models_trained: 0,
    total_epochs: 0,
    avg_accuracy: 0,
    performance_history: []
  });

  React.useEffect(() => {
    fetch('/api/data/dashboard-stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error("Failed to fetch dashboard stats", err));
  }, []);

  const performanceData = stats.performance_history && stats.performance_history.length > 0 
    ? stats.performance_history 
    : [
        { name: 'No Data', val1: 0, val2: 0, area: 0 }
      ];

  return (
    <div className="ds-wrapper">
      <div className="ds-header">
        <h1 className="ds-main-title">Classic Dashboard</h1>
        <p className="ds-main-subtitle">Creatively crafted Dashboard for your needs</p>
      </div>

      <div className="ds-grid">
        <GradientCard title="TOTAL DATASETS" value={stats.total_datasets} icon={Package} gradientClass="ds-grad-cyan" />
        <GradientCard title="MODELS TRAINED" value={stats.models_trained} icon={Users} gradientClass="ds-grad-pink" />
        <GradientCard title="TOTAL EPOCHS" value={stats.total_epochs} icon={ShoppingCart} gradientClass="ds-grad-orange" />
        <GradientCard title="AVG. ACCURACY" value={`${stats.avg_accuracy}%`} icon={MessageSquare} gradientClass="ds-grad-purple" />
      </div>

      <div className="ds-grid">
        <StatusCard title="TOTAL DATASETS" value={stats.total_datasets} status="none" icon={Package} />
        <StatusCard title="MODELS TRAINED" value={stats.models_trained} status="none" icon={Users} />
        <StatusCard title="TOTAL EPOCHS" value={stats.total_epochs} status="none" icon={ShoppingCart} />
        <StatusCard title="AVG. ACCURACY" value={`${stats.avg_accuracy}%`} status="none" icon={MessageSquare} />
      </div>

      <div className="ds-grid-panels">
        {/* Performance Panel */}
        <div className="ds-panel">
          <div className="ds-panel-header">
            <h3 className="ds-panel-title">Production <span className="ds-panel-date">February 2017</span></h3>
            <div className="ds-panel-actions">
              <button className="ds-btn"><Download size={14}/> Export</button>
              <button className="ds-btn ds-btn-green"><Printer size={14}/> Print</button>
              <RefreshCw size={14} className="ds-icon-btn" />
              <Maximize2 size={14} className="ds-icon-btn" />
              <ChevronDown size={14} className="ds-icon-btn" />
            </div>
          </div>
          <div className="ds-panel-body" style={{ height: '320px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={performanceData} margin={{ top: 20, right: 0, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#fc5c7d" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#fc5c7d" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2e334a" />
                <XAxis dataKey="name" tick={{fill: '#a1a3b1', fontSize: 11}} axisLine={false} tickLine={false} />
                <YAxis tick={{fill: '#a1a3b1', fontSize: 11}} axisLine={false} tickLine={false} domain={[-80, 80]} ticks={[-80, -60, -40, -20, 0, 20, 40, 60, 80]} />
                <Tooltip contentStyle={{background: '#272b3b', border: '1px solid #455486', color: '#fff'}} />
                <Bar dataKey="val1" fill="#718cff" barSize={24} />
                <Bar dataKey="val2" fill="#24cdd5" barSize={24} />
                <Area type="monotone" dataKey="area" stroke="#fc5c7d" strokeWidth={2} fillOpacity={1} fill="url(#colorArea)" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Global Activity Map */}
        <div className="ds-panel">
          <div className="ds-panel-header">
            <h3 className="ds-panel-title">Production <span className="ds-panel-date">February 2017</span></h3>
            <div className="ds-panel-actions">
              <button className="ds-btn"><Download size={14}/> Export</button>
              <button className="ds-btn ds-btn-green"><Printer size={14}/> Print</button>
              <Maximize2 size={14} className="ds-icon-btn" />
              <ChevronDown size={14} className="ds-icon-btn" />
            </div>
          </div>
          <div className="ds-panel-body ds-map-container" style={{ height: '320px', position: 'relative' }}>
            <div className="ds-map-controls">
              <button className="ds-map-zoom">+</button>
              <button className="ds-map-zoom">-</button>
            </div>
            <img src="/map.svg" alt="Deployments Map" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
