import { useState, useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';

const LossPage = () => {
  const [lossType, setLossType] = useState('MSE');
  const [groundTruth, setGroundTruth] = useState(0);

  const chartData = useMemo(() => {
    const data = [];
    if (lossType === 'MSE') {
      for(let i = -10; i <= 10; i+=0.2) {
         data.push({ x: i.toFixed(1), loss: Math.pow(groundTruth - i, 2) });
      }
    } else if (lossType === 'MAE') {
      for(let i = -10; i <= 10; i+=0.2) {
         data.push({ x: i.toFixed(1), loss: Math.abs(groundTruth - i) });
      }
    } else if (lossType === 'BCE') {
      for(let i = 0.001; i < 0.999; i+=0.01) {
         data.push({ x: i.toFixed(2), loss: groundTruth === 1 ? -Math.log(i) : -Math.log(1 - i) });
      }
    }
    return data;
  }, [lossType, groundTruth]);

  const config = {
    MSE: { color: '#58a6ff', labelName: 'MSE Gradient Path', xLabel: 'Network Output (ŷ)', yLabel: 'Loss Penalty' },
    MAE: { color: '#3fb950', labelName: 'MAE Gradient Path', xLabel: 'Network Output (ŷ)', yLabel: 'Loss Penalty' },
    BCE: { color: '#bf8700', labelName: `BCE Loss (y=${groundTruth})`, xLabel: 'Predicted Probability (ŷ)', yLabel: 'Logarithmic Penalty' }
  }[lossType];

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h2 className="page-title">Continuous Loss Analysis</h2>
        <p className="page-subtitle">Visually comprehend how optimization goals shift depending on the choice of loss metric.</p>
      </div>

      <div className="card">
         <div className="grid grid-cols-2">
            <div className="form-group">
               <label className="form-label">Select Optimization Metric</label>
               <select className="form-control" value={lossType} onChange={e => {
                  setLossType(e.target.value);
                  if (e.target.value === 'BCE' && groundTruth !== 0 && groundTruth !== 1) setGroundTruth(1);
               }}>
                 <option value="MSE">Mean Squared Error (MSE)</option>
                 <option value="MAE">Mean Absolute Error (MAE)</option>
                 <option value="BCE">Binary Cross-Entropy (BCE)</option>
               </select>
            </div>
            <div className="form-group">
               <label className="form-label">Ground Truth Object (y)</label>
               {lossType === 'BCE' ? (
                 <select className="form-control" value={groundTruth} onChange={e => setGroundTruth(parseInt(e.target.value))}>
                    <option value={1}>Class 1</option>
                    <option value={0}>Class 0</option>
                 </select>
               ) : (
                 <input type="number" step="any" className="form-control" value={groundTruth} onChange={e => setGroundTruth(parseFloat(e.target.value) || 0)} />
               )}
            </div>
         </div>
         <div className="mt-4" style={{ backgroundColor: 'rgba(0,0,0,0.2)', padding:'1.5rem', borderRadius: 'var(--radius-md)' }}>
            <h3 style={{marginBottom: '1rem', color: '#fff', fontSize: '1.1rem'}}>Gradient Path Generation</h3>
            <div style={{ height: '400px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="x" stroke="#8b949e" tick={{fill: '#8b949e'}} label={{ value: config.xLabel, position: 'insideBottom', offset: -10, fill: '#8b949e' }} />
                  <YAxis stroke="#8b949e" tick={{fill: '#8b949e'}} label={{ value: config.yLabel, angle: -90, position: 'insideLeft', fill: '#8b949e' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#161b22', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }} />
                  {lossType !== 'BCE' && <ReferenceLine x={groundTruth.toFixed(1)} stroke="rgba(239, 68, 68, 0.5)" strokeDasharray="3 3" />}
                  <Line type="monotone" dataKey="loss" name={config.labelName} stroke={config.color} strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
         </div>
      </div>
    </div>
  );
};

export default LossPage;
