import { useState } from 'react';
import { UploadCloud, CheckCircle, AlertCircle, Network, Cpu } from 'lucide-react';
import MetricsChart from '../components/MetricsChart';
import { uploadData, trainModel, predictModel } from '../api';

const BackpropPage = ({ sessionId, setSessionId }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [featureCols, setFeatureCols] = useState([]);
  const [lr, setLr] = useState(0.1);
  const [hiddenNodes, setHiddenNodes] = useState(8);
  const [epochs, setEpochs] = useState(1000);
  const [metrics, setMetrics] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [predictInputs, setPredictInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      setLoading(true); setError('');
      const res = await uploadData(file, sessionId);
      setSessionId(res.session_id); setData(res);
      if (res.columns && res.columns.length > 0) {
        setTargetCol(res.columns[res.columns.length - 1]);
        setFeatureCols(res.columns.slice(0, -1));
      }
    } catch (err) { setError(err.message || 'Error uploading file'); } finally { setLoading(false); }
  };

  const handleFeatureToggle = (col) => {
    setFeatureCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
  };

  const handleTrain = async () => {
    try {
      setTrainingStatus('training'); setError(''); setMetrics(null);
      const res = await trainModel('backprop', { session_id: sessionId, target_col: targetCol, feature_cols: featureCols, lr: parseFloat(lr), hidden_nodes: parseInt(hiddenNodes), epochs: parseInt(epochs) });
      setMetrics({ errors: res.errors, accuracies: res.accuracies });
      setTrainingStatus('completed');
      const initInputs = {}; featureCols.forEach(f => initInputs[f] = 0);
      setPredictInputs(initInputs);
    } catch (err) { setError(err.message || 'Error training model'); setTrainingStatus(''); }
  };

  const handlePredict = async () => {
    try {
      const res = await predictModel('backprop', { session_id: sessionId, feature_values: featureCols.map(f => parseFloat(predictInputs[f])) });
      setPredictionResult(res.prediction);
    } catch (err) { setError(err.message || 'Error predicting'); }
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h2 className="page-title">Multi-Layer Perceptron (Backprop)</h2>
        <p className="page-subtitle">Train a neural network with a hidden layer on your CSV data to capture non-linear relationships.</p>
      </div>

      {error && <div className="alert alert-error"><AlertCircle size={20} /> {error}</div>}

      <div className="card">
        <h3 className="card-title">1. Upload Dataset</h3>
        {!data ? (
          <label className="upload-zone">
            <div className="upload-icon flex justify-center text-center"><UploadCloud size={48} /></div>
            <p style={{fontWeight: 500}}>Click or drag CSV file to upload</p>
            <input type="file" accept=".csv" onChange={handleFileUpload} style={{display: 'none'}} />
            {loading && <div className="mt-4 flex justify-center text-center"><div className="spinner"></div></div>}
          </label>
        ) : (
          <div>
            <div className="alert alert-success"><CheckCircle size={20} /> Loaded {data.columns.length} numeric features.</div>
            <div className="grid grid-cols-2">
              <div className="form-group"><label className="form-label">Target Variable (y)</label><select className="form-control" value={targetCol} onChange={e => setTargetCol(e.target.value)}>{data.columns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
              <div className="form-group"><label className="form-label">Features (X)</label><div style={{maxHeight: '150px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius)', padding: '0.5rem'}}>{data.columns.filter(c => c !== targetCol).map(c => (<div key={c} style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem'}}><input type="checkbox" checked={featureCols.includes(c)} onChange={() => handleFeatureToggle(c)} /> <span>{c}</span></div>))}</div></div>
            </div>
          </div>
        )}
      </div>

      {data && (
        <div className="card">
          <h3 className="card-title">2. Configure & Train</h3>
          <div className="grid grid-cols-3">
            <div className="form-group"><label className="form-label">Hidden Nodes</label><input type="number" className="form-control" value={hiddenNodes} onChange={e => setHiddenNodes(e.target.value)} /></div>
            <div className="form-group"><label className="form-label">Learning Rate</label><input type="number" step="0.01" className="form-control" value={lr} onChange={e => setLr(e.target.value)} /></div>
            <div className="form-group"><label className="form-label">Epochs</label><input type="number" className="form-control" value={epochs} onChange={e => setEpochs(e.target.value)} /></div>
          </div>
          <button className="btn btn-primary mt-4" onClick={handleTrain} disabled={trainingStatus === 'training'}>
            {trainingStatus === 'training' ? <div className="spinner"></div> : <Network size={20} />} {trainingStatus === 'training' ? 'Training...' : 'Train MLP Model'}
          </button>
        </div>
      )}

      {metrics && <MetricsChart errors={metrics.errors} accuracies={metrics.accuracies} labels={['Loss (MSE)', 'Accuracy']} />}

      {trainingStatus === 'completed' && (
        <div className="card mt-4">
          <h3 className="card-title">3. Predict</h3>
          <div className="grid grid-cols-4">
            {featureCols.map(f => (
              <div className="form-group" key={f}><label className="form-label">{f}</label><input type="number" step="any" className="form-control" value={predictInputs[f] || 0} onChange={e => setPredictInputs({...predictInputs, [f]: e.target.value})} /></div>
            ))}
          </div>
          <button className="btn btn-success mt-4" onClick={handlePredict}><Cpu size={20} /> Predict Value</button>
          {predictionResult !== null && <div className="alert alert-success mt-4"><strong>Predicted Class:</strong> &nbsp;{predictionResult}</div>}
        </div>
      )}
    </div>
  );
};
export default BackpropPage;
