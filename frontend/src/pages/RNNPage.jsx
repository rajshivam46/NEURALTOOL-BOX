import { useState } from 'react';
import { UploadCloud, CheckCircle, AlertCircle, GitBranch, Cpu } from 'lucide-react';
import MetricsChart from '../components/MetricsChart';
import { uploadData, trainModel, predictModel } from '../api';

const RNNPage = ({ sessionId, setSessionId }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [seqLength, setSeqLength] = useState(5);
  const [hiddenDim, setHiddenDim] = useState(8);
  const [metrics, setMetrics] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [predictInputs, setPredictInputs] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      setLoading(true); setError('');
      const res = await uploadData(file, sessionId);
      setSessionId(res.session_id); setData(res);
      if (res.columns && res.columns.length > 0) {
        setTargetCol(res.columns[0]);
      }
    } catch (err) { setError(err.message || 'Error uploading file'); } finally { setLoading(false); }
  };

  const handleTrain = async () => {
    try {
      setTrainingStatus('training'); setError(''); setMetrics(null);
      const res = await trainModel('rnn', { session_id: sessionId, target_col: targetCol, seq_length: parseInt(seqLength), hidden_dim: parseInt(hiddenDim), epochs: 100 });
      setMetrics({ errors: res.errors, accuracies: [] });
      setTrainingStatus('completed');
      setPredictInputs(Array(parseInt(seqLength)).fill(0));
    } catch (err) { setError(err.message || 'Error training model'); setTrainingStatus(''); }
  };

  const handlePredict = async () => {
    try {
      const res = await predictModel('rnn', { session_id: sessionId, feature_values: predictInputs.map(f => parseFloat(f)) });
      setPredictionResult(res.prediction);
    } catch (err) { setError(err.message || 'Error predicting'); }
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h2 className="page-title">Recurrent Neural Network</h2>
        <p className="page-subtitle">Upload time-series continuous data to forecast the next sequential step via BPTT.</p>
      </div>

      {error && <div className="alert alert-error"><AlertCircle size={20} /> {error}</div>}

      <div className="card">
        <h3 className="card-title">1. Upload Sequence Data</h3>
        {!data ? (
          <label className="upload-zone">
            <div className="upload-icon flex justify-center text-center"><UploadCloud size={48} /></div>
            <p style={{fontWeight: 500}}>Click/drag CSV sequence to upload</p>
            <input type="file" accept=".csv" onChange={handleFileUpload} style={{display: 'none'}} />
            {loading && <div className="mt-4 flex justify-center text-center"><div className="spinner"></div></div>}
          </label>
        ) : (
          <div>
            <div className="alert alert-success"><CheckCircle size={20} /> Loaded data successfully!</div>
            <div className="grid grid-cols-2">
              <div className="form-group"><label className="form-label">Sequence Target Variable</label><select className="form-control" value={targetCol} onChange={e => setTargetCol(e.target.value)}>{data.columns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
              <div className="form-group"><label className="form-label">Window Length (T steps to view)</label><input type="number" className="form-control" min="2" max="20" value={seqLength} onChange={e => setSeqLength(e.target.value)} /></div>
            </div>
          </div>
        )}
      </div>

      {data && (
        <div className="card">
          <h3 className="card-title">2. Configure & Train</h3>
          <div className="form-group" style={{maxWidth: '300px'}}>
            <label className="form-label">Hidden Dimension</label>
            <input type="number" className="form-control" value={hiddenDim} onChange={e => setHiddenDim(e.target.value)} />
          </div>
          <button className="btn btn-primary mt-4" onClick={handleTrain} disabled={trainingStatus === 'training'}>
            {trainingStatus === 'training' ? <div className="spinner"></div> : <GitBranch size={20} />} {trainingStatus === 'training' ? 'Training...' : 'Train Sequence Model'}
          </button>
        </div>
      )}

      {metrics && <MetricsChart errors={metrics.errors} accuracies={metrics.accuracies} labels={['Training Loss (BPTT)']} />}

      {trainingStatus === 'completed' && (
        <div className="card mt-4">
          <h3 className="card-title">3. Forecast Next Phase</h3>
          <div className="grid grid-cols-4">
            {predictInputs.map((val, idx) => (
              <div className="form-group" key={`seq-${idx}`}><label className="form-label">Step T-{parseInt(seqLength) - idx}</label><input type="number" step="any" className="form-control" value={val} onChange={e => { const newInputs = [...predictInputs]; newInputs[idx] = e.target.value; setPredictInputs(newInputs); }} /></div>
            ))}
          </div>
          <button className="btn btn-success mt-4" onClick={handlePredict}><Cpu size={20} /> Generate Forecast</button>
          {predictionResult !== null && <div className="alert alert-success mt-4"><strong>Forecasted Next Value:</strong> &nbsp;{predictionResult.toFixed(4)}</div>}
        </div>
      )}
    </div>
  );
};
export default RNNPage;
