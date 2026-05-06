import { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Save, UserCheck, AlertCircle, RefreshCw } from 'lucide-react';
import { registerFace, markAttendance, getLogs } from '../api';

const CNNPage = () => {
  const [activeTab, setActiveTab] = useState('attendance');
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  
  const [registerName, setRegisterName] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'error' | 'success', msg: '' }
  
  const [logs, setLogs] = useState([]);

  const fetchLogs = async () => {
    try {
      const data = await getLogs();
      setLogs(data);
    } catch(err) {
      console.error("Failed to fetch logs", err);
    }
  };

  useEffect(() => {
    if (activeTab === 'logs') {
      fetchLogs();
    }
  }, [activeTab]);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
  }, [webcamRef]);

  const retake = () => {
    setImgSrc(null);
    setStatus(null);
  };

  const handleRegister = async () => {
    if (!registerName.trim()) {
      setStatus({ type: 'error', msg: 'Please enter a name before registering.' });
      return;
    }
    setLoading(true); setStatus(null);
    try {
      const res = await registerFace({ name: registerName, image: imgSrc });
      setStatus({ type: 'success', msg: res.message || 'Face Registered successfully!' });
      setRegisterName('');
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || err.message || 'Registration failed.' });
    } finally {
      setLoading(false);
    }
  };

  const handleAttendance = async () => {
    setLoading(true); setStatus(null);
    try {
      const res = await markAttendance({ image: imgSrc });
      if (res.match) {
        setStatus({ type: 'success', msg: `Identified: ${res.name} (Confidence: ${res.confidence}%) - Logged at ${res.timestamp}` });
      } else {
        setStatus({ type: 'error', msg: res.message || 'Face not recognized.' });
      }
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || err.message || 'Identification failed.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h2 className="page-title">CNN Facial Recognition</h2>
        <p className="page-subtitle">Deep learning face embedding and attendance marker using WebCam.</p>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        <button className={`btn ${activeTab === 'attendance' ? 'btn-primary' : ''}`} onClick={() => {setActiveTab('attendance'); setStatus(null); setImgSrc(null);}}>
           <UserCheck size={18} /> Mark Attendance
        </button>
        <button className={`btn ${activeTab === 'register' ? 'btn-primary' : ''}`} onClick={() => {setActiveTab('register'); setStatus(null); setImgSrc(null);}}>
           <Save size={18} /> Register Face
        </button>
        <button className={`btn ${activeTab === 'logs' ? 'btn-primary' : ''}`} onClick={() => {setActiveTab('logs'); setStatus(null);}}>
           <RefreshCw size={18} /> View Logs
        </button>
      </div>

      {status && (
        <div className={`alert ${status.type === 'error' ? 'alert-error' : 'alert-success'}`}>
          <AlertCircle size={20} /> {status.msg}
        </div>
      )}

      {(activeTab === 'register' || activeTab === 'attendance') && (
        <div className="card" style={{ maxWidth: '640px', margin: '0 auto' }}>
          <h3 className="card-title justify-center text-center mb-4">
            {activeTab === 'register' ? 'Step 1: Save User to DB' : 'Scan Face for Attendance'}
          </h3>
          
          <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', backgroundColor: '#000', marginBottom: '1rem', border: '1px solid var(--border-color)' }}>
            {!imgSrc ? (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                style={{ width: '100%', display: 'block' }}
              />
            ) : (
              <img src={imgSrc} alt="snapshot" style={{ width: '100%', display: 'block' }} />
            )}
          </div>

          <div className="flex justify-center gap-2">
            {!imgSrc ? (
              <button className="btn btn-primary" onClick={capture}>
                <Camera size={20} /> Capture Snapshot
              </button>
            ) : (
              <>
                <button className="btn" style={{backgroundColor: 'var(--bg-secondary)', color: 'white'}} onClick={retake}>
                   Retake Photo
                </button>
                {activeTab === 'register' ? (
                  <div className="flex gap-2">
                    <input type="text" className="form-control" placeholder="Enter full name" value={registerName} onChange={e => setRegisterName(e.target.value)} />
                    <button className="btn btn-success" onClick={handleRegister} disabled={loading}>
                      {loading ? <div className="spinner"></div> : 'Register CNN'}
                    </button>
                  </div>
                ) : (
                  <button className="btn btn-success" onClick={handleAttendance} disabled={loading}>
                     {loading ? <div className="spinner"></div> : <UserCheck size={20} />}
                     {loading ? 'Analyzing...' : 'Identify Face'}
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="card">
          <h3 className="card-title flex items-center">
            Attendance Records
            <button className="btn" style={{ marginLeft: 'auto', padding: '0.5rem', background: 'transparent' }} onClick={fetchLogs}>
              <RefreshCw size={16} color="var(--accent-blue)" />
            </button>
          </h3>
          
          <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
             <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
               <thead>
                 <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                   <th style={{ padding: '0.75rem' }}>Name</th>
                   <th style={{ padding: '0.75rem' }}>Timestamp</th>
                   <th style={{ padding: '0.75rem' }}>CNN Confidence</th>
                 </tr>
               </thead>
               <tbody>
                 {logs.length > 0 ? logs.map((log, idx) => (
                   <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                     <td style={{ padding: '0.75rem', fontWeight: 500 }}>{log.Name}</td>
                     <td style={{ padding: '0.75rem', color: 'var(--text-secondary)' }}>{log.Timestamp}</td>
                     <td style={{ padding: '0.75rem' }}>
                       <span style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#6ee7b7', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                         {log.Confidence}%
                       </span>
                     </td>
                   </tr>
                 )) : (
                   <tr>
                     <td colSpan="3" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>No attendance records found.</td>
                   </tr>
                 )}
               </tbody>
             </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default CNNPage;
