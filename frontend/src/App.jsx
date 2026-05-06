import { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import PerceptronPage from './pages/PerceptronPage';
import BackpropPage from './pages/BackpropPage';
import RNNPage from './pages/RNNPage';
import LossPage from './pages/LossPage';
import CNNPage from './pages/CNNPage';
import DashboardPage from './pages/DashboardPage';

function App() {
  const [sessionId, setSessionId] = useState('');

  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content" style={{ padding: 0 }}>
        <Header />
        <div style={{ padding: '2rem' }}>
          <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/perceptron" element={<PerceptronPage sessionId={sessionId} setSessionId={setSessionId} />} />
          <Route path="/backprop" element={<BackpropPage sessionId={sessionId} setSessionId={setSessionId} />} />
          <Route path="/rnn" element={<RNNPage sessionId={sessionId} setSessionId={setSessionId} />} />
          <Route path="/loss" element={<LossPage />} />
          <Route path="/cnn" element={<CNNPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
