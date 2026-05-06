import axios from 'axios';

const api = axios.create({
  baseURL: '/api'
});

export const uploadData = async (file, sessionId) => {
  const formData = new FormData();
  formData.append('file', file);
  if (sessionId) formData.append('session_id', sessionId);
  
  const response = await api.post('/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const trainModel = async (modelType, payload) => {
  const response = await api.post(`/models/train/${modelType}`, payload);
  return response.data;
};

export const predictModel = async (modelType, payload) => {
  const response = await api.post(`/predict/${modelType}`, payload);
  return response.data;
};

// CNN API endpoints
export const registerFace = async (payload) => {
  const response = await api.post('/cnn/register', payload);
  return response.data;
};

export const markAttendance = async (payload) => {
  const response = await api.post('/cnn/attendance', payload);
  return response.data;
};

export const getLogs = async () => {
  const response = await api.get('/cnn/logs');
  return response.data;
};

export const getDashboardStats = async () => {
  const response = await api.get('/data/dashboard-stats');
  return response.data;
};

export default api;
