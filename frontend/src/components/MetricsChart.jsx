import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const MetricsChart = ({ errors = [], accuracies = [], labels = ['Loss (Error)', 'Accuracy'] }) => {
  if (errors.length === 0) return null;

  const data = errors.map((err, idx) => ({
    epoch: `Epoch ${idx + 1}`,
    loss: err,
    ...(accuracies.length > 0 && { accuracy: accuracies[idx] })
  }));

  return (
    <div className="card mt-4" style={{ display: 'flex', flexDirection: 'column' }}>
      <h3 className="card-title">Training Metrics</h3>
      <div style={{ width: '100%', height: '350px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" vertical={false} />
            <XAxis dataKey="epoch" stroke="#6C757D" tick={{fill: '#6C757D', fontSize: 12}} dy={10} />
            
            <YAxis yAxisId="left" stroke="#6C757D" tick={{fill: '#6C757D', fontSize: 12}} dx={-10} 
                   label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#6C757D' }} />
                   
            {accuracies.length > 0 && (
              <YAxis yAxisId="right" orientation="right" stroke="#6C757D" tick={{fill: '#6C757D', fontSize: 12}} dx={10} 
                     domain={[0, 1.05]} label={{ value: 'Accuracy', angle: 90, position: 'insideRight', fill: '#6C757D' }} />
            )}
            
            <Tooltip 
              contentStyle={{ backgroundColor: '#FFFFFF', borderColor: 'rgba(0,0,0,0.06)', borderRadius: '12px', color: '#1A1A1A', boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.05)' }}
              itemStyle={{ color: '#1A1A1A' }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            
            <Line yAxisId="left" type="monotone" dataKey="loss" name={labels[0]} stroke="#1B5E38" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
            {accuracies.length > 0 && (
              <Line yAxisId="right" type="monotone" dataKey="accuracy" name={labels[1] || 'Accuracy'} stroke="#8FBC8F" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MetricsChart;
