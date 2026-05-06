import { Search, Bell, Settings } from 'lucide-react';

const Header = () => {
  return (
    <header style={{
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between',
      padding: '1rem 2rem',
      backgroundColor: 'var(--bg-card)',
      borderBottom: '1px solid var(--border-color)',
      marginBottom: '1rem'
    }}>
      
      {/* Centered Search Bar */}
      <div style={{
         display: 'flex', alignItems: 'center', backgroundColor: 'var(--bg-primary)',
         borderRadius: 'var(--radius-md)', padding: '0.5rem 1rem', width: '350px',
         border: '1px solid var(--border-color)', color: 'var(--text-secondary)'
      }}>
         <Search size={18} style={{ marginRight: '0.75rem' }} />
         <input 
            type="text" 
            placeholder="Search projects..." 
            style={{ 
              background: 'transparent', border: 'none', outline: 'none', 
              width: '100%', fontSize: '0.9rem', color: 'var(--text-primary)' 
            }}
         />
         <div style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            backgroundColor: '#fff', border: '1px solid var(--border-color)',
            borderRadius: '4px', padding: '2px 6px', fontSize: '0.7rem',
            fontWeight: 600
         }}>
           <span>⌘</span><span>F</span>
         </div>
      </div>

      {/* Right Side UI */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
         <div style={{ display: 'flex', gap: '0.75rem', color: 'var(--text-secondary)' }}>
            <button className="btn" style={{ padding: '0.5rem', border: 'none' }}><Bell size={20} /></button>
            <button className="btn" style={{ padding: '0.5rem', border: 'none' }}><Settings size={20} /></button>
         </div>
         
         <div style={{ width: '1px', height: '30px', backgroundColor: 'var(--border-color)' }}></div>
         
         <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
            <div style={{ textAlign: 'right' }}>
               <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>Alex Mercer</div>
               <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>alex@enterprise.ai</div>
            </div>
            <img 
               src="https://ui-avatars.com/api/?name=Alex+Mercer&background=1B5E38&color=fff&rounded=true" 
               alt="User Avatar" 
               style={{ width: '40px', height: '40px', borderRadius: '50%', boxShadow: '0 2px 5px rgba(0,0,0,0.1)' }}
            />
         </div>
      </div>
    </header>
  );
};

export default Header;
