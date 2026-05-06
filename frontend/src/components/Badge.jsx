import React from 'react';

const Badge = ({ children, variant = "info", style = {} }) => {
  const baseStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '0.25rem 0.75rem',
    borderRadius: '99px',
    fontSize: '0.75rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    ...style
  };

  const variants = {
    success: { backgroundColor: 'var(--accent-light)', color: 'var(--accent-green)' },
    warning: { backgroundColor: '#FFF3CD', color: '#856404' },
    info: { backgroundColor: '#E2E3E5', color: 'var(--text-secondary)' },
    danger: { backgroundColor: '#F8D7DA', color: '#721C24' },
  };

  const activeStyle = variants[variant] || variants.info;

  return (
    <span style={{ ...baseStyle, ...activeStyle }}>
      {children}
    </span>
  );
};

export default Badge;
