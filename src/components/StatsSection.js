import React from 'react';

function StatsSection({ statistics }) {
  return (
    <section className="stats-section">
      <div className="stat-card">
        <span className="stat-icon">🚨</span>
        <div className="stat-label">Total Incidents</div>
        <div className="stat-value">{statistics.totalIncidents || 0}</div>
      </div>
      <div className="stat-card">
        <span className="stat-icon">🚗</span>
        <div className="stat-label">With Vehicle</div>
        <div className="stat-value">{statistics.incidentsWithVehicle || 0}</div>
      </div>
      <div className="stat-card">
        <span className="stat-icon">⏱️</span>
        <div className="stat-label">Last Incident</div>
        <div className="stat-value" style={{ fontSize: statistics.lastIncident ? '18px' : '32px' }}>
          {statistics.lastIncident || '—'}
        </div>
      </div>
    </section>
  );
}

export default StatsSection;
