import React from 'react';
import IncidentCard from './IncidentCard';

function RecentIncidents({ incidents, currentIncidentId, onSelectIncident }) {
  return (
    <div className="recent-incidents-section">
      <div className="section-header">
        <span className="section-icon">📌</span>
        <h2>Recent Incidents</h2>
      </div>
      <div className="incidents-list">
        {incidents.map(incident => (
          <IncidentCard
            key={incident.id}
            incident={incident}
            isActive={incident.id === currentIncidentId}
            onSelect={() => onSelectIncident(incident.id)}
          />
        ))}
      </div>
    </div>
  );
}

export default RecentIncidents;

