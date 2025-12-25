import React from 'react';

function IncidentCard({ incident, isActive, onSelect }) {
  return (
    <div 
      className={`incident-card ${isActive ? 'active' : ''}`}
      onClick={onSelect}
    >
      <img 
        src={incident.imageUrl} 
        alt={`Incident ${incident.id}`} 
        className="incident-thumbnail"
      />
      <div className="incident-info">
        <div className="incident-info-row">
          <span className="icon">🕐</span>
          <span className="incident-timestamp">{incident.timestamp}</span>
        </div>
        <div className="incident-info-row">
          <span className="icon">🚛</span>
          <span className="incident-vehicle">{incident.vehicleStatus}</span>
        </div>
      </div>
      <button 
        className={`view-button ${isActive ? 'active' : ''}`}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      >
        View
      </button>
    </div>
  );
}

export default IncidentCard;

