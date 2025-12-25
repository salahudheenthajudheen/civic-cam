import React, { useRef } from 'react';
import FaceDetectionCanvas from './FaceDetectionCanvas';

function IncidentDetails({ incident }) {
  const mainImageRef = useRef(null);
  const suspectImageRef = useRef(null);

  if (!incident) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '60px 20px',
        color: 'rgba(255, 255, 255, 0.5)'
      }}>
        <p>No incident selected</p>
      </div>
    );
  }

  return (
    <>
      <div className="incident-content">
        <div className="main-incident-view">
          <div className="incident-image-container">
            <img
              ref={mainImageRef}
              src={incident.imageUrl}
              alt="Incident"
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
            <FaceDetectionCanvas
              imageRef={mainImageRef}
              faceBox={incident.faceBox}
              isSuspectFace={false}
            />
          </div>
        </div>

        <div className="details-panel">
          <div className="detail-item">
            <span className="detail-icon">🕐</span>
            <div className="detail-content">
              <div className="detail-label">Timestamp</div>
              <div className="detail-value">{incident.timestamp}</div>
            </div>
          </div>
          <div className="detail-item">
            <span className="detail-icon">📍</span>
            <div className="detail-content">
              <div className="detail-label">Camera</div>
              <div className="detail-value">{incident.camera}</div>
            </div>
          </div>
          <div className="detail-item">
            <span className="detail-icon">🚛</span>
            <div className="detail-content">
              <div className="detail-label">Vehicle Status</div>
              <div className="detail-value">{incident.vehicleStatus}</div>
            </div>
          </div>
          <div className="detail-item">
            <span className="detail-icon">{incident.vehicleDetected ? '✅' : '❌'}</span>
            <div className="detail-content">
              <div className="detail-label">Vehicle Detected</div>
              <div className="detail-value" style={{
                color: incident.vehicleDetected ? '#22c55e' : '#ef4444'
              }}>
                {incident.vehicleDetected ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
          {incident.objectType && (
            <div className="detail-item">
              <span className="detail-icon">🗑️</span>
              <div className="detail-content">
                <div className="detail-label">Object Type</div>
                <div className="detail-value">{incident.objectType}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="suspect-face-section">
        <h3>👤 Suspect Face</h3>
        <div className="suspect-face-container">
          <img
            ref={suspectImageRef}
            src={incident.imageUrl}
            alt="Suspect"
          />
          <FaceDetectionCanvas
            imageRef={suspectImageRef}
            faceBox={incident.faceBox}
            isSuspectFace={true}
          />
        </div>
      </div>
    </>
  );
}

export default IncidentDetails;
