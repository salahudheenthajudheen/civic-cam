import React from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function LiveFeed({ isConnected }) {
    if (!isConnected) {
        return (
            <div style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#0a0a0f',
                color: 'rgba(255,255,255,0.4)'
            }}>
                <span style={{ fontSize: '48px', opacity: 0.3, marginBottom: '16px' }}>📹</span>
                <p style={{ fontSize: '14px' }}>Camera offline</p>
                <p style={{ fontSize: '12px', marginTop: '4px' }}>Start API server to view feed</p>
            </div>
        );
    }

    return (
        <div style={{ height: '100%', position: 'relative', background: '#000' }}>
            {/* Live badge */}
            <div style={{
                position: 'absolute',
                top: '12px',
                left: '12px',
                background: '#ef4444',
                color: 'white',
                padding: '4px 10px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                zIndex: 10
            }}>
                <span style={{ width: '6px', height: '6px', background: 'white', borderRadius: '50%', animation: 'pulse 1s infinite' }}></span>
                LIVE
            </div>

            <img
                src={`${API_BASE_URL}/api/video`}
                alt="Live Feed"
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                }}
            />
        </div>
    );
}

export default LiveFeed;
