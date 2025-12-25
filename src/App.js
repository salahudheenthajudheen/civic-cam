import React, { useState, useEffect, useMemo, useCallback } from 'react';
import LiveFeed from './components/LiveFeed';
import { fetchEvents, fetchStats, EventSocket, transformEvent, checkApiHealth } from './api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [incidents, setIncidents] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState({ totalIncidents: 0, lastIncident: '' });
  const [selectedIncident, setSelectedIncident] = useState(null);

  const handleNewEvent = useCallback((event) => {
    const transformed = transformEvent(event);
    setIncidents(prev => {
      if (prev.some(e => e.id === transformed.id)) return prev;
      return [transformed, ...prev].slice(0, 20);
    });
  }, []);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      const apiAvailable = await checkApiHealth();
      if (apiAvailable) {
        const [events, statsData] = await Promise.all([fetchEvents(10), fetchStats()]);
        if (events?.length > 0) setIncidents(events.map(transformEvent));
        if (statsData) setStats(statsData);
        setIsConnected(true);
      }
      setIsLoading(false);
    }
    loadData();
  }, []);

  useEffect(() => {
    const socket = new EventSocket(handleNewEvent, () => setIsConnected(true), () => setIsConnected(false));
    socket.connect();
    return () => socket.disconnect();
  }, [handleNewEvent]);

  useEffect(() => {
    if (!isConnected) return;
    const interval = setInterval(async () => {
      const [events, statsData] = await Promise.all([fetchEvents(10), fetchStats()]);
      if (events) setIncidents(events.map(transformEvent));
      if (statsData) setStats(statsData);
    }, 5000);
    return () => clearInterval(interval);
  }, [isConnected]);

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#0a0a0f' }}>
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>Loading...</div>
      </div>
    );
  }

  return (
    <div style={{ background: '#0a0a0f', minHeight: '100vh', color: 'white', fontFamily: "'Inter', sans-serif" }}>
      {/* Minimal Header */}
      <header style={{
        padding: '16px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid rgba(255,255,255,0.1)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '20px' }}>🎯</span>
          <span style={{ fontWeight: '600', fontSize: '16px' }}>CivicCam</span>
          <span style={{
            padding: '4px 10px',
            borderRadius: '12px',
            fontSize: '11px',
            background: isConnected ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)',
            color: isConnected ? '#22c55e' : '#ef4444'
          }}>
            {isConnected ? '● Live' : '○ Offline'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '24px', fontSize: '13px', color: 'rgba(255,255,255,0.6)' }}>
          <span>Incidents: <strong style={{ color: '#00d4ff' }}>{stats.totalIncidents}</strong></span>
          <span>FPS: <strong style={{ color: '#22c55e' }}>{stats.detectionFps || '—'}</strong></span>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ display: 'grid', gridTemplateColumns: selectedIncident ? '1fr 400px' : '1fr 320px', gap: '1px', background: 'rgba(255,255,255,0.1)', height: 'calc(100vh - 53px)' }}>

        {/* Live Feed */}
        <section style={{ background: '#0a0a0f', position: 'relative' }}>
          {selectedIncident ? (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '14px', fontWeight: '500' }}>📋 Incident Details</span>
                <button
                  onClick={() => setSelectedIncident(null)}
                  style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', color: 'rgba(255,255,255,0.7)', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
                >
                  ← Back to Live
                </button>
              </div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
                <img src={selectedIncident.imageUrl} alt="Incident" style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '8px' }} />
              </div>
              <div style={{ padding: '16px', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', fontSize: '12px' }}>
                <div><span style={{ color: 'rgba(255,255,255,0.5)' }}>Time</span><br /><strong>{selectedIncident.timestamp}</strong></div>
                <div><span style={{ color: 'rgba(255,255,255,0.5)' }}>Object</span><br /><strong>{selectedIncident.objectType || 'Unknown'}</strong></div>
                <div><span style={{ color: 'rgba(255,255,255,0.5)' }}>Vehicle</span><br /><strong style={{ color: selectedIncident.vehicleDetected ? '#22c55e' : '#ef4444' }}>{selectedIncident.vehicleDetected ? 'Yes' : 'No'}</strong></div>
              </div>
            </div>
          ) : (
            <LiveFeed isConnected={isConnected} />
          )}
        </section>

        {/* Incidents List */}
        <aside style={{ background: '#0d0d12', overflowY: 'auto' }}>
          <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.1)', fontSize: '13px', fontWeight: '500', color: 'rgba(255,255,255,0.7)' }}>
            Recent Events
          </div>
          {incidents.length === 0 ? (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: 'rgba(255,255,255,0.3)', fontSize: '13px' }}>
              No incidents yet
            </div>
          ) : (
            incidents.map(inc => (
              <div
                key={inc.id}
                onClick={() => setSelectedIncident(inc)}
                style={{
                  display: 'flex',
                  gap: '12px',
                  padding: '12px 16px',
                  borderBottom: '1px solid rgba(255,255,255,0.05)',
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                  background: selectedIncident?.id === inc.id ? 'rgba(0,212,255,0.1)' : 'transparent'
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                onMouseLeave={e => e.currentTarget.style.background = selectedIncident?.id === inc.id ? 'rgba(0,212,255,0.1)' : 'transparent'}
              >
                <img
                  src={inc.imageUrl}
                  alt=""
                  style={{ width: '48px', height: '48px', borderRadius: '6px', objectFit: 'cover', background: '#1a1a2e' }}
                  onError={e => e.target.style.display = 'none'}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.9)', marginBottom: '4px' }}>{inc.timestamp}</div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>{inc.objectType || 'Littering detected'}</div>
                </div>
              </div>
            ))
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;
