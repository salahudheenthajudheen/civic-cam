/**
 * CivicCam API Client
 * 
 * Handles communication with the CivicCam Python backend.
 * Provides REST API calls and WebSocket connection for real-time updates.
 */

// For production, set REACT_APP_API_URL to your VPS IP
// Example: REACT_APP_API_URL=http://your-vps-ip:8000
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || API_BASE_URL.replace('http', 'ws') + '/ws';

/**
 * Fetch recent littering events from API
 * @param {number} limit - Maximum events to fetch
 * @returns {Promise<Array>} Array of event objects
 */
export async function fetchEvents(limit = 20) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/events?limit=${limit}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to fetch events:', error);
        return null;
    }
}

/**
 * Fetch detection statistics
 * @returns {Promise<Object>} Statistics object
 */
export async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to fetch stats:', error);
        return null;
    }
}

/**
 * WebSocket connection manager for real-time updates
 */
export class EventSocket {
    constructor(onEvent, onConnect, onDisconnect) {
        this.ws = null;
        this.onEvent = onEvent;
        this.onConnect = onConnect || (() => { });
        this.onDisconnect = onDisconnect || (() => { });
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.pingInterval = null;
    }

    connect() {
        try {
            this.ws = new WebSocket(WS_URL);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.onConnect();

                // Start ping interval
                this.pingInterval = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send('ping');
                    }
                }, 25000);
            };

            this.ws.onmessage = (event) => {
                try {
                    if (event.data === 'pong') return;

                    const message = JSON.parse(event.data);

                    if (message.type === 'new_event') {
                        console.log('New event received:', message.data);
                        this.onEvent(message.data);
                    } else if (message.type === 'initial') {
                        console.log('Initial events received:', message.data.length);
                    }
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.cleanup();
                this.onDisconnect();
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay);
        }
    }

    cleanup() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    disconnect() {
        this.cleanup();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

/**
 * Transform API event to frontend format
 * @param {Object} apiEvent - Event from API
 * @returns {Object} Event in frontend format
 */
export function transformEvent(apiEvent) {
    return {
        id: apiEvent.id,
        timestamp: apiEvent.timestamp,
        camera: apiEvent.camera || 'Camera 1',
        vehicleDetected: apiEvent.vehicleDetected || false,
        vehicleStatus: apiEvent.vehicleStatus || 'No vehicle detected',
        imageUrl: apiEvent.imageUrl.startsWith('http')
            ? apiEvent.imageUrl
            : `${API_BASE_URL}${apiEvent.imageUrl}`,
        faceDetected: apiEvent.faceDetected || false,
        faceBox: apiEvent.faceBox || { x: 200, y: 100, width: 150, height: 180 },
        objectType: apiEvent.objectType || 'unknown',
        confidence: apiEvent.confidence || 0
    };
}

/**
 * Check if API is available
 * @returns {Promise<boolean>}
 */
export async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        return response.ok;
    } catch {
        return false;
    }
}

export { API_BASE_URL, WS_URL };
