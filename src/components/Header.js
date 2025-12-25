import React from 'react';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-text">
          <h1>🎯 CIVICCAM</h1>
          <p className="subtitle">
            AI-powered surveillance for detecting illegal waste dumping, capturing evidence, and automatic authority notification.
          </p>
        </div>
        <button className="deploy-label">
          ⚡ Deploy
        </button>
      </div>
    </header>
  );
}

export default Header;
