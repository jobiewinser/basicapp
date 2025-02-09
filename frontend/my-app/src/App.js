import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <nav className="navbar">
        <div className="logo">ELEPHANT</div>
        <ul className="nav-links">
          <li>HOME</li>
          <li>UPLOAD</li>
          <li>ACCOUNT</li>
        </ul>
      </nav>
      <header className="App-header">
        <div className="content">
          <h1>FACT CHECK</h1>
          <div className="elephants">
            <img src="elephant2.png" alt="Elephant 1" className="elephant" />
            <img src="elephant2.png" alt="Elephant 2" className="elephant" />
          </div>
          <button className="cta-button">Fact Check Now.</button>
        </div>
      </header>
    </div>
  );
}

export default App;