import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import Cookies from 'js-cookie';

function App() {
  const [inputText, setInputText] = useState('');

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const getToken = async () => {
    try {
      // Replace '/get-csrf-token/' with your actual endpoint for fetching the CSRF token
      await axios.get('http://localhost:8000/get-csrf-token/', { withCredentials: true });
      return Cookies.get('csrftoken');
    } catch (error) {
      console.error('Unable to fetch CSRF token', error);
      return null;
    }
  };
  
  const handleFactCheck = async () => {
    let csrftoken = Cookies.get('csrftoken');
  
    // If CSRF token is undefined, try to fetch it
    if (!csrftoken) {
      console.log('CSRF token is undefined, attempting to fetch...');
      csrftoken = await getToken();
      if (!csrftoken) {
        console.error('CSRF token could not be retrieved');
        return;
      }
    }
  
    try {
      const response = await axios.post(
        'http://localhost:8000/calculate-confidence/',
        {
          uploaded_statement: inputText,
        },
        {
          headers: {
            'X-CSRFToken': csrftoken,
            'Content-Type': 'multipart/form-data',
          },
          withCredentials: true,
        }
      );
  
      console.log(response.data);
      alert(`Confidence score: ${response.data.confidence}`);
    } catch (error) {
      console.error('There was an error fetching the confidence score!', error);
    }
  };

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
          <textarea value={inputText} onChange={handleInputChange}></textarea>
          <br />
          <br />
          <button onClick={handleFactCheck} className="cta-button">
            Fact Check Now
          </button>
        </div>
      </header>
    </div>
  );
}

export default App;