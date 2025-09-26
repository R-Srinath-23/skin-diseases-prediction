import React, { useState } from 'react';
import './App.css';


function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
  e.preventDefault();

  const formData = new FormData();
  formData.append('image', image);

  setLoading(true);      
  setPrediction('');
  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: formData,
    });

    const contentType = response.headers.get("content-type");

    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text(); // 🔍 this will show raw HTML
      setPrediction("Error: Non-JSON response. Response was:\n" + text);
      return;
    }

    const data = await response.json();
    setPrediction(data.prediction || data.error);
  } catch (error) {
    setPrediction('Error: ' + error.message);
  }
  finally {
    setLoading(false);   // ✅ stop loading
  }
};


  return (
    <div className="App">
      <h1>Skin Disease Classifier</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit">Predict</button>
      </form>
      {loading && <p>Loading...</p>}
      {prediction && <p><strong>Prediction:</strong> {prediction}</p>}
    </div>
  );
}

export default App;
