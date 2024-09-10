import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Optional for styling

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();  // Prevent form refresh
    setLoading(true);  // Set loading state
    setError(null);  // Clear previous errors
    setAnswer('');  // Clear previous answer

    try {
      const response = await axios.post('http://localhost:5000/ask', { question });
      setAnswer(response.data.answer);
    } catch (err) {
      setError('Failed to get an answer. Please try again.');
    } finally {
      setLoading(false);  // Stop loading
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Application</h1>
        <form onSubmit={handleSubmit}>
          <label>
            Ask a question:
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question"
              required
            />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? 'Getting Answer...' : 'Submit'}
          </button>
        </form>

        {/* Display the answer or loading/error state */}
        {loading && <p>Loading...</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {answer && (
          <div className="answer-box">
            <h2>Answer:</h2>
            <p>{answer}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
