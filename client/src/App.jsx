import "./App.css";
import Classify from "./Classify";
import { useState } from "react";

function App() {
  const [classification, setClassification] = useState(null);
  const [text, setText] = useState(null);

  const getText = async (retrievedText) => {
    setText(retrievedText);
    const classification = await fetch(`http://127.0.0.1:5000/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: retrievedText }),
    })
      .then((res) => res.json())
      .then((res) => res.classification);

    setClassification(classification);
  };

  return (
    <>
      <h2>Sentiment Analysis</h2>
      <Classify sendText={getText} />
      {classification && (
        <>
          <h3>
            "{text}" is: {classification}
          </h3>
          {classification == "Positive" ? (
            <>
              <p
                style={{
                  fontSize: "100px",
                  margin: "0px",
                }}
                className="emoji"
              >
                &#128513;
              </p>
            </>
          ) : (
            <>
              <p
                style={{
                  fontSize: "100px",
                  margin: "0px",
                }}
                className="emoji"
              >
                &#128530;
              </p>
            </>
          )}
        </>
      )}
    </>
  );
}

export default App;
