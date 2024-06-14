import React from "react";
import { useState } from "react";

function Classify(props) {
  const [text, setText] = useState(null);

  const handleChange = (e) => {
    setText(e.target.value);
  };

  const sendText = (e) => {
    e.preventDefault();
    props.sendText(text);
  };

  return (
    <>
      <form className="GUI" onSubmit={sendText}>
        <input placeholder="Classify Text" onChange={handleChange}></input>
        <button>Classify</button>
      </form>
    </>
  );
}

export default Classify;
