import React from "react";
import "./App.css";
import Card from "@material-ui/core/Card";
import Classifier from "./Classifier";

function App() {
  return (
    <div className="App">
      <Classifier className="cardClass" />
      <Classifier className="cardClass" />
      <Classifier className="cardClass" />
    </div>
  );
}

export default App;
