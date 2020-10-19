import React from "react";
import "./App.css";
import Classifier from "./Classifier";
import ModelData from "./ModelData";

function App() {
  return (
    <div className="App">
      <Classifier />
      <ModelData
        k={325}
        trainingInstances={256}
        precision={0.56}
        recall={0.2}
      />
      <Classifier />
    </div>
  );
}

export default App;
