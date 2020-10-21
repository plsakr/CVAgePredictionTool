import React from "react";
import "./App.css";
import Classifier from "./Cards/Classifier";
import ModelData from "./Cards/ModelData";
import TrainingCard from "./Cards/TrainingCard";
import TrainingMenu from "./TrainingMenu";

import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  var isTrainingOpen = false;
  const trainingMenu = React.createRef();

  const handleTrainButton = (e) => {
    console.log("TRAINING CLICKED");
    trainingMenu.current.open();
  };

  return (
    <div className="App">
      <Classifier />
      <ModelData
        k={325}
        trainingInstances={256}
        precision={0.56}
        recall={0.2}
      />
      <TrainingCard onTrain={handleTrainButton} />
      <TrainingMenu ref={trainingMenu} />
    </div>
  );
}

export default App;
