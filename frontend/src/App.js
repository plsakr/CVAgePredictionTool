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

  const handleResetButton = (e) => {
    console.log("HANDLING MODEL RESET!");
  };

  const handleTrainNewModel = () => {
    console.log("HANDLING NEW MODEL CREATION!");
    // this method should take all needed info as parameters.
    // then create the backend request to train the model
    // somehow create a spinner that disables everything until training is done
    // after training, reenable reset button
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
      <TrainingCard onTrain={handleTrainButton} onReset={handleResetButton} />
      <TrainingMenu ref={trainingMenu} onTrain={handleTrainNewModel} />
    </div>
  );
}

export default App;
