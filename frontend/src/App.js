import React from "react";
import "./App.css";
import Classifier from "./Classifier";
import ModelData from "./ModelData";
import TrainingCard from "./TrainingCard";
import TrainingMenu from "./TrainingMenu";

function App() {
  var isTrainingOpen = false;
  const trainingMenu = React.createRef();

  const handleTrainButton = (e) => {
    console.log("TRAINING CLICKED");
    trainingMenu.current.handleTrainingClick();
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
      <TrainingMenu open={true} ref={trainingMenu} />
    </div>
  );
}

export default App;
