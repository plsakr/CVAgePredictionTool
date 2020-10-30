import React, { useEffect, useState } from "react";
import "./App.css";
import Classifier from "./Cards/Classifier";
import ModelData from "./Cards/ModelData";
import TrainingCard from "./Cards/TrainingCard";
import TrainingMenu from "./TrainingMenu";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/css/bootstrap.css"; // or include from a CDN
import "react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css";

const backend = "http://127.0.0.1:5000";
const defaultModel = "pretrained_knn_model";

function App() {
  const [modelName, setModelName] = useState("");
  const [modelScores, setModelScores] = useState();
  const [modelParams, setModelParams] = useState();

  var isTrainingOpen = false;
  const trainingMenu = React.createRef();

  useEffect(() => {
    fetch(`${backend}/info`).then((result) => {
      if (result.ok) {
        result.json().then((body) => {
          console.log(body);
          setModelParams(body.model_params);
          setModelName(body.model_name);
          setModelScores(body.model_scores);
        });
      }
    });
  }, []);

  const handlePredict = (base64) => {};

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

  const app = () => {
    if (typeof modelScores !== "undefined") {
      return (
        <div className="App">
          <Classifier onPredict={handlePredict} />
          <ModelData
            k={modelParams.K}
            trainingInstances={modelParams.train_nbr}
            testingInstances={modelParams.test_nbr}
            precisionY={modelScores.Young.precision}
            recallY={modelScores.Young.recall}
            precisionO={modelScores.Old.precision}
            recallO={modelScores.Old.recall}
          />
          <TrainingCard
            onTrain={handleTrainButton}
            onReset={handleResetButton}
            modelName={modelName}
          />
          <TrainingMenu ref={trainingMenu} onTrain={handleTrainNewModel} />
        </div>
      );
    } else {
      return <div></div>;
    }
  };

  return app();
}

export default App;
