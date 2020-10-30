import React, { useEffect } from "react";
import "./App.css";
import Classifier from "./Cards/Classifier";
import ModelData from "./Cards/ModelData";
import TrainingCard from "./Cards/TrainingCard";
import TrainingMenu from "./TrainingMenu";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/css/bootstrap.css"; // or include from a CDN
import "react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css";
import TrainingLoad from "./TrainingLoad";

const backend = "http://127.0.0.1:5000";
const defaultModel = "pretrained_knn_model";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      modelName: "",
      modelScores: undefined,
      modelParams: {},
      isTraining: false,
      trainingId: -1,
    };
    this.isTrainingOpen = false;
    this.trainingMenu = React.createRef();
  }

  componentDidMount() {
    fetch(`${backend}/info`).then((result) => {
      if (result.ok) {
        result.json().then((body) => {
          console.log(body);
          this.setState({
            modelName: body.model_name,
            modelScores: body.model_scores,
            modelParams: body.model_params,
            isTraining: body.isTraining,
          });
        });
      }
    });
  }

  handlePredict(base64) {}

  handleTrainButton(e) {
    console.log("TRAINING CLICKED");
    this.trainingMenu.current.open();
  }

  handleResetButton(e) {
    console.log("HANDLING MODEL RESET!");
  }

  handleTrainNewModel(jobId) {
    console.log("HANDLING NEW MODEL CREATION!");
    this.setState({ isTraining: true, trainingId: jobId });
  }

  handleTrainDone() {
    console.log("Received train finished!");
    this.setState({ isTraining: false, trainingId: -1 });
  }

  render() {
    if (typeof this.state.modelScores !== "undefined") {
      return (
        <div className="App">
          <Classifier onPredict={this.handlePredict.bind(this)} />
          <ModelData
            k={this.state.modelParams.K}
            trainingInstances={this.state.modelParams.train_nbr}
            testingInstances={this.state.modelParams.test_nbr}
            precisionY={this.state.modelScores.Young.precision}
            recallY={this.state.modelScores.Young.recall}
            precisionO={this.state.modelScores.Old.precision}
            recallO={this.state.modelScores.Old.recall}
          />
          <TrainingCard
            onTrain={this.handleTrainButton.bind(this)}
            onReset={this.handleResetButton.bind(this)}
            modelName={this.state.modelName}
          />
          <TrainingMenu
            ref={this.trainingMenu}
            onTrain={this.handleTrainNewModel.bind(this)}
          />
          <TrainingLoad
            isTraining={this.state.isTraining}
            jobId={this.state.trainingId}
          />
        </div>
      );
    } else {
      return <div></div>;
    }
  }
}

export default App;
