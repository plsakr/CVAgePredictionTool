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
            trainingId: body.trainingId,
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
    var req = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        isReset: true,
      }),
    };
    fetch(`${backend}/train`, req).then((res) => {
      if (!res.ok) {
        console.log("Could not reset model!");
      } else {
        res.json().then((body) => {
          const jobId = body.jobId;

          setTimeout(() => {
            fetch(`${backend}/jobinfo?jobId=${jobId}`).then((resetRes) => {
              if (resetRes.ok) {
                this.componentDidMount();
              }
            });
          }, 200);
        });
      }
    });
  }

  handleTrainNewModel(jobId) {
    console.log(`HANDLING NEW MODEL CREATION! JOBID ${jobId}`);
    this.setState({ isTraining: true, trainingId: jobId });
  }

  handleTrainDone() {
    console.log("Received train finished!");
    this.setState({ isTraining: false, trainingId: -1 });
    this.componentDidMount();
  }

  render() {
    if (typeof this.state.modelScores !== "undefined") {
      return (
        <div className="App">
          <Classifier onPredict={this.handlePredict.bind(this)} />
          <ModelData
            modelName={this.state.modelName}
            k={this.state.modelParams.K}
            trainingInstances={this.state.modelParams.train_nbr}
            testingInstances={this.state.modelParams.test_nbr}
            precisionY={this.state.modelScores.Young.precision}
            recallY={this.state.modelScores.Young.recall}
            precisionO={this.state.modelScores.Old.precision}
            recallO={this.state.modelScores.Old.recall}
            testScore={this.state.modelScores.test_score}
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
            onTrainDone={this.handleTrainDone.bind(this)}
          />
          <div className="credits">
            <img src="https://soe.lau.edu.lb/images/soe.png" />
            <h4>IEA Project Fall 2020</h4>
            <p>Peter Louis Sakr</p>
            <p>Melissa Chehade</p>
            <p>Samar Saleme</p>
          </div>
        </div>
      );
    } else {
      return <div></div>;
    }
  }
}

export default App;
