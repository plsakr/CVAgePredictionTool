import * as React from "react";
import "./App.css";
import Classifier from "./Cards/Classifier";
import ModelData from "./Cards/ModelData";
import TrainingCard from "./Cards/TrainingCard";
import TrainingMenu from "./Dialogs/TrainingMenu";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/css/bootstrap.css"; // or include from a CDN
import "react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css";
import TrainingLoad from "./Dialogs/TrainingLoad";
import {type} from "os";
import {AppState, ModelInfo, PrecisionRecall} from "./API/MyTypes";
import {RefObject} from "react";
import {subscribeToInfo} from "./API/BackendCalls";
import ScoresDialog from "./Dialogs/ScoresDialog";

const defaultModel = "pretrained_knn_model";


class App extends React.Component<{}, AppState> {
  state: AppState = {
    ensembleData: {
      modelName: "",
      modelScores: {Old: {precision:0, recall:0, support:0},
                    Young: {precision:0, recall:0, support:0},
                    acc: 0,
                    test_score: 0},
      modelParams: {K:0, train_nbr:0, test_nbr:0},
    },
    isTraining: false,
    trainingId: -1,
    isScoresOpen: false
  };

  trainingMenu: RefObject<TrainingMenu> = React.createRef();

  recieveInfo(data: ModelInfo) {
    console.log(data)
    this.setState({
      isTraining: data.isTraining,
      ensembleData: {
        modelName: data.model_name,
        modelScores: {
          Old: { precision: data.model_scores.Old.precision,
          recall: data.model_scores.Old.recall, support: data.model_scores.Old.support},
          Young: { precision: data.model_scores.Young.precision, recall: data.model_scores.Young.recall, support: data.model_scores.Young.support},
          acc: data.model_scores.acc,
          test_score: data.model_scores.test_score
        },
        modelParams: {
          K: data.model_params.K,
          test_nbr: data.model_params.test_nbr,
          train_nbr: data.model_params.train_nbr
        }
      },
    })
  }

  componentDidMount() {
    console.log('mounted')
    subscribeToInfo(this.recieveInfo.bind(this));
  }

  handlePredict(base64: any) {}

  handleScoreButton() {
    console.log("score button clicked")
    this.setState({isScoresOpen: true})
  }

  handleScoreClose() {
    this.setState({isScoresOpen: false})
  }

  handleTrainButton(e: any) {
    console.log("TRAINING CLICKED");
    if (this.trainingMenu.current != null)
      this.trainingMenu.current.open();
  }

  handleResetButton(e: any) {
    // console.log("HANDLING MODEL RESET!");
    // var req = {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify({
    //     isReset: true,
    //   }),
    // };
    // fetch(`${backend}/train`, req).then((res) => {
    //   if (!res.ok) {
    //     console.log("Could not reset model!");
    //   } else {
    //     res.json().then((body) => {
    //       const jobId = body.jobId;
    //
    //       setTimeout(() => {
    //         fetch(`${backend}/jobinfo?jobId=${jobId}`).then((resetRes) => {
    //           if (resetRes.ok) {
    //             this.componentDidMount();
    //           }
    //         });
    //       }, 200);
    //     });
    //   }
    // });
  }

  handleTrainNewModel(jobId: number) {
    console.log(`HANDLING NEW MODEL CREATION! JOBID ${jobId}`);
    this.setState({ isTraining: true, trainingId: jobId });
  }

  handleTrainDone() {
    console.log("Received train finished!");
    this.setState({ isTraining: false, trainingId: -1 });
    this.componentDidMount();
  }

  render() {
    if (this.state.ensembleData.modelScores.Old.support !== 0) {
      return (
        <div className="App">
          <Classifier onPredict={this.handlePredict.bind(this)} />
          <ModelData
            onClickScore={this.handleScoreButton.bind(this)}
          />
          <TrainingCard
            onTrain={this.handleTrainButton.bind(this)}
            onReset={this.handleResetButton.bind(this)}
            modelName={this.state.ensembleData.modelName}
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
          <ScoresDialog isOpen={this.state.isScoresOpen} onClose={this.handleScoreClose.bind(this)} ensembleScores={this.state.ensembleData}/>
          <div className="credits">
            <img src="https://soe.lau.edu.lb/images/soe.png" />
            <h4>IEA Project Fall 2020</h4>
            <p>Peter Louis Sakr</p>
            <p>Melissa Chehade</p>
            <p>Samar Salame</p>
          </div>
        </div>
      );
    } else {
      return <div></div>;
    }
  }
}

export default App;
