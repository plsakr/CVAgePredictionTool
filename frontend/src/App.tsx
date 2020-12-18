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
import TrainingDialog from "./Dialogs/Training/TrainingDialog";
import {backend} from "./Config";

const defaultModel = "pretrained_knn_model";


class App extends React.Component<{}, AppState> {
  state: AppState = {
      cnnsData: undefined,
      ensembleData: {
      modelName: "",
      modelScores: {Old: {precision:0, recall:0, support:0},
                    Young: {precision:0, recall:0, support:0},
                    acc: 0,},
      modelParams: {K:0, train_nbr:0, test_nbr:0},
    },
    isTraining: false,
    trainingId: -1,
    isScoresOpen: false,
    isTrainingOpen: false
  };

  trainingMenu: RefObject<TrainingMenu> = React.createRef();

  // recieveInfo(data: ModelInfo) {
  //   console.log(data.young_nn_score)
  //   this.setState({
  //     ensembleData: {
  //       modelName: data.model_name,
  //       modelScores: {
  //         Old: { precision: data.ensemble_score.KNNScore["1"].precision,
  //         recall: data.ensemble_score.KNNScore["1"].recall, support: data.ensemble_score.KNNScore["1"].support},
  //         Young: { precision: data.ensemble_score.KNNScore["0"].precision, recall: data.ensemble_score.KNNScore["0"].recall, support: data.ensemble_score.KNNScore["0"].support},
  //         acc: data.ensemble_score.KNNScore.accuracy,
  //         // test_score: data.model_scores.test_score
  //       },
  //       modelParams: {
  //         K: data.ensemble_score.K,
  //         test_nbr: 0,
  //         train_nbr: 0
  //       }
  //     },
  //       cnnsData: {
  //         young: data.young_nn_score,
  //           old: data.old_nn_score
  //       }
  //   })
  // }

  componentDidMount() {
      fetch(`${backend}/info`).then((result) => {
          if (result.ok) {
              result.json().then((body) => {
                  console.log(body);
                  this.setState({
                      ensembleData: {
                          modelName: body.model_name,
                          modelScores: {
                              Old: { precision: body.ensemble_score.KNNScore["1"].precision,
                                  recall: body.ensemble_score.KNNScore["1"].recall, support: body.ensemble_score.KNNScore["1"].support},
                              Young: { precision: body.ensemble_score.KNNScore["0"].precision, recall: body.ensemble_score.KNNScore["0"].recall, support: body.ensemble_score.KNNScore["0"].support},
                              acc: body.ensemble_score.KNNScore.accuracy,
                              // test_score: data.model_scores.test_score
                          },
                          modelParams: {
                              K: body.ensemble_score.K,
                              test_nbr: 0,
                              train_nbr: 0
                          }
                      },
                      cnnsData: {
                          young: body.young_nn_score,
                          old: body.old_nn_score
                      }
                  })
              });
          }
      });
  }

  handlePredict(base64: any) {
      console.log('predicting', base64)
  }

  handleScoreButton() {
    console.log("score button clicked")
    this.setState({isScoresOpen: true})
  }

  handleScoreClose() {
    this.setState({isScoresOpen: false})
  }

  handleTrainClose() {
    this.setState({isTrainingOpen: false})
  }

  handleTrainButton(e: any) {
    console.log("TRAINING CLICKED");
    this.setState({isTrainingOpen: true})
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
    if (this.state.ensembleData.modelScores.Old.support !== 0 && this.state.cnnsData !== undefined) {
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
          <ScoresDialog isOpen={this.state.isScoresOpen} onClose={this.handleScoreClose.bind(this)} ensembleScores={this.state.ensembleData} neuralNetworkScores={{young: this.state.cnnsData.young, old: this.state.cnnsData.old}}/>
          <TrainingDialog isOpen={this.state.isTrainingOpen} onClose={this.handleTrainClose.bind(this)} onTrain={this.handleTrainNewModel.bind(this)}/>
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
