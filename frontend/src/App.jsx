"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var React = require("react");
require("./App.css");
var Classifier_1 = require("./Cards/Classifier");
var ModelData_1 = require("./Cards/ModelData");
var TrainingCard_1 = require("./Cards/TrainingCard");
var TrainingMenu_1 = require("./TrainingMenu");
require("bootstrap/dist/css/bootstrap.min.css");
require("bootstrap/dist/css/bootstrap.css"); // or include from a CDN
require("react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css");
var TrainingLoad_1 = require("./TrainingLoad");
var BackendCalls_1 = require("./API/BackendCalls");
var defaultModel = "pretrained_knn_model";
var App = /** @class */ (function (_super) {
    __extends(App, _super);
    function App() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.state = {
            modelName: "",
            modelScores: undefined,
            modelParams: { K: 0, trainInstances: 0, testInstances: 0 },
            isTraining: false,
            trainingId: -1,
        };
        _this.trainingMenu = React.createRef();
        return _this;
    }
    App.prototype.recieveInfo = function (data) {
        console.log(data);
    };
    App.prototype.componentDidMount = function () {
        BackendCalls_1.subscribeToInfo(this.recieveInfo);
        // fetch(`${backend}/info`).then((result) => {
        //   if (result.ok) {
        //     result.json().then((body) => {
        //       console.log(body);
        //       this.setState({
        //         modelName: body.model_name,
        //         modelScores: body.model_scores,
        //         modelParams: body.model_params,
        //         isTraining: body.isTraining,
        //         trainingId: body.trainingId,
        //       });
        //     });
        //   }
        // });
    };
    App.prototype.handlePredict = function (base64) { };
    App.prototype.handleTrainButton = function (e) {
        console.log("TRAINING CLICKED");
        this.trainingMenu.current.open();
    };
    App.prototype.handleResetButton = function (e) {
        var _this = this;
        console.log("HANDLING MODEL RESET!");
        var req = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                isReset: true,
            }),
        };
        fetch(backend + "/train", req).then(function (res) {
            if (!res.ok) {
                console.log("Could not reset model!");
            }
            else {
                res.json().then(function (body) {
                    var jobId = body.jobId;
                    setTimeout(function () {
                        fetch(backend + "/jobinfo?jobId=" + jobId).then(function (resetRes) {
                            if (resetRes.ok) {
                                _this.componentDidMount();
                            }
                        });
                    }, 200);
                });
            }
        });
    };
    App.prototype.handleTrainNewModel = function (jobId) {
        console.log("HANDLING NEW MODEL CREATION! JOBID " + jobId);
        this.setState({ isTraining: true, trainingId: jobId });
    };
    App.prototype.handleTrainDone = function () {
        console.log("Received train finished!");
        this.setState({ isTraining: false, trainingId: -1 });
        this.componentDidMount();
    };
    App.prototype.render = function () {
        if (typeof this.state.modelScores !== "undefined") {
            return (<div className="App">
          <Classifier_1.default onPredict={this.handlePredict.bind(this)}/>
          <ModelData_1.default modelName={this.state.modelName} k={this.state.modelParams.K} trainingInstances={this.state.modelParams.trainInstances} testingInstances={this.state.modelParams.testInstances} precisionY={this.state.modelScores.youngPrecision} recallY={this.state.modelScores.youngRecall} precisionO={this.state.modelScores.oldPrecision} recallO={this.state.modelScores.oldRecall} testScore={this.state.modelScores.ensembleACC}/>
          <TrainingCard_1.default onTrain={this.handleTrainButton.bind(this)} onReset={this.handleResetButton.bind(this)} modelName={this.state.modelName}/>
          <TrainingMenu_1.default ref={this.trainingMenu} onTrain={this.handleTrainNewModel.bind(this)}/>
          <TrainingLoad_1.default isTraining={this.state.isTraining} jobId={this.state.trainingId} onTrainDone={this.handleTrainDone.bind(this)}/>
          <div className="credits">
            <img src="https://soe.lau.edu.lb/images/soe.png"/>
            <h4>IEA Project Fall 2020</h4>
            <p>Peter Louis Sakr</p>
            <p>Melissa Chehade</p>
            <p>Samar Salame</p>
          </div>
        </div>);
        }
        else {
            return <div></div>;
        }
    };
    return App;
}(React.Component));
exports.default = App;
