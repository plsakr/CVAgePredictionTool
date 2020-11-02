import React from "react";
import Card from "@material-ui/core/Card";
import "./ModelData.css";

function ModelData(props) {
  var modelName =
    props.modelName === "pretrained_knn_model"
      ? "KNN Classifier"
      : "User KNN Classifier";
  var k = props.k || 32;
  var trainingInstances = props.trainingInstances || 3600;
  var testingInstances = props.testingInstances || 0;
  var precisionYoung = props.precisionY || 1.0;
  var recallYoung = props.recallY || 1.0;
  var precisionOld = props.precisionO || 1.0;
  var recallOld = props.recallO || 1.0;
  var testScore = props.testScore;

  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Model Information</h1>
        <p className="info">
          <em>Classifier Type: </em> {modelName}
        </p>
        <p className="info">
          <em>Parameters: </em> K = {k}
        </p>
        <p className="info">
          <em>Training Instances: </em> {trainingInstances}
        </p>
        <p className="info">
          <em>Testing Instances: </em> {testingInstances}
        </p>
        <p className="info">
          <em>Accuracy: </em> {Math.round(testScore * 100)}%
        </p>
        <table>
          <thead>
            <tr>
              <th />
              <th>Precision</th>
              <th>Recall</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Young</td>
              <td>{Math.round(precisionYoung * 100)}%</td>
              <td>{Math.round(recallYoung * 100)}%</td>
            </tr>
            <tr>
              <td>Old</td>
              <td>{Math.round(precisionOld * 100)}%</td>
              <td>{Math.round(recallOld * 100)}%</td>
            </tr>
          </tbody>
        </table>
        {/* <em>Precision: </em> {precision} */}
        {/* <p className="info">
          <em>Recall: </em> {recall}
        </p> */}
      </div>
    </Card>
  );
}

export default ModelData;
