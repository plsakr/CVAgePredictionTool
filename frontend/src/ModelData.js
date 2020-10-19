import React from "react";
import Card from "@material-ui/core/Card";

function ModelData(props) {
  var k = props.k || 32;
  var trainingInstances = props.trainingInstances || 3600;
  var precision = props.precision || 1.0;
  var recall = props.recall || 1.0;

  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Model Information</h1>
        <p className="info">
          <em>Classifier Type: </em> KNN Classifier
        </p>
        <p className="info">
          <em>Parameters: </em> K = {k}
        </p>
        <p className="info">
          <em>Training Instances: </em> {trainingInstances}
        </p>
        <p className="info">
          <em>Precision: </em> {precision}
        </p>
        <p className="info">
          <em>Recall: </em> {recall}
        </p>
      </div>
    </Card>
  );
}

export default ModelData;
