import React from "react";
import Card from "@material-ui/core/Card";
import "./ModelData.css";
import Button from "@material-ui/core/Button";
import ScoreIcon from "@material-ui/icons/Score"

function ModelData(props) {


  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Model Information</h1>
        <p>The model is separated into an ensemble model made of a KNN and an SVM classifier, in addition to two separate
        convolutional neural network classifiers. <br />Click the button below to view the model scores.</p>
        <div className="buttonDiv">
          <Button color="primary" variant="contained" startIcon={<ScoreIcon />} onClick={props.onClickScore}>Model Scores</Button>
        </div>
      </div>
    </Card>
  );
}

export default ModelData;
