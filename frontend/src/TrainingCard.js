import React, { useState } from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";

function TrainingCard(props) {
  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Training</h1>
        <p className="longText">
          To train a different model, click the Train button below
        </p>
        <p className="longText">
          To reset the model to the default pre-trained model, click the reset
          button
        </p>
        <div className="trainingButtonsDiv">
          <Button
            className="trainingButtons"
            variant="contained"
            color="primary"
            onClick={props.onTrain}
            disableElevation
          >
            Train
          </Button>
          <Button
            className="trainingButtons"
            variant="contained"
            color="secondary"
            disableElevation
          >
            Reset
          </Button>
        </div>
      </div>
    </Card>
  );
}

export default TrainingCard;
