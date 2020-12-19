import React, { useState, useEffect } from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";
import "./TrainingCard.css";

function TrainingCard(props) {
  const [isReset, setReset] = useState(
    props.modelName == "pretrained_knn_model"
  );

  const handleReset = (e) => {
    console.log("Resetting model!");
    setReset(true);
    props.onReset(e);
  };

  useEffect(() => {
    setReset(props.modelName == "pretrained_knn_model");
  }, [props.modelName]);

  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Training</h1>
        <p className="longText">
          To train a different model, click the Train button below
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
          {/*<Button*/}
          {/*  disabled={isReset}*/}
          {/*  className="trainingButtons"*/}
          {/*  variant="contained"*/}
          {/*  color="secondary"*/}
          {/*  disableElevation*/}
          {/*  onClick={handleReset}*/}
          {/*>*/}
          {/*  Reset*/}
          {/*</Button>*/}
        </div>
      </div>
    </Card>
  );
}

export default TrainingCard;
