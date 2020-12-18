import React, { useState } from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";
import "./Classifier.css";
import { backend } from "../Config";
import {predict, subscribeToPrediction} from "../API/BackendCalls";

function Classifier(props) {
  const [image, setImage] = useState();
  const [prediction, setPrediction] = useState("");

  const hiddenFileInput = React.useRef(null);
  const handleClick = (event) => {
    hiddenFileInput.current.click();
  };

  const handlePredictionResult = (label) => {
      setPrediction(label);
  };

  subscribeToPrediction(handlePredictionResult);

  async function handlePredict(e) {
    console.log("sending prediction request!");
    console.log(image);

    let reader = new FileReader();
    let blob = await fetch(image).then((r) => r.blob());
    console.log(blob);
    reader.readAsDataURL(blob);
    reader.onload = function () {
      console.log(reader.result);
      URL.revokeObjectURL(image);
      setImage();

      const req = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: reader.result }),
      };

      const body = {image: reader.result}

      // predict(body)

      // console.log(`${backend}/predict`);
      fetch(`${backend}/predict`, req).then((postRes) => {
        // we now get the id, and fetch the result in 0.5s

        if (postRes.ok) {
          postRes.json().then((body) => {
            const jobId = body.jobId;

            let timer = setInterval(() => {
              fetch(`${backend}/jobinfo?jobId=${jobId}`).then((predRes) => {
                if (predRes.ok) {
                  predRes.json().then((bodyPred) => {
                    console.log(bodyPred);
                    if ("label" in bodyPred.jobResults) {
                      console.log("inside IF");
                      setPrediction(bodyPred.jobResults.label);
                      clearInterval(timer)
                    }
                  });
                }
              });
            }, 1000);
          });
        }
      });
    };
  }

  function browseButton(isPredictOn) {
    return (
      <div className="predictionButtons">
        <Button
          className="trainingButtons"
          variant="contained"
          color="primary"
          onClick={handleClick}
          disableElevation
        >
          Browse
        </Button>
        <input
          type="file"
          ref={hiddenFileInput}
          style={{ display: "none" }}
          accept="image/*"
          onChange={(e) => {
            console.log("GOT A FILE!");
            console.log(hiddenFileInput.current.files[0].name);
            if (typeof image != "undefined") URL.revokeObjectURL(image);
            setPrediction("");
            setImage(URL.createObjectURL(hiddenFileInput.current.files[0]));
          }}
        />
        <Button
          disabled={!isPredictOn}
          color={isPredictOn ? "secondary" : "default"}
          className="trainingButtons"
          variant="contained"
          disableElevation
          onClick={handlePredict}
        >
          Predict
        </Button>
      </div>
    );
  }

  var imageOrButton;
  if (typeof image == "undefined") {
    imageOrButton = browseButton(false);
  } else {
    imageOrButton = (
      <div className="cardBody">
        <img className="myImage" src={image} alt="" />
        {browseButton(true)}
      </div>
    );
  }

  return (
    <Card className="cardClass" variant="outlined">
      <div className="cardBody">
        <h1>Classification</h1>
        {imageOrButton}
        <div className="predictionResult">
          <h3 className="predictionText">{prediction}</h3>
        </div>
      </div>
    </Card>
  );
}

export default Classifier;
