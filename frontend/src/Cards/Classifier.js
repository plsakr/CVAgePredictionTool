import React, { useState } from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";
import "./Classifier.css";

function Classifier() {
  const [image, setImage] = useState();

  const hiddenFileInput = React.useRef(null);
  const handleClick = (event) => {
    hiddenFileInput.current.click();
  };

  const handlePredict = (e) => {
    console.log("sending prediction request!");
    // TODO: send request by sending image, receive result immediately, show it somehow!
    setImage();
  };

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
      </div>
    </Card>
  );
}

export default Classifier;
