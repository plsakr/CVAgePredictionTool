import React from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";

function Classifier() {
  var image = "";

  const hiddenFileInput = React.useRef(null);
  const handleClick = (event) => {
    hiddenFileInput.current.click();
  };

  var imageOrButton;
  if (image == "") {
    imageOrButton = (
      <div>
        <Button
          variant="contained"
          color="primary"
          onClick={handleClick}
          disableElevation
        >
          Browse
        </Button>
        <input type="file" ref={hiddenFileInput} style={{ display: "none" }} />
      </div>
    );
  } else {
    imageOrButton = <img src={image} />;
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
