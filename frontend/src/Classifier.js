import React, { useState } from "react";
import Card from "@material-ui/core/Card";
import { Button } from "@material-ui/core";

function Classifier() {
  const [image, setImage] = useState();

  const hiddenFileInput = React.useRef(null);
  const handleClick = (event) => {
    hiddenFileInput.current.click();
  };

  var imageOrButton;
  if (typeof image == "undefined") {
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
      </div>
    );
  } else {
    imageOrButton = <img src={image} alt="" />;
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
