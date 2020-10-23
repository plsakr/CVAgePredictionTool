import { Typography } from "@material-ui/core";
import React, { Component } from "react";
import DnDImgUploader from "../HelperComponents/DnDImgUploader";
import Slider from "@material-ui/core/Slider";
import TextField from "@material-ui/core/TextField";
import "./Dataset.css";

export default class Dataset extends Component {
  constructor(props) {
    super(props);

    this.state = {};
  }

  render() {
    return (
      <div>
        <h4 className="header">Create Custom Dataset:</h4>
        <div className="datasetContainer">
          <div className="imageUploader">
            <Typography>Young pics</Typography>
            <DnDImgUploader />
          </div>
          <div className="imageUploader">
            <Typography>Old pics</Typography>
            <DnDImgUploader />
          </div>
        </div>
        <div className="datasetContainer">
          <TextField variant="outlined" label="# Of Neighbors (K)" />
          <div className="bottomSlider">
            <Typography>Percentage of Training Instances</Typography>
            <Slider
              min={0}
              max={0.9}
              valueLabelDisplay="auto"
              defaultValue={0.2}
              step={0.05}
            />
          </div>
        </div>
      </div>
    );
  }
}
