import React, { Component } from "react";
import TextField from "@material-ui/core/TextField";
import Typography from "@material-ui/core/Typography";
import Slider from "@material-ui/core/Slider";

import "./Params.css";

export default class Params extends Component {
  constructor(props) {
    super(props);
    this.k = props.k | 0;
    this.oldPicsTrain = props.oldPics | 0;
    this.youngPicsTrain = props.oldPics | 0;
    this.testRatio = props.testRatio | 0.2;
  }

  render() {
    return (
      <div>
        <h4 className="header">Set Training Parameters:</h4>
        <div className="myForm">
          <div className="myFormComponent">
            <TextField variant="outlined" label="# Of Neighbors (K)" />
          </div>

          <div className="myFormComponent">
            <Typography># Of Old Pictures</Typography>
            <Slider
              min={1}
              max={36000}
              valueLabelDisplay="auto"
              defaultValue={1}
            />
            <Typography># Of Young Pictures</Typography>
            <Slider
              min={1}
              max={36000}
              valueLabelDisplay="auto"
              defaultValue={1}
            />
          </div>
        </div>

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
    );
  }
}
