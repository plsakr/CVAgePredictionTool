import React, { Component } from "react";
import Checkbox from "@material-ui/core/Checkbox";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Typography from "@material-ui/core/Typography";
import RangeSlider from "react-bootstrap-range-slider";

import "./Params.css";

export default class Params extends Component {
  constructor(props) {
    super(props);
    this.k = props.k;
    this.maxK = props.maxK;
    this.oldPicsTrain = props.oldPics;
    this.youngPicsTrain = props.oldPics;
    this.testRatio = props.testRatio;
    this.propshandleChange = props.handleChange;
    this.state = {
      isChecked: false,
      minK: 1,
      maxK: 100,
      oldPicsNbr: 10,
      youngPicsNbr: 10,
      testingRatio: 0.2,
    };
  }

  handleCheckbox = (e) => {
    this.setState({ isChecked: e.target.checked });
    this.propshandleChange(e, true);
  };

  handleChange = (e) => {
    const name = e.target.name;
    const value = e.target.value;
    console.log(name);
    console.log(value);
    this.setState({ [name]: Number(value) });
    this.propshandleChange(e);
  };

  render() {
    return (
      <div>
        <h4 className="header">Set Training Parameters:</h4>
        <div className="myForm">
          <div className="myFormComponent">
            <FormControlLabel
              label="Optimize K"
              control={
                <Checkbox
                  onChange={this.handleCheckbox.bind(this)}
                  color="primary"
                />
              }
            />
            <Typography>{this.state.isChecked ? "Minimum K" : "K"}</Typography>
            <RangeSlider
              onChange={this.handleChange}
              value={this.state.minK}
              min={1}
              max={100}
              inputProps={{ name: "minK" }}
            />
            <Typography>Maximum K</Typography>
            <RangeSlider
              onChange={this.handleChange}
              value={this.state.maxK}
              min={1}
              max={100}
              inputProps={{ name: "maxK" }}
              disabled={!this.state.isChecked}
            />
          </div>

          <div className="myFormComponent">
            <Typography># Of Old Pictures</Typography>
            <RangeSlider
              inputProps={{ name: "oldPicsNbr" }}
              onChange={this.handleChange}
              value={this.state.oldPicsNbr}
              min={1}
              max={36000}
            />
            <Typography># Of Young Pictures</Typography>
            <RangeSlider
              inputProps={{ name: "youngPicsNbr" }}
              value={this.state.youngPicsNbr}
              onChange={this.handleChange}
              min={1}
              max={36000}
            />
          </div>
        </div>

        <div className="bottomSlider">
          <Typography>Percentage of Testing Instances</Typography>
          <RangeSlider
            inputProps={{ name: "testingRatio" }}
            onChange={this.handleChange}
            value={this.state.testingRatio}
            min={0}
            max={0.9}
            step={0.05}
          />
        </div>
      </div>
    );
  }
}
