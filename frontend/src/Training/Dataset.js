import { Typography } from "@material-ui/core";
import React, { Component } from "react";
import DnDImgUploader from "../HelperComponents/DnDImgUploader";
import RangeSlider from "react-bootstrap-range-slider";
import Checkbox from "@material-ui/core/Checkbox";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import "./Dataset.css";

export default class Dataset extends Component {
  constructor(props) {
    super(props);
    this.isChecked = props.isChecked;
    this.k = props.k;
    this.maxK = props.maxK;
    this.testRatio = props.testRatio;
    this.propshandleChange = props.handleChange;
    this.handleYoungUpload = props.handleYoungUpload;
    this.handleOldUpload = props.handleOldUpload;
    this.state = {
      isChecked: this.isChecked,
      minK: this.k,
      maxK: this.maxK,
      testingRatio: this.testRatio,
      youngPics: [],
      oldPics: [],
    };
  }

  handleCheckbox = (e) => {
    this.setState({ isChecked: e.target.checked });
    this.propshandleChange(e, true);
  };

  onReceiveURLsYoung(urls) {
    this.setState((state, props) => {
      return {
        youngPics: [...state.youngPics, ...urls],
      };
    });
    this.handleYoungUpload(urls);
  }

  onReceiveURLsOld(urls) {
    this.setState((state, props) => {
      return {
        oldPics: [...state.oldPics, ...urls],
      };
    });
    this.handleOldUpload(urls);
  }

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
        <h4 className="header">Create Custom Dataset:</h4>
        <div className="datasetContainer">
          <div className="imageUploader">
            <Typography>Young pics</Typography>
            <DnDImgUploader onDropURLs={this.onReceiveURLsYoung.bind(this)} />
          </div>
          <div className="imageUploader">
            <Typography>Old pics</Typography>
            <DnDImgUploader onDropURLs={this.onReceiveURLsOld.bind(this)} />
          </div>
        </div>
        <div className="datasetContainer">
          <div className="myFormComponent">
            <FormControlLabel
              label="Optimize K"
              control={
                <Checkbox
                  checked={this.state.isChecked}
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
      </div>
    );
  }
}
