import React from "react";
import Modal from "react-bootstrap/Modal";

import "./Dialog.css";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import ToggleButton from "react-bootstrap/ToggleButton";

import Params from "./Training/Params";
import Dataset from "./Training/Dataset";
import { backend } from "./Config";

class TrainingMenu extends React.Component {
  constructor(props) {
    super(props);
    this.onTrain = props.onTrain;
    this.state = {
      isOpen: false,
      tabValue: "0",
      minK: 1,
      maxK: 100,
      oldPicsNbr: 10,
      youngPicsNbr: 10,
      testingRatio: 0.2,
      isChecked: false,
      youngUrls: [],
      oldUrls: [],
    };
  }

  // allow others to open me
  open() {
    this.setState({ isOpen: true });
  }

  resolveURLs(urls) {
    return new Promise((resolve, reject) => {
      console.log(urls);
      const count = urls.length;
      var result = [];
      urls.forEach((url) => {
        let reader = new FileReader();
        let blob = fetch(url).then((r) =>
          r.blob().then((blob) => {
            reader.readAsDataURL(blob);
            reader.onload = function () {
              result.push(reader.result);
              if (result.length == count) {
                resolve(result);
              }
            };
          })
        );
      });
    });
  }

  performRequest(req) {
    if (req !== "undefined") {
      fetch(`${backend}/train`, req).then((res) => {
        if (!res.ok) {
          console.log("There was a problem with the train request!");
        } else {
          res.json().then((body) => {
            const jobId = body.jobId;

            console.log(`MY TRAINING ID IS ${jobId}`);
            this.setState({ isOpen: false });
            this.onTrain(jobId);
          });
        }
      });
    }
  }

  // train button clicked. gather all data, tell the app what happened, and close the dialog
  async handleOnTrain() {
    console.log("dialog train button pressed");

    if (this.state.tabValue === "0") {
      const minKVal = this.state.minK;
      const maxKVal = this.state.maxK;
      const isChecked = this.state.isChecked;
      const oldPicsNbr = this.state.oldPicsNbr;
      const youngPicsNbr = this.state.youngPicsNbr;
      const testingRatio = this.state.testingRatio;

      if (isChecked) {
        if (maxKVal > minKVal) {
          const req = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              isReset: false,
              isCustom: false,
              optimizeK: true,
              minK: minKVal,
              maxK: maxKVal,
              nbrYoung: youngPicsNbr,
              nbrOld: oldPicsNbr,
              testRatio: testingRatio,
            }),
          };

          this.performRequest(req);
        } else {
          console.log("maxK cannot be larger than minK");
        }
      } else {
        const req = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            isReset: false,
            isCustom: false,
            optimizeK: false,
            minK: minKVal,
            nbrYoung: youngPicsNbr,
            nbrOld: oldPicsNbr,
            testRatio: testingRatio,
          }),
        };

        this.performRequest(req);
      }
    } else {
      const minKVal = this.state.minK;
      const maxKVal = this.state.maxK;
      const isChecked = this.state.isChecked;
      const testingRatio = this.state.testingRatio;
      const youngUrls = this.state.youngUrls;
      const oldUrls = this.state.oldUrls;

      if (isChecked) {
        if (maxKVal > minKVal) {
          this.resolveURLs(youngUrls).then((youngData) => {
            this.resolveURLs(oldUrls).then((oldData) => {
              const body = {
                isReset: false,
                isCustom: true,
                optimizeK: true,
                minK: minKVal,
                maxK: maxKVal,
                youngPics: youngData,
                oldPics: oldData,
                testRatio: testingRatio,
              };
              const req = {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              };
              this.performRequest(req);
            });
          });
        }
      } else {
        this.resolveURLs(youngUrls).then((youngData) => {
          this.resolveURLs(oldUrls).then((oldData) => {
            const req = {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                isReset: false,
                isCustom: true,
                optimizeK: false,
                minK: minKVal,
                youngPics: youngData,
                oldPics: oldData,
                testRatio: testingRatio,
              }),
            };

            this.performRequest(req);
          });
        });
      }
    }
  }

  // close button clicked. close dialog and do nothing
  handleOnClose() {
    this.setState({
      isOpen: false,
      isChecked: false,
      tabValue: "0",
      minK: 0,
      maxK: 100,
      oldPicsNbr: 1,
      youngPicsNbr: 1,
      testingRatio: 0.2,
      oldUrls: [],
      youngUrls: [],
    });
  }

  handleFormChange(e, isCheckbox = false) {
    const name = isCheckbox ? "isChecked" : e.target.name;
    const value = isCheckbox ? e.target.checked : Number(e.target.value);
    console.log(name);
    console.log(value);
    this.setState({
      [name]: value,
    });
  }

  handleYoungUpload(urls) {
    this.setState((state, props) => {
      return {
        youngUrls: [...state.youngUrls, ...urls],
      };
    });
  }

  handleOldUpload(urls) {
    this.setState((state, props) => {
      return {
        oldUrls: [...state.oldUrls, ...urls],
      };
    });
  }

  render() {
    // the tabs
    const radios = [
      { name: "Custom Parameters", value: "0" },
      { name: "Custom Dataset", value: "1" },
    ];

    // the tab content
    var currentState;

    if (this.state.tabValue == "0") {
      currentState = (
        <Params
          handleChange={this.handleFormChange.bind(this)}
          k={this.state.minK}
          maxK={this.state.maxK}
          oldPics={this.state.oldPicsNbr}
          youngPics={this.state.youngPicsNbr}
          testRatio={this.state.testingRatio}
          isChecked={this.state.isChecked}
        />
      );
    } else {
      currentState = (
        <Dataset
          handleChange={this.handleFormChange.bind(this)}
          k={this.state.minK}
          maxK={this.state.maxK}
          testRatio={this.state.testingRatio}
          isChecked={this.state.isChecked}
          handleYoungUpload={this.handleYoungUpload.bind(this)}
          handleOldUpload={this.handleOldUpload.bind(this)}
        />
      );
    }

    return (
      <Modal
        show={this.state.isOpen}
        onHide={() => this.setState({ isOpen: false })}
        dialogClassName="modal-90w"
        aria-labelledby="example-custom-modal-styling-title"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title id="example-custom-modal-styling-title">
            Train New Model
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ButtonGroup toggle>
            {radios.map((radio, idx) => (
              <ToggleButton
                key={idx}
                type="radio"
                variant="primary"
                name="radio"
                value={radio.value}
                checked={this.state.tabValue == radio.value}
                onChange={(e) =>
                  this.setState({
                    tabValue: radio.value,
                    oldUrls: [],
                    youngUrls: [],
                  })
                }
              >
                {radio.name}
              </ToggleButton>
            ))}
          </ButtonGroup>
          {currentState}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="success" onClick={this.handleOnTrain.bind(this)}>
            Train
          </Button>
          <Button
            variant="outline-secondary"
            onClick={this.handleOnClose.bind(this)}
          >
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default TrainingMenu;
