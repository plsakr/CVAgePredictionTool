import React from "react";
import Modal from "react-bootstrap/Modal";

import "./Dialog.css";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import ToggleButton from "react-bootstrap/ToggleButton";

import Params from "./Training/Params";
import Dataset from "./Training/Dataset";

class TrainingMenu extends React.Component {
  constructor(props) {
    super(props);
    this.onTrain = props.onTrain;
    this.state = {
      isOpen: false,
      tabValue: "0",
      minK: 1,
      maxK: 100,
      OldPicsNbr: 1,
      YoungPicsNbr: 1,
      testingRatio: 0.2,
      isChecked: false,
    };
  }

  // allow others to open me
  open() {
    this.setState({ isOpen: true });
  }

  // train button clicked. gather all data, tell the app what happened, and close the dialog
  handleOnTrain() {
    console.log("dialog train button pressed");

    if (this.state.tabValue === "0") {
      const minK = this.state.minK;
      const maxK = this.state.maxK;
      const isChecked = this.state.isChecked;
      const oldPicsNbr = this.state.oldPicsNbr;
      const youngPicsNbr = this.state.youngPicsNbr;
      const testingRatio = this.state.testingRatio;

      if (isChecked){
        if (maxK > minK) {
          const req  

        } else {
          console.log("maxK cannot be larger than minK");
        }
      }
      
    } else {
      // TODO
    }

    this.setState({ isOpen: false });
    this.onTrain(); // this method should take all the needed data!
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
          oldPics={this.state.OldPicsNbr}
          youngPics={this.state.YoungPicsNbr}
          testRatio={this.state.testingRatio}
        />
      );
    } else {
      currentState = (
        <Dataset handleChange={this.handleFormChange.bind(this)} />
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
                onChange={(e) => this.setState({ tabValue: radio.value })}
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
