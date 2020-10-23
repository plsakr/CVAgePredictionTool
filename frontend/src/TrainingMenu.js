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
    this.state = { isOpen: false, tabValue: "0" };
  }

  // allow others to open me
  open() {
    this.setState({ isOpen: true });
  }

  // train button clicked. gather all data, tell the app what happened, and close the dialog
  handleOnTrain() {
    this.setState({ isOpen: false });
  }

  // close button clicked. close dialog and do nothing
  handleOnClose() {
    this.setState({ isOpen: false });
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
      currentState = <Params />;
    } else {
      currentState = <Dataset />;
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
