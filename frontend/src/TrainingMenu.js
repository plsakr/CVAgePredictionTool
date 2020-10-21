import React from "react";
import Dialog from "@material-ui/core/Dialog";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import DialogTitle from "@material-ui/core/DialogTitle";
import Modal from "react-bootstrap/Modal";

import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";

import "./Dialog.css";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import ToggleButton from "react-bootstrap/ToggleButton";

class TrainingMenu extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isOpen: false, tabValue: "0" };
  }

  open() {
    this.setState({ isOpen: true });
  }

  handleOnTrain() {
    this.setState({ isOpen: false });
  }

  handleOnClose() {
    this.setState({ isOpen: false });
  }

  render() {
    const radios = [
      { name: "Custom Parameters", value: "0" },
      { name: "Custom Dataset", value: "1" },
    ];

    var currentState;

    if (this.state.tabValue == "0") {
      currentState = <p>You are changing the params!</p>;
    } else {
      currentState = <p>You are uploading custom images!</p>;
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
            Custom Modal Styling
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
