import React, { Component } from "react";
import Modal from "react-bootstrap/esm/Modal";
import ProgressBar from "react-bootstrap/ProgressBar";
import { backend } from "./Config";

export default class TrainingLoad extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTraining: props.isTraining,
      jobId: props.jobId,
      progress: 0.0,
    };
    this.onFinish = props.onTrainDone;
  }

  componentDidUpdate() {
    if (this.state.jobId != -1) {
      fetch(`${backend}/jobinfo?jobId=${this.state.jobId}`).then((result) => {
        if (!result.ok) {
          console.log("The request did not complete successfully!");
        } else {
          result.json().then((body) => {
            const newProgress = body.jobProgress;
            if (newProgress == 1.0) {
              this.onFinish();
              this.setState({ isTraining: false, jobId: -1, progress: 0.0 });
            } else {
              const oldProgress = this.state.progress;
              if (oldProgress != newProgress) {
                this.setState({ progress: newProgress });
              } else {
                setTimeout(this.componentDidUpdate, 100);
              }
            }
          });
        }
      });
    }
  }

  render() {
    return (
      <Modal
        show={this.state.isTraining}
        backdrop="static"
        keyboard={false}
        centered
      >
        <Modal.Header>
          <Modal.Title>Training New Model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <h3>Training Progress</h3>
          <ProgressBar
            now={this.state.progress * 100}
            label={`${this.state.progress * 100}%`}
          />
        </Modal.Body>
      </Modal>
    );
  }
}
