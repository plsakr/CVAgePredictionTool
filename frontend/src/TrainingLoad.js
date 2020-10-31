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
    this.timer = false;
  }

  componentDidMount() {
    if (this.state.isTraining && this.state.jobId !== -1) {
      this.timer = setInterval(this.tick.bind(this), 100);
    }
  }

  componentDidUpdate(prevProps, prevState) {
    console.log("TRAINING LOAD COMPONENT DID UPDATE!");
    if (prevProps.isTraining !== this.props.isTraining) {
      console.log(
        `Updating isTraining to ${this.props.isTraining} and jobId to ${this.props.jobId}`
      );
      this.setState({
        isTraining: this.props.isTraining,
        jobId: this.props.jobId,
      });
    }
    if (
      prevState.jobId !== this.state.jobId &&
      this.state.jobId != -1 &&
      this.state.jobId !== "undefined"
    ) {
      if (!this.timer) {
        this.timer = setInterval(this.tick.bind(this), 100);
      }
    }
  }

  tick() {
    fetch(`${backend}/jobinfo?jobId=${this.state.jobId}`).then((result) => {
      if (!result.ok) {
        console.log("The request did not complete successfully!");
      } else {
        result.json().then((body) => {
          const newProgress = body.jobProgress;
          if (newProgress == 1.0) {
            clearInterval(this.timer);
            this.timer = false;
            this.setState({ isTraining: false, jobId: -1, progress: 0.0 });
            this.onFinish();
          } else {
            const oldProgress = this.state.progress;
            if (oldProgress != newProgress) {
              this.setState({ progress: newProgress });
            }
          }
        });
      }
    });
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
            label={`${Math.round(this.state.progress * 100)}%`}
          />
        </Modal.Body>
      </Modal>
    );
  }
}
