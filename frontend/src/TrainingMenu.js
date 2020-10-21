import React from "react";
import Dialog from "@material-ui/core/Dialog";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import DialogTitle from "@material-ui/core/DialogTitle";

import "./Dialog.css";
import { Button } from "@material-ui/core";

class TrainingMenu extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isOpen: false };
  }

  handleTrainingClick() {
    this.setState({ isOpen: true });
  }

  handleOnClose() {
    this.setState({ isOpen: false });
  }

  render() {
    return (
      <Dialog open={this.state.isOpen} onClose={this.handleOnClose.bind(this)}>
        <DialogTitle>Train Model</DialogTitle>
        <DialogContent>
          <p>
            Lorem ipsum dolor sit, amet consectetur adipisicing elit. Iusto,
            alias quam repellat vero dignissimos modi aperiam porro quasi
            suscipit voluptate reiciendis, cupiditate sed placeat quis quidem?
            Perferendis ea temporibus corrupti?
          </p>
        </DialogContent>
        <DialogActions>
          <Button color="primary" onClick={this.handleOnClose.bind(this)}>
            Train
          </Button>
          <Button onClick={this.handleOnClose.bind(this)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    );
  }
}

export default TrainingMenu;
