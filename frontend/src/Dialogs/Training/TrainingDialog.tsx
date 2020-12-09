import React from "react";
import {Dialog} from "@material-ui/core";
import {TransitionProps} from "@material-ui/core/transitions";
import Slide from "@material-ui/core/Slide";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";
import {TrainingDialogProps, TrainingDialogState} from "../../API/MyTypes";
import Step from "@material-ui/core/Step";
import StepLabel from "@material-ui/core/StepLabel";
import Button from "@material-ui/core/Button";
import Stepper from "@material-ui/core/Stepper";


const Transition = React.forwardRef(function Transition(
    props: TransitionProps & { children?: React.ReactElement },
    ref: React.Ref<unknown>,
) {
    return <Slide direction="up" ref={ref} {...props} />;
});


class TrainingDialog extends React.Component<TrainingDialogProps, TrainingDialogState> {
    isOpen: boolean;
    onClose: () => void;
    steps = ['Choose Model', 'Configure Models', 'Choose/Import Data']
    state: TrainingDialogState = {
        activeStep: 0
    }

    constructor(props: TrainingDialog) {
        super(props);
        this.isOpen = props.isOpen;
        this.onClose = props.onClose;
    }

    getStepContent(step: number) {
        switch (step) {
            default: return 'what'
        }
    }

    handleClose = () => {
        this.setState({activeStep: 0})
        this.onClose()
    }

    handleBack = () => {
        const prev = this.state.activeStep;
        this.setState( {activeStep: prev-1})
    }

    handleNext = () => {
        const prev = this.state.activeStep;
        this.setState( {activeStep: prev+1})
    }

    handleTrain = () => {

    }

    shouldComponentUpdate(nextProps: Readonly<TrainingDialogProps>, nextState: Readonly<{}>, nextContext: any): boolean {
        this.isOpen = nextProps.isOpen;
        return true;
    }

    render() {
        return (<Dialog fullScreen open={this.isOpen} TransitionComponent={Transition}>
            <AppBar className="appBar">
                <Toolbar>
                    <IconButton edge="start" color="inherit" onClick={this.handleClose.bind(this)}>
                        <CloseIcon />
                    </IconButton>
                    <Typography variant="h6" className="title">Train Your Model</Typography>
                </Toolbar>
            </AppBar>
            <div className="dialogContent">
            <Stepper activeStep={this.state.activeStep}>
                {this.steps.map((label) => {
                    const stepProps: {completed?: boolean} = {};
                    const labelProps: {optional?: React.ReactNode} = {};

                    return (
                        <Step key={label} {...stepProps}>
                            <StepLabel {...labelProps}>{label}</StepLabel>
                        </Step>
                    )
                })}
            </Stepper>
                <div>
                    {this.state.activeStep === this.steps.length ? (
                        <div>
                            <Typography>
                                Configuration completed - Press button to begin training!
                            </Typography>
                            <Button onClick={this.handleTrain.bind(this)}>
                                Finish
                            </Button>
                        </div>
                    ) : (
                        <div>
                            <Typography>
                                {this.getStepContent(this.state.activeStep)}
                            </Typography>
                            <div>
                                <Button disabled={this.state.activeStep === 0} onClick={this.handleBack.bind(this)}>
                                    Back
                                </Button>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={this.handleNext.bind(this)}>
                                    {this.state.activeStep === this.steps.length - 1 ? 'Finish' : 'Next'}
                                </Button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </Dialog>);
    }
}

export default TrainingDialog;
