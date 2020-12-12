import React from "react";
import {Dialog} from "@material-ui/core";
import {TransitionProps} from "@material-ui/core/transitions";
import Slide from "@material-ui/core/Slide";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";
import {ChooseState, ConfigureState, TrainingDialogProps, TrainingDialogState} from "../../API/MyTypes";
import Step from "@material-ui/core/Step";
import StepLabel from "@material-ui/core/StepLabel";
import Button from "@material-ui/core/Button";
import Stepper from "@material-ui/core/Stepper";
import ChooseModelComponent from "./ChooseModelComponent";

import './TrainingDialog.css';
import ConfigureModels from "./ConfigureModels";

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
    originalState: TrainingDialogState = {
        activeStep: 0,
        chooseModel: {
            modelType: 0,
            initialType: "KNN",
            trainInitial: true,
            trainYoung: false,
            trainOld: false,
            fullClassesTrain: false
        },
        configureModel: {
            chosen: null,
            optimizeK: true,
            minK: 0,
            maxK: 100,
            youngLayers: 5,
            oldLayers: 5,
            svmKernel: '',
            fullClassifierLayers: 5
        }
    }
    state: TrainingDialogState = this.originalState;

    stepOneHandler = (state: ChooseState) => {
        this.setState({chooseModel: state});
    }

    stepTwoHandler = (state: ConfigureState) => {
        this.setState({configureModel: state});
    }

    constructor(props: TrainingDialog) {
        super(props);
        this.isOpen = props.isOpen;
        this.onClose = props.onClose;
    }

    getStepContent(step: number) {
        switch (step) {
            case 0: return <ChooseModelComponent modelType={this.state.chooseModel.modelType}
                                                 initialType={this.state.chooseModel.initialType}
                                                 trainInitial={this.state.chooseModel.trainInitial}
                                                 fullClassification={this.state.chooseModel.fullClassesTrain}
                                                 trainOld={this.state.chooseModel.trainOld}
                                                 trainYoung={this.state.chooseModel.trainYoung}
                                                 onStateChange={this.stepOneHandler.bind(this)}/>
            case 1: return <ConfigureModels toConfigure={this.state.chooseModel}
                                            optimizeK={this.state.configureModel.optimizeK}
                                            minK={this.state.configureModel.minK}
                                            maxK={this.state.configureModel.maxK}
                                            svmKernel={this.state.configureModel.svmKernel}
                                            youngLayers={this.state.configureModel.youngLayers}
                                            oldLayers={this.state.configureModel.oldLayers}
                                            fullClassifierLayers={this.state.configureModel.fullClassifierLayers}
            onStateChange={this.stepTwoHandler.bind(this)}/>
            default: return 'what'
        }
    }

    handleClose = () => {
        this.setState(this.originalState)
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
                        <div className="stepperContent">
                            {this.getStepContent(this.state.activeStep)}
                            <div className="stepperButtons">
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
        </Dialog>);
    }
}

export default TrainingDialog;
