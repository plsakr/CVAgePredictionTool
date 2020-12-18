import React from "react";
import {Dialog} from "@material-ui/core";
import {TransitionProps} from "@material-ui/core/transitions";
import Slide from "@material-ui/core/Slide";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";
import {ChooseState, ConfigureState, DatasetState, TrainingDialogProps, TrainingDialogState} from "../../API/MyTypes";
import Step from "@material-ui/core/Step";
import StepLabel from "@material-ui/core/StepLabel";
import Button from "@material-ui/core/Button";
import Stepper from "@material-ui/core/Stepper";
import ChooseModelComponent from "./ChooseModelComponent";

import './TrainingDialog.css';
import ConfigureModels from "./ConfigureModels";
import ConfigureDatasetComponent from "./ConfigureDatasetComponent";
import {backend} from "../../Config";

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
            initialType: "Pretrained",
            trainInitial: false,
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
        },
        configureData: {
            imageURLs: [[],[],[],[]],
            tabValue: 0,
            custom: false,
            nbrClasses: 4,
            testingRatio: 0.2
        }
    }
    state: TrainingDialogState = this.originalState;
    onTrain: (arg0: number) => void;

    stepOneHandler = (state: ChooseState) => {
        this.setState({chooseModel: state});
    }

    stepTwoHandler = (state: ConfigureState) => {
        this.setState({configureModel: state});
    }

    stepThreeHandler = (state: DatasetState) => {
        this.setState({configureData: state});
    }

    constructor(props: TrainingDialog) {
        super(props);
        this.isOpen = props.isOpen;
        this.onClose = props.onClose;
        this.onTrain = props.onTrain;
    }

    componentWillUnmount() {

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
            case 2: return <ConfigureDatasetComponent custom={this.state.configureData.custom}
                                                      testingRatio={this.state.configureData.testingRatio}
                                                      imageURLs={this.state.configureData.imageURLs}
                                                      nbrClasses={this.state.configureData.nbrClasses}
                                                      needData={this.state.chooseModel.trainInitial || this.state.chooseModel.fullClassesTrain || this.state.chooseModel.trainOld || this.state.chooseModel.trainYoung}
            onStateChange={this.stepThreeHandler.bind(this)}/>
            default: return 'what'
        }
    }

    handleClose = () => {
        this.state.configureData.imageURLs.forEach((arr) => arr.forEach((url) => URL.revokeObjectURL(url)));
        this.setState(this.originalState)
        this.onClose()
    }

    handleBack = () => {
        const prev = this.state.activeStep;
        this.setState( {activeStep: prev-1})
    }

    handleNext = () => {
        const prev = this.state.activeStep;
        if (prev === 0) {
            if (this.state.chooseModel.modelType === 0) {
                let fixedStuff = this.state.configureData;
                fixedStuff.imageURLs = [[],[],[],[]]
                fixedStuff.nbrClasses = 4
                this.state.configureData.imageURLs.forEach((arr) => arr.forEach((url) => URL.revokeObjectURL(url)));
                this.setState({configureData: fixedStuff})
            } else {
                let fixedStuff = this.state.configureData;
                fixedStuff.imageURLs = [[],[],[],[],[],[],[],[],[]]
                fixedStuff.nbrClasses = 9
                this.state.configureData.imageURLs.forEach((arr) => arr.forEach((url) => URL.revokeObjectURL(url)));
                this.setState({configureData: fixedStuff})
            }
        }
        this.setState( {activeStep: prev+1})
        if (prev+1 === this.steps.length) {
            this.handleTrain();
        }
    }

    performRequest(req: any) {
        if (req !== "undefined") {
            fetch(`${backend}/train`, req).then((res) => {
                if (!res.ok) {
                    console.log("There was a problem with the train request!");
                } else {
                    res.json().then((body) => {
                        const jobId = body.jobId;

                        console.log(`MY TRAINING ID IS ${jobId}`);
                        this.onClose();
                        this.onTrain(jobId);
                    });
                }
            });
        }
    }

    resolveURLs(urls: any) {
        return new Promise((resolve, reject) => {
            console.log(urls);
            const count = urls.length;
            var result: any[] = [];
            urls.forEach((url: string) => {
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

    handleTrain = () => {
        console.log('creating training request!')

        if (this.state.chooseModel.modelType === 0)
        {
            // We are training an ensemble model
            if (this.state.configureData.custom && this.state.chooseModel.trainInitial) {
                this.resolveURLs(this.state.configureData.imageURLs[0]).then((youngImages) => {
                    this.resolveURLs(this.state.configureData.imageURLs[1]).then((adultImages) => {
                        this.resolveURLs(this.state.configureData.imageURLs[2]).then((middleAgedImages) => {
                            this.resolveURLs(this.state.configureData.imageURLs[3]).then((oldImages) => {
                                const body = {
                                    model_type: 'ensemble',
                                    train_initial: this.state.chooseModel.trainInitial,
                                    train_young: this.state.chooseModel.trainYoung,
                                    train_old: this.state.chooseModel.trainOld,
                                    optimizeK: this.state.configureModel.optimizeK,
                                    minK: this.state.configureModel.minK,
                                    maxK: this.state.configureModel.maxK,
                                    useCustomData: this.state.configureData.custom,
                                    youngImages: youngImages,
                                    adultImages: adultImages,
                                    middleAgedImages: middleAgedImages,
                                    oldImages: oldImages,
                                    testingRatio: this.state.configureData.testingRatio
                                }
                                const req = {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify(body),
                                };

                                this.performRequest(req);
                            })
                        })
                    })
                })
            } else {
                const body = {
                    model_type: 'ensemble',
                    train_initial: this.state.chooseModel.trainInitial,
                    train_young: this.state.chooseModel.trainYoung,
                    train_old: this.state.chooseModel.trainOld,
                    optimizeK: this.state.configureModel.optimizeK,
                    minK: this.state.configureModel.minK,
                    maxK: this.state.configureModel.maxK,
                    useCustomData: this.state.configureData.custom,
                    testingRatio: this.state.configureData.testingRatio
                }
                const req = {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                };

                this.performRequest(req);
            }

        } else {
            // we are trainging 9 classes
            if (this.state.configureData.custom  && this.state.chooseModel.trainInitial) {
                this.resolveURLs(this.state.configureData.imageURLs[0]).then((one) => {
                    this.resolveURLs(this.state.configureData.imageURLs[1]).then((two) => {
                        this.resolveURLs(this.state.configureData.imageURLs[2]).then((three) => {
                            this.resolveURLs(this.state.configureData.imageURLs[3]).then((four) => {
                                this.resolveURLs(this.state.configureData.imageURLs[4]).then((five) => {
                                    this.resolveURLs(this.state.configureData.imageURLs[5]).then((six) => {
                                        this.resolveURLs(this.state.configureData.imageURLs[6]).then((seven) => {
                                            this.resolveURLs(this.state.configureData.imageURLs[7]).then((eight) => {
                                                this.resolveURLs(this.state.configureData.imageURLs[8]).then((nine) => {
                                                    const body = {
                                                        model_type: 'classes',
                                                        train_initial: this.state.chooseModel.trainInitial,
                                                        useCustomData: this.state.configureData.custom,
                                                        one: one,
                                                        two: two,
                                                        three: three,
                                                        four: four,
                                                        five: five,
                                                        six: six,
                                                        seven: seven,
                                                        eight: eight,
                                                        nine: nine,
                                                        testingRatio: this.state.configureData.testingRatio
                                                    }
                                                    const req = {
                                                        method: "POST",
                                                        headers: { "Content-Type": "application/json" },
                                                        body: JSON.stringify(body),
                                                    };

                                                    this.performRequest(req);

                                                })
                                            })
                                        })
                                    })
                                })
                            })
                        })
                    })
                })
            } else {
                const body = {
                    model_type: 'classes',
                    train_initial: this.state.chooseModel.trainInitial,
                    useCustomData: this.state.configureData.custom,
                    testingRatio: this.state.configureData.testingRatio
                }
                const req = {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                };

                this.performRequest(req);

            }
        }
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
                </div>
        </Dialog>);
    }
}

export default TrainingDialog;
