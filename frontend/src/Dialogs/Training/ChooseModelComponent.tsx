import React from "react";
import {FormControl} from "@material-ui/core";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import FormLabel from "@material-ui/core/FormLabel";
import RadioGroup from "@material-ui/core/RadioGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Radio from "@material-ui/core/Radio";

import './ChooseModel.css';
import {ChooseState} from "../../API/MyTypes";

type ChooseProps = {
    modelType: number, // 0 -> ensemble, 1 -> 9-class CNN
    initialType: string,
    trainInitial: boolean, // false -> go with pretrained. if initialType === 'ensemble' && trainInitial === false, then go with pretrained, else train
    onStateChange: (state: ChooseState) => void,
    trainYoung: boolean,
    trainOld: boolean,
    fullClassification: boolean
}

class ChooseModelComponent extends React.Component<ChooseProps, ChooseState>{
    state: ChooseState;

    constructor(props: ChooseProps) {
        super(props);
        this.state = {
            modelType: props.modelType,
            initialType: props.initialType,
            trainInitial: props.trainInitial,
            trainYoung: props.trainYoung,
            trainOld: props.trainOld,
            fullClassesTrain: props.fullClassification
        }
    }

    onStateChange = () => {
        this.props.onStateChange(this.state);
    }

    onTypeChange = (event: React.ChangeEvent<{ value: unknown }>) => {
        this.setState({modelType: event.target.value as number}, this.onStateChange.bind(this))
    }

    onInitialClassifierChange = (event: React.ChangeEvent<{value: unknown}>) => {
        if (event.target.value === 'Pretrained') {
            this.setState({initialType: event.target.value as string, trainInitial: false}, this.onStateChange.bind(this))
        } else {
            this.setState({initialType: event.target.value as string, trainInitial: true}, this.onStateChange.bind(this))
        }
    }

    onYoungClassifierChange = (event: React.ChangeEvent<{value: unknown}>) => {
        this.setState({trainYoung: event.target.value === 'true'}, this.onStateChange.bind(this))
    }

    onOldClassifierChange = (event: React.ChangeEvent<{value: unknown}>) => {
        this.setState({trainOld: event.target.value === 'true'}, this.onStateChange.bind(this))
    }

    onFUllClassifierChange = (event: React.ChangeEvent<{value: unknown}>) => {
        this.setState({fullClassesTrain: event.target.value === 'true'}, this.onStateChange.bind(this))
    }

    render() {
        return (<div className="bigContent">
            <div className="textField formItem">
                <FormControl>
                    <InputLabel>Model Type</InputLabel>
                    <Select value={this.state.modelType} onChange={this.onTypeChange.bind(this)}>
                        <MenuItem value={0}>Ensemble Model</MenuItem>
                        <MenuItem value={1}>9-Class CNN</MenuItem>
                    </Select>
                </FormControl>
            </div>

            {this.state.modelType === 0 ? (
                <div className="bigContent">
                    <FormControl className="formItem">
                        <FormLabel>Initial Classifier:</FormLabel>
                        <RadioGroup value={this.state.initialType} onChange={this.onInitialClassifierChange.bind(this)} row>
                            <FormControlLabel value="KNN+SVM" control={<Radio />} label="KNN+SVM" />
                            <FormControlLabel value="Pretrained" control={<Radio />} label="Pretrained KNN+SVM" />
                        </RadioGroup>
                    </FormControl>
                    <FormControl className="formItem">
                        <FormLabel>Young Classifier:</FormLabel>
                        <RadioGroup value={this.state.trainYoung} onChange={this.onYoungClassifierChange.bind(this)} row>
                            <FormControlLabel value={true} control={<Radio />} label="Train New CNN" />
                            <FormControlLabel value={false} control={<Radio />} label="Use Pretrained CNN" />
                        </RadioGroup>
                    </FormControl>
                    <FormControl className="formItem">
                        <FormLabel>Old Classifier:</FormLabel>
                        <RadioGroup value={this.state.trainOld} onChange={this.onOldClassifierChange.bind(this)} row>
                            <FormControlLabel value={true} control={<Radio />} label="Train New CNN" />
                            <FormControlLabel value={false} control={<Radio />} label="Use Pretrained CNN" />
                        </RadioGroup>
                    </FormControl>
                </div>
            ) : (<div className="myContent">
                <FormControl className="formItem">
                    <FormLabel>Choose a Classifier:</FormLabel>
                    <RadioGroup value={this.state.fullClassesTrain} onChange={this.onFUllClassifierChange.bind(this)} row>
                        <FormControlLabel value={false} control={<Radio />} label="Use Pretrained CNN" />
                    </RadioGroup>
                </FormControl>
            </div>)}

        </div>);
    }
}

export default ChooseModelComponent;
