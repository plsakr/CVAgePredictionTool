import React from "react";
import {ChooseState, ConfigureState} from "../../API/MyTypes";
import {FormControl, FormGroup} from "@material-ui/core";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import {CheckBox} from "@material-ui/icons";
import Checkbox from "@material-ui/core/Checkbox";
import Slider from "@material-ui/core/Slider";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import Typography from "@material-ui/core/Typography";
import TextField from "@material-ui/core/TextField";

type ConfigureProps = {
    toConfigure: ChooseState,
    optimizeK: boolean,
    minK: number,
    maxK: number,
    svmKernel: string,
    youngLayers: number,
    oldLayers: number,
    fullClassifierLayers: number,
    onStateChange: (state: ConfigureState) => void
}

class ConfigureModels extends React.Component<ConfigureProps, ConfigureState>{
    state: ConfigureState;

    constructor(props: ConfigureProps) {
        super(props);
        this.state = {
            chosen: props.toConfigure,
            optimizeK: props.optimizeK,
            minK: props.minK,
            maxK: props.maxK,
            svmKernel: props.svmKernel,
            youngLayers: props.youngLayers,
            oldLayers: props.oldLayers,
            fullClassifierLayers: props.fullClassifierLayers,
        }
    }

    onOptimizeChange = (event: React.ChangeEvent<HTMLInputElement>, checked: boolean) => {
        this.setState({optimizeK: checked}, this.onStateChange.bind(this))
    };

    onMinSliderChange = (event: React.ChangeEvent<{}>, value: (number | number[])) => {
        this.setState({minK: value as number}, this.onStateChange.bind(this))
    }

    onMaxSliderChange = (event: React.ChangeEvent<{}>, value: (number | number[])) => {
        this.setState({maxK: value as number}, this.onStateChange.bind(this))
    }

    onYoungNbrChange = (event: {target: {value: string}}) => {
        this.setState({youngLayers: parseInt(event.target.value)}, this.onStateChange.bind(this))
    }

    onOldNbrChange = (event: {target: {value: string}}) => {
        this.setState({oldLayers: parseInt(event.target.value)}, this.onStateChange.bind(this))
    }

    onFullNbrChange = (event: {target: {value: string}}) => {
        this.setState({fullClassifierLayers: parseInt(event.target.value)}, this.onStateChange.bind(this))
    }

    onStateChange = () => {
        this.props.onStateChange(this.state);
    }


    render() {
        return (<div>
            { (this.state.chosen?.modelType === 0) ? <div>
                    {((!this.state.chosen.trainInitial || this.state.chosen.initialType === "SVM")) ?
                        <p>No Configuration Needed!</p>
                        :
                        <div>
                            { (this.state.chosen.initialType.includes('KNN')) ? <div>

                                <FormGroup>
                                    <FormControlLabel control={<Checkbox checked={this.state.optimizeK}
                                                                         onChange={this.onOptimizeChange.bind(this)}/>}
                                                      label="Optimize K"/>
                                    <Typography>{this.state.optimizeK ? "Minimum K" : "K"}</Typography>
                                    <Slider onChange={this.onMinSliderChange.bind(this)}
                                        value={this.state.minK}
                                        min={1}
                                        max={99}
                                        valueLabelDisplay="auto"/>
                                    <Typography>Maximum K</Typography>
                                    <Slider onChange={this.onMaxSliderChange.bind(this)}
                                            value={this.state.maxK}
                                            min={this.state.minK + 1}
                                            max={100}
                                            disabled={!this.state.optimizeK}
                                            valueLabelDisplay="auto"/>
                                </FormGroup>
                            </div> : <div></div>}
                            {/*{ (this.state.chosen.trainYoung) ? <div>*/}
                            {/*        <TextField id="outlined-basic" label="Young Connected Layers" variant="outlined" onChange={this.onYoungNbrChange.bind(this)} value={this.state.youngLayers}/>*/}
                            {/*</div> : <div></div>}*/}
                            {/*{ (this.state.chosen.trainOld) ? <div>*/}
                            {/*        <TextField id="outlined-basic" label="Old Connected Layers" variant="outlined" onChange={this.onOldNbrChange.bind(this)} value={this.state.oldLayers}/>*/}
                            {/*</div> : <div></div>}*/}
                        </div>
                    }
                </div>
             :
                <div>
                <p>No Configuration Needed!</p>
                </div>
            }
        </div>);
    }
}

export default ConfigureModels;
