import React, {ChangeEvent, ReactNode} from "react";
import {ConfigureState, DatasetState} from "../../API/MyTypes";
import {Box, Theme} from "@material-ui/core";
import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";
import './ConfigureDataset.css';
import DnDImgUploader from "../../HelperComponents/DnDImgUploader";
import {CheckBox} from "@material-ui/icons";
import Checkbox from "@material-ui/core/Checkbox";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Slider from "@material-ui/core/Slider";
import Typography from "@material-ui/core/Typography";


type DatasetProps = {
    custom: boolean,
    testingRatio: number,
    imageURLs: string[][],
    nbrClasses: number,
    onStateChange: (state: DatasetState) => void,
    needData: boolean
}

const classNamesFour = ['Young', 'Adult', 'Middle Aged', 'Old']
const classNamesNine = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-']

interface TabPanelProps {
    children?: React.ReactNode;
    index: any;
    value: any;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            className='uploader'
            role="tabpanel"
            hidden={value !== index}
            id={`vertical-tabpanel-${index}`}
            aria-labelledby={`vertical-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box p={3}>
                    <div>{children}</div>
                </Box>
            )}
        </div>
    );
}


function a11yProps(index: any) {
    return {
        id: `vertical-tab-${index}`,
        'aria-controls': `vertical-tabpanel-${index}`,
    };
}

const useStyles = (theme: Theme) => ({

    tabs: {
        borderRight: `1px solid ${theme.palette.divider}`,
    },
});


class ConfigureDatasetComponent extends React.Component<DatasetProps, DatasetState> {
    state: DatasetState;
    constructor(props: DatasetProps) {
        super(props);
        this.state = {
            custom: props.custom,
            testingRatio: props.testingRatio,
            imageURLs: props.imageURLs,
            nbrClasses: props.nbrClasses,
            tabValue: 0
        }
    }

    createOnReceiveURLs = (index: number) => {
        return (urls: string[]) => {
            this.setState((state: Readonly<DatasetState>, props: Readonly<DatasetProps>) => {
                const oldImages = state.imageURLs;
                oldImages[index] = [...state.imageURLs[index], ...urls]
                return {
                    imageURLs: oldImages
                }
            }, this.onStateChange.bind(this))
        }
    }

    onStateChange = () => {
        this.props.onStateChange(this.state);
    }

    handleTabChange = (event: React.ChangeEvent<{}>, newValue: number) => {
        this.setState({tabValue: newValue}, this.onStateChange.bind(this))
    }

    handleCustomChange = (event: React.ChangeEvent<{}>, newValue: boolean) => {
        this.setState({custom: newValue}, this.onStateChange.bind(this))
    }

    handleRatioChange = (event: ChangeEvent<{}>, newValue: number | number[]) => {
        this.setState({testingRatio: newValue as number}, this.onStateChange.bind(this))
    }

    render() {
        var child: ReactNode;
        if (!this.props.needData) {
            return (<p>No Configuration Needed!</p>)
        }
        const checkbox = (<FormControlLabel
            control={<Checkbox checked={this.state.custom} onChange={this.handleCustomChange.bind(this)} name="checkedA" />}
            label="Custom Dataset"
        />)
        if (this.state.custom) {
            const count = this.state.imageURLs.length
            child = (<div  className='root'>
                <Tabs
                    variant="scrollable"
                    orientation="vertical"
                    value={this.state.tabValue}
                    className='myTabs'
                onChange={this.handleTabChange.bind(this)}>
                    {this.state.imageURLs.map((_, index) => {
                       return <Tab label={count === 4 ? classNamesFour[index] : classNamesNine[index]} {...a11yProps(index)}/>
                    })}
                </Tabs>
                {
                    this.state.imageURLs.map((myURLS, index) => {
                        // @ts-ignore
                        return <TabPanel index={index} value={this.state.tabValue}>
                            <DnDImgUploader onDropURLs={this.createOnReceiveURLs(index).bind(this)} currentURLs={this.state.imageURLs[index]}/>
                        </TabPanel>
                    })
                }
            </div>);
        }
        const slider = ( <div className="testingRatio">
            <Typography>Testing Ratio:</Typography>
            <Slider
            onChange={this.handleRatioChange.bind(this)}
            value={this.state.testingRatio}
            valueLabelDisplay="auto"
            step={0.05}
            min={0.1}
            max={1}

        />
        </div>)

        return (<div>{checkbox}{child}{slider}</div>);
    }
}

export default ConfigureDatasetComponent;
