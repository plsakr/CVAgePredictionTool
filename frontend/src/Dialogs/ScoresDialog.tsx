import * as React from "react";
import {Dialog} from "@material-ui/core";
import {EnsembleScoreData, ModelScoreDialogProps} from "../API/MyTypes";
import Slide from "@material-ui/core/Slide";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";
import "./Scores.css"
import {TransitionProps} from "@material-ui/core/transitions";
import Box from "@material-ui/core/Box";
import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";
import SwipeableViews from 'react-swipeable-views';
import EnsembleScores from "./Scores/EnsembleScores";
import CNNScores from "./Scores/CNNScores";




const Transition = React.forwardRef(function Transition(
    props: TransitionProps & { children?: React.ReactElement },
    ref: React.Ref<unknown>,
) {
    return <Slide direction="up" ref={ref} {...props} />;
});

interface TabPanelProps {
    children?: React.ReactNode;
    dir?: string;
    index: any;
    value: any;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box p={3}>
                    {children}
                </Box>
            )}
        </div>
    );
}

function a11yProps(index: any) {
    return {
        id: `simple-tab-${index}`,
        'aria-controls': `simple-tabpanel-${index}`,
    };
}

class ScoresDialog extends React.Component<ModelScoreDialogProps, {value: number}> {
    isOpen: boolean;
    onClose: () => void;
    ensembleData: EnsembleScoreData;

    state: {value: number} = {
        value: 0
    }

    handleChange = (event: React.ChangeEvent<{}>, newValue: number) => {
        this.setState({value: newValue})
    };

    handleChangeIndex = (index: number) => {
        this.setState({value: index});
    }

    constructor(props: ModelScoreDialogProps) {
        super(props);
        this.isOpen = props.isOpen;
        this.onClose = props.onClose;
        this.ensembleData = props.ensembleScores;
    }

    shouldComponentUpdate(nextProps: Readonly<ModelScoreDialogProps>, nextState: Readonly<{}>, nextContext: any): boolean {
        this.isOpen = nextProps.isOpen;
        this.ensembleData = nextProps.ensembleScores;
        return true;
    }

    randomNbrGenerator(): number[] {
        let result = [];
        for (let i = 0; i < 52; i++)
            result.push(Math.random())

        return result;
    }


    render() {
        console.log('rendering scores!')
        return (<Dialog fullScreen open={this.isOpen} onClose={this.onClose.bind(this)} TransitionComponent={Transition}>
            <AppBar className="appBar">
                <Toolbar>
                    <IconButton edge="start" color="inherit" onClick={this.onClose.bind(this)}>
                        <CloseIcon />
                    </IconButton>
                    <Typography variant="h6" className="title">Model Scores</Typography>
                </Toolbar>

                <Tabs value={this.state.value} onChange={this.handleChange.bind(this)} aria-label="simple tabs example">
                    <Tab label="Ensemble Model" {...a11yProps(0)} />
                    <Tab label="Young CNN" {...a11yProps(1)} />
                    <Tab label="Old CNN" {...a11yProps(2)} />
                </Tabs>
            </AppBar>
            <div className="dialogContent">
                <TabPanel value={this.state.value} index={0} dir="x">
                    <EnsembleScores modelName={this.ensembleData.modelName} modelParams={this.ensembleData.modelParams}
                    modelScores={this.ensembleData.modelScores}/>
                </TabPanel>
                <TabPanel value={this.state.value} index={1} dir="x">
                    <CNNScores scores={{accuracy: 0,
                        oneOff: 0,
                        twoOff: 0,
                        train_acc: this.randomNbrGenerator(),
                        val_acc: this.randomNbrGenerator(),
                        loss_train: this.randomNbrGenerator(),
                        loss_val: this.randomNbrGenerator()}} />
                </TabPanel>
                <TabPanel value={this.state.value} index={2}  dir="x">
                    <CNNScores scores={{accuracy: 0,
                        oneOff: 0,
                        twoOff: 0,
                        train_acc: this.randomNbrGenerator(),
                        val_acc: this.randomNbrGenerator(),
                        loss_train: this.randomNbrGenerator(),
                        loss_val: this.randomNbrGenerator()}} />
                </TabPanel>
            </div>

        </Dialog>);
    }
}

export default ScoresDialog;
