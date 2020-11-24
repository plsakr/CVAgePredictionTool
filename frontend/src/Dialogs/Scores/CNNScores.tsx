import { ResponsiveLine, Serie } from "@nivo/line";
import * as React from "react";
import {CNNScoreData} from "../../API/MyTypes";



function generateData(train_acc: number[], val_acc: number[]): Serie[] {
    let dataTrainAcc = [];

    for (let i = 0; i < train_acc.length; i++) {
        dataTrainAcc.push({"x": i+1, "y":train_acc[i]});
    }

    const train_acc_data = {"id": "Training", "color": "rgb(255,0,0)", "data": dataTrainAcc}

    let dataValAcc = [];
    for (let i = 0; i < val_acc.length; i++) {
        dataValAcc.push({"x": i+1, "y":val_acc[i]});
    }

    const val_acc_data = {"id": "Validation", "color": "rgb(0,255,0)", "data": dataValAcc}

    return [train_acc_data, val_acc_data]
}

class CNNScores extends React.Component<{scores: CNNScoreData}, {}> {
    scores: CNNScoreData;
    trainingData: Serie[];
    lossData: Serie[];


    constructor(props: {scores: CNNScoreData}) {
        super(props);
        this.scores = props.scores;
        this.trainingData = generateData(this.scores.train_acc, this.scores.val_acc);
        this.lossData = generateData(this.scores.loss_train, this.scores.loss_val);
    }

    shouldComponentUpdate(nextProps: Readonly<{ scores: CNNScoreData }>, nextState: Readonly<{}>, nextContext: any): boolean {
        this.scores = nextProps.scores;
        this.trainingData = generateData(this.scores.train_acc, this.scores.val_acc);
        this.lossData = generateData(this.scores.loss_train, this.scores.loss_val);
        return true;
    }

    MyResponsiveLine = () => {
        return (
            <ResponsiveLine
                data={this.trainingData}
                margin={{top: 50, right: 110, bottom: 50, left: 60}}
                xScale={{type: 'point'}}
                yScale={{type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false}}
                yFormat=" >-.2f"
                axisTop={null}
                axisRight={null}
                axisBottom={{
                    orient: 'bottom',
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'transportation',
                    legendOffset: 36,
                    legendPosition: 'middle'
                }}
                axisLeft={{
                    orient: 'left',
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'count',
                    legendOffset: -40,
                    legendPosition: 'middle'
                }}
                pointSize={10}
                pointColor={{theme: 'background'}}
                pointBorderWidth={2}
                pointBorderColor={{from: 'serieColor'}}
                pointLabelYOffset={-12}
                enableSlices="x"
                legends={[
                    {
                        anchor: 'bottom-right',
                        direction: 'column',
                        justify: false,
                        translateX: 100,
                        translateY: 0,
                        itemsSpacing: 0,
                        itemDirection: 'left-to-right',
                        itemWidth: 80,
                        itemHeight: 20,
                        itemOpacity: 0.75,
                        symbolSize: 12,
                        symbolShape: 'circle',
                        symbolBorderColor: 'rgba(0, 0, 0, .5)',
                        effects: [
                            {
                                on: 'hover',
                                style: {
                                    itemBackground: 'rgba(0, 0, 0, .03)',
                                    itemOpacity: 1
                                }
                            }
                        ]
                    }
                ]}
            />
        );
    }

    render() {
        return undefined;
    }
}

export default CNNScores;
