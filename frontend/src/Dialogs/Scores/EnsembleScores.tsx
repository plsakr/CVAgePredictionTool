import * as React from "react";
import {EnsembleScoreData, ModelParams, ModelScores} from "../../API/MyTypes";


class EnsembleScores extends React.Component<EnsembleScoreData, any> {
    modelName: string;
    modelScores: ModelScores;
    modelParams: ModelParams;


    constructor(props: EnsembleScoreData) {
        super(props);
        this.modelName = props.modelName;
        this.modelScores = props.modelScores;
        this.modelParams = props.modelParams
    }

    shouldComponentUpdate(nextProps: Readonly<EnsembleScoreData>, nextState: Readonly<any>, nextContext: any): boolean {
        this.modelName = nextProps.modelName;
        this.modelScores = nextProps.modelScores;
        this.modelParams = nextProps.modelParams;
        return true;
    }

    render() {
        return (
            <div>
                <p className="info">
                    <em>Classifier Type: </em> {this.modelName === "pretrained_ensemble" ? "Pretrained Model" : "User Trained Model"}
                </p>
                <p className="info">
                    <em>Parameters: </em> K = {this.modelParams.K}
                </p>
                <p className="info">
                    <em>Training Instances: </em> {this.modelParams.train_nbr}
                </p>
                <p className="info">
                    <em>Testing Instances: </em> {this.modelParams.test_nbr}
                </p>
                <p className="info">
                    <em>Accuracy: </em> {Math.round(this.modelScores.acc * 100)}%
                </p>
                <table>
                    <thead>
                    <tr>
                        <th />
                        <th>Precision</th>
                        <th>Recall</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td>Young</td>
                        <td>{Math.round(this.modelScores.Young.precision * 100)}%</td>
                        <td>{Math.round(this.modelScores.Young.recall * 100)}%</td>
                    </tr>
                    <tr>
                        <td>Old</td>
                        <td>{Math.round(this.modelScores.Old.precision * 100)}%</td>
                        <td>{Math.round(this.modelScores.Old.recall * 100)}%</td>
                    </tr>
                    </tbody>
                </table>
            </div>

        );
    }
}

export default EnsembleScores;
