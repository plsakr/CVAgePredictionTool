export type ModelScore = {
    classAccuracy: number,
    classOneoff: number,
    youngPrecision: number,
    youngRecall: number,
    oldPrecision: number,
    oldRecall: number,
    ensembleACC: number
}

export type ModelParams = {
    K: number,
    train_nbr: number,
    test_nbr: number
}

export type PrecisionRecall = {
    precision: number,
    recall: number,
    support: number
}

export type ModelScores = {
    Old: PrecisionRecall,
    Young: PrecisionRecall,
    acc: number
}

export type ModelInfo = {
    model_name: string
    ensemble_score: {
        K: number,
        KNNScore: {
            0: PrecisionRecall,
            1: PrecisionRecall,
            accuracy: number
        },


    },
    old_nn_score: CNNScoreData,
    young_nn_score: CNNScoreData,
    // isTraining: boolean,
    // model_name: string,
    // model_params: ModelParams,
    // model_scores: ModelScores,
    // trainingId: number
}

export type ModelParam = {
    K: number,
    trainInstances: number,
    testInstances: number
}

export type AppState = {
    ensembleData: EnsembleScoreData,
    cnnsData: {
        young: CNNScoreData,
        old: CNNScoreData
    } | undefined
    isTraining: boolean,
    trainingId: number,
    isScoresOpen: boolean,
    isTrainingOpen: boolean,
};


export type ModelScoreDialogProps = {
    isOpen: boolean,
    onClose: () => void,
    ensembleScores: EnsembleScoreData,
    neuralNetworkScores: {
        young: CNNScoreData,
        old: CNNScoreData
    }
}

export type EnsembleScoreData = {
    modelName: string,
    modelParams: ModelParams,
    modelScores: ModelScores
}

export type TrainingDialogProps = {
    isOpen: boolean,
    onClose: () => void,
    onTrain: (jobId: number) => void
}

export type TrainingDialogState = {
    activeStep: number,
    chooseModel: ChooseState,
    configureModel: ConfigureState,
    configureData: DatasetState
}

export type CNNScoreData = {
    accuracy: number,
    oneOff: number,
    twoOff: number,
    trainTime: number,
    history: {
        loss: number[],
        accuracy: number[],
        val_loss: number[],
        val_accuracy: number[]
    }
}

export type TrainMenuState = {
    isOpen: boolean,
    tabValue: string,
    minK: number,
    maxK: number,
    oldPicsNbr: number,
    youngPicsNbr: number,
    testingRatio: number,
    isChecked: boolean,
    youngUrls: string[],
    oldUrls: string[]
}

export type TrainMenuProps = {
    onTrain: (arg0: number) => void
}

export type ChooseState = {
    modelType: number,
    initialType: string,
    trainInitial: boolean,
    trainYoung: boolean,
    trainOld: boolean,
    fullClassesTrain: boolean,
}

export type ConfigureState = {
    chosen: ChooseState | null,
    optimizeK: boolean,
    minK: number,
    maxK: number,
    svmKernel: string,
    youngLayers: number,
    oldLayers: number,
    fullClassifierLayers: number
}

export type DatasetState = {
    custom: boolean,
    testingRatio: number,
    imageURLs: string[][],
    nbrClasses: number,
    tabValue: number
}
