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
    acc: number,
    test_score: number
}

export type ModelInfo = {
    isTraining: boolean,
    model_name: string,
    model_params: ModelParams,
    model_scores: ModelScores,
    trainingId: number
}

export type ModelParam = {
    K: number,
    trainInstances: number,
    testInstances: number
}

export type AppState = {
    ensembleData: EnsembleScoreData,
    isTraining: boolean,
    trainingId: number,
    isScoresOpen: boolean,
    isTrainingOpen: boolean,
};


export type ModelScoreDialogProps = {
    isOpen: boolean,
    onClose: () => void,
    ensembleScores: EnsembleScoreData,
}

export type EnsembleScoreData = {
    modelName: string,
    modelParams: ModelParams,
    modelScores: ModelScores
}

export type TrainingDialogProps = {
    isOpen: boolean,
    onClose: () => void
}

export type TrainingDialogState = {
    activeStep: number
}

export type CNNScoreData = {
    accuracy: number,
    oneOff: number,
    twoOff: number,
    train_acc: number[],
    val_acc: number[],
    loss_train: number[],
    loss_val: number[]
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

