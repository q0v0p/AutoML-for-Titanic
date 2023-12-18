using AutoML_for_Titanic;

using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace.Option;
using Microsoft.ML.SearchSpace;

using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using System.Collections.Immutable;
using LightGBM_for_Titanic;
using System.Net.Sockets;
using System.Xml.Linq;

// Initialize MLContext
MLContext mlContext = new MLContext();

var dataPath = Path.GetFullPath(@"..\..\..\Data\train.csv");

// Infer column information
ColumnInferenceResults columnInference =
    mlContext.Auto().InferColumns(dataPath, labelColumnName: "Survived", groupColumns: true);

// Modify column inference results
columnInference.ColumnInformation.NumericColumnNames.Remove("PassengerId");
columnInference.ColumnInformation.NumericColumnNames.Remove("Name");
columnInference.ColumnInformation.NumericColumnNames.Remove("Pclass");
columnInference.ColumnInformation.CategoricalColumnNames.Add("Pclass");

// Create text loader
TextLoader loader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);

// Load data into IDataView
IDataView data = loader.Load(dataPath);

// Split into train (80%), validation (20%) sets
TrainTestData trainValidationData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

//Define pipeline
SweepablePipeline pipeline =
    mlContext.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation)
        .Append(mlContext.Auto().BinaryClassification(labelColumnName: columnInference.ColumnInformation.LabelColumnName, useFastForest:false, useLgbm: true));

// Create AutoML experiment
AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

// Configure experiment
experiment
    .SetPipeline(pipeline)
    .SetBinaryClassificationMetric(BinaryClassificationMetric.Accuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName)
    .SetTrainingTimeInSeconds(120)
    .SetEciCostFrugalTuner()
    .SetDataset(trainValidationData);

// Log experiment trials
var monitor = new AutoMLMonitor(pipeline);
experiment.SetMonitor(monitor);

//// Set checkpoints
//var checkpointPath = Path.Join(Directory.GetCurrentDirectory(), "automl");
//experiment.SetCheckpoint(checkpointPath);

// Run experiment
var cts = new CancellationTokenSource();
TrialResult experimentResults = await experiment.RunAsync(cts.Token);

// Get best model
var model = experimentResults.Model;

// Get all completed trials
var completedTrials = monitor.GetCompletedTrials();

// Measure trained model performance
// Apply data prep transformer to test data
// Use trained model to make inferences on test data
IDataView testDataPredictions = model.Transform(trainValidationData.TestSet);

// Save model
mlContext.Model.Save(model, data.Schema, "model.zip");
using FileStream stream = File.Create("./onnx_model.onnx");

var trainedModelMetrics = mlContext.BinaryClassification.Evaluate(testDataPredictions, labelColumnName: "Survived");

Console.WriteLine();
Console.WriteLine("Model quality metrics evaluation");
Console.WriteLine("--------------------------------");
Console.WriteLine($"Accuracy: {trainedModelMetrics.Accuracy:P2}");
Console.WriteLine($"Auc: {trainedModelMetrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1Score: {trainedModelMetrics.F1Score:P2}");
Console.WriteLine("=============== End of model evaluation ===============");

// Load Trained Model
DataViewSchema predictionPipelineSchema;
ITransformer predictionPipeline = model;

// Create PredictionEngines
PredictionEngine<Passenger, TitanicPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<Passenger, TitanicPrediction>(predictionPipeline);

// Input Data
var inputData = new List<Passenger>()
{
    new Passenger()
    {
        PassengerId = 1,
        Pclass = 3,
        Name = "Braund, Mr. Owen Harris",
        Sex = "male",
        Age = 22,
        SibSp = 1,
        Parch = 0,
        Ticket = "A/5 21171",
        Fare = 7.25f,
        Cabin = "",
        Embarked = "S"
    },
    new Passenger()
    {
        PassengerId = 2,
        Pclass = 1,
        Name = "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        Sex = "female",
        Age = 38,
        SibSp = 1,
        Parch = 0,
        Ticket = "PC 17599",
        Fare = 71.2833f,
        Cabin = "C85",
        Embarked = "C"
    },
    new Passenger()
    {
        PassengerId = 3,
        Pclass = 3,
        Name = "Heikkinen, Miss. Laina",
        Sex = "female",
        Age = 26,
        SibSp = 0,
        Parch = 0,
        Ticket = "STON/O2. 3101282",
        Fare = 7.925f,
        Cabin = "",
        Embarked = "S"
    },
};

// Get Prediction
foreach (var input in inputData)
{
    var prediction = predictionEngine.Predict(input);
    Console.WriteLine($"Id:{input.PassengerId} Name:{input.Name} Survived:{prediction.PredictedSurvived}");
}

Console.ReadLine();