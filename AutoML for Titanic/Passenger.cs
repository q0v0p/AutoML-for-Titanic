using Microsoft.ML.Data;

namespace LightGBM_for_Titanic
{
    internal class Passenger
    {
        [LoadColumn(0)]
        public float PassengerId;

        [LoadColumn(1)]
        public string Survived;

        [LoadColumn(2)]
        public float Pclass;

        [LoadColumn(3)]
        public string Name;

        [LoadColumn(4)]
        public string Sex;

        [LoadColumn(5)]
        public float Age;

        [LoadColumn(6)]
        public float SibSp;

        [LoadColumn(7)]
        public float Parch;

        [LoadColumn(8)]
        public string Ticket;

        [LoadColumn(9)]
        public float Fare;

        [LoadColumn(10)]
        public string Cabin;

        [LoadColumn(11)]
        public string Embarked;
    }

    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        // Predicted label from the trainer.
        public bool PredictedSurvived { get; set; }
    }
}
