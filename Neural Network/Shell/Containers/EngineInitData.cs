using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra.Double;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Containers
{
    public class EngineInitData
    {
        public const double ERROR_SCALE = 1000.0;
        public const double DISCARD_FACTOR = 0.2;
        public const int DISPLAY_LIMIT = 10;
        public int RunCounter = 0;
        public int RunsPerSettings = 1;
        public bool PlotAgainstInput = false;
        public string ResultsDirectoryPath;

        public Dictionary<LearningSettings, List<SingleRunReport>> resultsBySettings = new Dictionary<LearningSettings, List<SingleRunReport>>();

        public ReportingOptions ReportingOptions;
        public List<DenseVector> CsvLines;
        public List<LearningSettings> SettingsToRun = new List<LearningSettings>();
        public IDataSet TestDataSet;
        public IDataSet TrainDataSet;
        public ILearningStrategy LearningStrategy;

        public string DataSetName;
        public string ParametersFileName = "";
    }
}
