using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra.Double;
using Shell.Enums;
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
        public double ErrorScale;
        public double DiscardWorstFactor;
        public float TrainSetPercentage;

        public int CtsPrevValuesCount;
        public int OutputCount;
        public int InputCount;
        public int PcaDimensions;
        public int RunsPerSettings;

        public bool UseBiases;
        public bool PlotAgainstInput;

        public NetworkType NetworkType;
        public PartIIProblemType ProblemType;

        public ReportingOptions ReportingOptions;

        public List<int> HiddenNeuronCounts;
        public List<DenseVector> CsvLines;
        public List<LearningSettings> SettingsToRun = new List<LearningSettings>();

        public string DataSetName;
        public string ParametersFileName;
    }
}
