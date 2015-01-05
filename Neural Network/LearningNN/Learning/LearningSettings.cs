using SharpNN;
using SharpNN.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.Learning
{
    /// <summary>
    /// Settings concerning learning.  
    /// </summary>
    public class LearningSettings
    {
        private Dictionary<string, Action<string>> requiredParamParsers;
        public int MaxIterations { get; set; }
        public int BadIterations { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public float ValidationSetSize { get; set; }
        public int MinIterations { get; set; }
        public IActivation Activation { get; set; }
        public List<int> HiddenNeuronCounts { get; set; }

        private static List<string> requiredTitles = new List<string>()
        {
            "LR",
            "M",
            "MAXIT",
            "BADIT", 
            "FUNC",
            "HL"
        };

        public static List<string> RequiredTitles
        {
            get
            {
                return requiredTitles;
            }
        }

        public LearningSettings()
        {
            ValidationSetSize = 0.2f;
            MaxIterations = 2000;
            MinIterations = 50;
            BadIterations = 10;
            LearningRate = 0.4;
            Momentum = 0.3;
            Activation = null;
            requiredParamParsers = new Dictionary<string, Action<string>>()
            {
                {"LR", ParseLearningRate},
                {"M", ParseMomentum},
                {"MAXIT", ParseMaxIterations},
                {"BADIT", ParseBadIterations},
                {"FUNC", ParseActivationFunc},
                {"HL", ParseHiddenNeuronCounts},
            };
        }

        private LearningSettings(LearningSettings settings) : this()
        {
            ValidationSetSize = settings.ValidationSetSize;
            MaxIterations = settings.MaxIterations;
            BadIterations = settings.BadIterations;
            LearningRate = settings.LearningRate;
            Momentum = settings.Momentum;
            MinIterations = settings.MinIterations;
            Activation = settings.Activation.Clone();
            HiddenNeuronCounts = settings.HiddenNeuronCounts.ToList();
        }

        public LearningSettings Clone()
        {
            LearningSettings clone = new LearningSettings(this);
            return clone;
        }

        public void SetParamByTitle(string title, string value)
        {
            title = title.ToUpper();
            value = value.ToUpper();
            if(!requiredParamParsers.ContainsKey(title))
            {
                throw new ArgumentException(string.Format("Parameter name not recognized: {0}", title));
            }

            requiredParamParsers[title](value);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat(@"Learning Rate: {0}
Momentum: {1}
Activation: {2}
Hidden neuron counts: {3}
Iteration limits: (Max/Bad/Min) {4}/{5}/{6}
Validation Set Size: {7}",
                LearningRate, Momentum, Activation.Name, string.Join("-", HiddenNeuronCounts),
                MaxIterations, BadIterations, MinIterations, ValidationSetSize);
            return sb.ToString();
        }

        private void ParseLearningRate(string str)
        {
            LearningRate = double.Parse(str, CultureInfo.InvariantCulture);
        }

        private void ParseMomentum(string str)
        {
            Momentum = double.Parse(str, CultureInfo.InvariantCulture);
        }

        private void ParseMaxIterations(string str)
        {
            MaxIterations = int.Parse(str, CultureInfo.InvariantCulture);
        }

        private void ParseBadIterations(string str)
        {
            BadIterations = int.Parse(str, CultureInfo.InvariantCulture);
        }

        private void ParseHiddenNeuronCounts(string str)
        {
            string[] layersText = str.Split(new string[] { ",", " ", "-", "_", "." }, 
                StringSplitOptions.RemoveEmptyEntries);
            HiddenNeuronCounts = layersText.Select(s => int.Parse(s)).ToList();
        }

        private void ParseActivationFunc(string str)
        {
            if(str == "U")
            {
                Activation = new UnipolarSigmoidActivation();
            }
            else if(str == "B")
            {
                Activation = new BipolarTanhActivation();
            }
            else
            {
                throw new ArgumentException("Unknown activation function specification");
            }
        }
    }
}
