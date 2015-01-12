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
        public static string LearningRateTitle { get { return "LR"; } }
        public static string MomentumTitle { get { return "M"; } }        
        public static string MaxIterationsTitle { get { return "MAXIT"; } }
        public static string BadIterationsTitle { get {return "BADIT";} }
        public static string ActivationFuncTitle { get { return "FUNC"; } }
        public static string HLTitle { get { return "HL"; } }

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
            LearningRateTitle,
            MomentumTitle,
            MaxIterationsTitle,
            BadIterationsTitle,
            ActivationFuncTitle,
            HLTitle,
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
            ValidationSetSize = 0.125f;
            MaxIterations = 2000;
            MinIterations = 50;
            BadIterations = 10;
            LearningRate = 0.4;
            Momentum = 0.3;
            Activation = null;
            requiredParamParsers = new Dictionary<string, Action<string>>()
            {
                {LearningRateTitle, ParseLearningRate},
                {MomentumTitle, ParseMomentum},
                {MaxIterationsTitle, ParseMaxIterations},
                {BadIterationsTitle, ParseBadIterations},
                {ActivationFuncTitle, ParseActivationFunc},
                {HLTitle, ParseHiddenNeuronCounts},
            };
        }

        private LearningSettings(LearningSettings settings) : this()
        {
            ValidationSetSize = settings.ValidationSetSize;
            MaxIterations = settings.MaxIterations;
            BadIterations = settings.BadIterations;
            LearningRate = settings.LearningRate;
            Momentum = settings.Momentum;
            Activation = settings.Activation.Clone();
            MinIterations = settings.MinIterations;
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
                LearningRate, Momentum,
                Activation.Name, string.Join("-", HiddenNeuronCounts),
                MaxIterations, BadIterations, MinIterations, ValidationSetSize);
            return sb.ToString();
        }

        private void ParseLearningRate(string str)
        {
            LearningRate = double.Parse(str, CultureInfo.InvariantCulture);
            if(LearningRate <= 0)
            {
                throw new ArgumentException("Learning rate must be above 0");
            }
        }

        private void ParseMomentum(string str)
        {
            Momentum = double.Parse(str, CultureInfo.InvariantCulture);
            if(Momentum < 0 || Momentum >= 1)
            {
                throw new ArgumentException("Momentum must be within [0, 1)");
            }
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
            string[] layersText = str.Split(new string[] { "-", "_", "." }, 
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
