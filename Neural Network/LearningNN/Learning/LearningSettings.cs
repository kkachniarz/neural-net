using SharpNN;
using SharpNN.ActivationFunctions;
using System;
using System.Collections.Generic;
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

        private static List<string> requiredTitles = new List<string>()
        {
            "LR",
            "M",
            "MAXIT",
            "BADIT", 
            "FUNC"
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
        }

        public LearningSettings Clone()
        {
            LearningSettings clone = new LearningSettings(this);
            return clone;
        }

        public void SetParamByTitle(string title, string value)
        {
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
Iteration limits: (Max/Bad/Min) {2}/{3}/{4}
Activation: {5}
Validation Set Size: {6}",
                LearningRate, Momentum, MaxIterations, BadIterations, MinIterations, 
                Activation.Name, ValidationSetSize);
            return sb.ToString();
        }

        private void ParseLearningRate(string str)
        {
            LearningRate = double.Parse(str);
        }

        private void ParseMomentum(string str)
        {
             Momentum = double.Parse(str);
        }

        private void ParseMaxIterations(string str)
        {
            MaxIterations = int.Parse(str);
        }

        private void ParseBadIterations(string str)
        {
             BadIterations = int.Parse(str);
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
