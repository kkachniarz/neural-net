using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.Learning
{
    /// <summary>
    /// Settings concerning learning.  
    /// The settings here never require rebuilding the data set (train and test sets are never changed).
    /// </summary>
    public class LearningSettings
    {
        public static readonly List<string> RequiredTitles = new List<string>()
        {
            "LR",
            "M",
            "MAXIT",
            "BADIT",
        };

        public LearningSettings()
        {
            ValidationSetSize = 0.2f;
            MaxIterations = 2000;
            BadIterations = 10;
            LearningRate = 0.4;
            Momentum = 0.3;
        }

        private LearningSettings(LearningSettings settings)
        {
            ValidationSetSize = settings.ValidationSetSize;
            MaxIterations = settings.MaxIterations;
            BadIterations = settings.BadIterations;
            LearningRate = settings.LearningRate;
            Momentum = settings.Momentum;
        }

        public LearningSettings Clone()
        {
            LearningSettings clone = new LearningSettings(this);
            return clone;
        }

        public int MaxIterations { get; set; }
        public int BadIterations { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public float ValidationSetSize { get; set; }

        public void SetParamByTitle(string title, string value)
        {
            switch(title) // could be refactored
            {
                case "LR":
                    LearningRate = double.Parse(value);
                    break;
                case "M":
                    Momentum = double.Parse(value);
                    break;
                case "MAXIT":
                    MaxIterations = int.Parse(value);
                    break;
                case "BADIT":
                    BadIterations = int.Parse(value);
                    break;
                default:
                    throw new ArgumentException(string.Format("Unknown parameter title {0}", title));
            }
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Learning Rate: {0}\r\nMomentum: {1}\r\nMax Iterations: {2}\r\nBad Iterations: {3}",
                LearningRate, Momentum, MaxIterations, BadIterations);
            return sb.ToString();
        }
    }
}
