using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    /// <summary>
    /// Settings concerning learning.  
    /// The settings here never require rebuilding the data set (train and test sets are never changed).
    /// </summary>
    public class LearningSettings
    {
        public int MaxIterations { get; set; }
        public int BadIterations { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public float ValidationSetSize { get; set; }
    }
}
