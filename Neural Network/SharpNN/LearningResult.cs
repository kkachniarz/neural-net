using SharpNN.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN
{
    public class LearningResult
    {
        public List<double> MSEHistory { get; set; }
        public int IterationsExecuted
        {
            get
            {
                return MSEHistory.Count - 1;
            }
        }
        // + data for plots

        public LearningResult()
        {
            MSEHistory = new List<double>();
        }
    }
}
