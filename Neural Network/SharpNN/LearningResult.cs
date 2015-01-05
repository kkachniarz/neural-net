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
        public double TestSetError { get; set; }
        public double DirectionGuessRate { get; set; }
        public TimeSpan TimeTaken { get; set; }
        public double DirectionMisguessRate
        {
            get { return 1.0 - DirectionGuessRate; }
        }

        public int IterationsExecuted
        {
            get
            {
                return MSEHistory.Count - 1;
            }
        }

        public double FinalTrainError // last error in the history recorded during training. This will normally be the last error measured on the validation set.
        {  
            get
            {
                return MSEHistory[MSEHistory.Count - 1];
            }
        }

        public LearningResult()
        {
            MSEHistory = new List<double>();
            TestSetError = 0;
            DirectionGuessRate = 0;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Iterations executed: {0}\r\n", IterationsExecuted);
            sb.AppendFormat("Error on validation set: {0}\r\n", FinalTrainError.ToString("E2"));
            sb.AppendFormat("Error on test set: {0}\r\n", TestSetError.ToString("E2"));
            sb.AppendFormat("Direction misguess: {0}\r\n", DirectionMisguessRate.ToString("E2"));
            sb.AppendFormat("Time taken: {0}s.\r\n", TimeTaken.TotalSeconds.ToString("F1"));
            return sb.ToString();
        }
    }
}
