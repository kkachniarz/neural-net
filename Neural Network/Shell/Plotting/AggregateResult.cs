using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN;
using Shell.Containers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Plotting
{
    /// <summary>
    /// Contains all results (runs) for given LearningSettings. This way we get averages over N runs for given settings.
    /// </summary>
    public class AggregateResult
    {
        public double AverageError { get; private set; }
        public double AverageSecondsTaken { get; private set; }
        public double PercentageLearningStuck { get; private set; }
        public int RunCount { get; private set; }       
        public int BestErrorIndex { get; private set; }
        public SingleRunReport BestReport { get { return Reports[0]; } }

        private Vector<double> Errors;
        private Vector<double> DirectionMisguessRates;
        private Vector<double> MPVer2;
        private Vector<double> Iterations;
        private List<SingleRunReport> Reports;

        private LearningSettings SettingsUsed;      

        public AggregateResult(List<SingleRunReport> sortedReports, LearningSettings settings)
        {
            Errors = new DenseVector(sortedReports.Count);
            DirectionMisguessRates = new DenseVector(sortedReports.Count);
            MPVer2 = new DenseVector(sortedReports.Count);
            Iterations = new DenseVector(sortedReports.Count);
            List<TimeSpan> timesTaken = new List<TimeSpan>(sortedReports.Count);

            for (int i = 0; i < sortedReports.Count; i++)
            {
                Errors[i] = sortedReports[i].LearningResult.TestSetError;
                DirectionMisguessRates[i] = sortedReports[i].LearningResult.DirectionMisguessRate;
                MPVer2[i] = sortedReports[i].LearningResult.DirectionGuessVer2;
                Iterations[i] = sortedReports[i].LearningResult.IterationsExecuted;
                timesTaken.Add(sortedReports[i].LearningResult.TimeTaken);
            }

            RunCount = sortedReports.Count;
            SettingsUsed = settings;
            AverageError = Errors.Average();
            AverageSecondsTaken = timesTaken.Average(x => x.TotalSeconds);
            Reports = sortedReports.ToList(); // create copies of references
            BestErrorIndex = Errors.MinimumIndex();
            PercentageLearningStuck = (double)sortedReports.Count(x => x.LearningResult.GotStuck) / (double)sortedReports.Count;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Test Set Error: M = {0}   SD = {1}\r\n", Errors.Average().ToString("E2"), 
                Errors.StandardDeviation().ToString("E2"));
            sb.AppendFormat("Test Set direction misguess rate: M = {0}   SD = {1}\r\n", 
                DirectionMisguessRates.Average().ToString("E2"), 
                DirectionMisguessRates.StandardDeviation().ToString("E2"));
            sb.AppendFormat("MP ver. 2: M = {0}   SD = {1}\r\n",
                MPVer2.Average().ToString("E2"), MPVer2.StandardDeviation().ToString("E2"));
            sb.AppendFormat("Iterations: M = {0}  SD = {1}\r\n", Iterations.Average().ToString("F1"),
                Iterations.StandardDeviation().ToString("F1"));
            sb.AppendFormat("Time taken: M = {0}s.\r\n", AverageSecondsTaken.ToString("F1"));
            sb.AppendLine();

            // the two best scores are not necessarily from the same run!
            sb.AppendFormat("Best error: {0}, iterations used: {1}\r\n", Errors.Min().ToString("E2"), Iterations[BestErrorIndex]);
            sb.AppendFormat("Best direction misguess factor: {0}, iterations used: {1}\r\n", 
                DirectionMisguessRates.Min().ToString("E2"), Iterations[DirectionMisguessRates.MinimumIndex()]);
            sb.AppendFormat("Training got stuck: {0} of the time", PercentageLearningStuck.ToString("P1"));
            sb.AppendLine();

            //sb.AppendFormat("Worst error: {0}\r\n", Errors.Max().ToString());
            //sb.AppendFormat("Worst direction guess factor: {0}\r\n", Directions.Min().ToString());
            //sb.AppendLine();

            //sb.AppendFormat("Run count: {0}\r\n", RunCount.ToString());
            sb.AppendLine(SettingsUsed.ToString());
            sb.AppendLine("-----------------");

            return sb.ToString();
        }
    }
}
