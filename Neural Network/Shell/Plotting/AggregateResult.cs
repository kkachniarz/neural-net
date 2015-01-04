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
        public int RunCount;
        public Vector<double> Errors;
        public Vector<double> Directions;
        public Vector<double> Iterations;
        public List<SingleRunReport> Reports;
        public double AverageError;

        public LearningSettings SettingsUsed;
        public INetwork Network;

        public AggregateResult(List<SingleRunReport> reports, LearningSettings settings)
        {
            Errors = new DenseVector(reports.Count);
            Directions = new DenseVector(reports.Count);
            Iterations = new DenseVector(reports.Count);

            for (int i = 0; i < reports.Count; i++)
            {
                Errors[i] = reports[i].LearningResult.TestSetError;
                Directions[i] = reports[i].LearningResult.TestSetDirectionGuessed;
                Iterations[i] = reports[i].LearningResult.IterationsExecuted;
            }

            RunCount = reports.Count;
            SettingsUsed = settings;
            Network = reports[0].Network;
            AverageError = Errors.Average();
            Reports = reports.ToList(); // create copies of references
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Test Set Error: M = {0}   SD = {1}\r\n", Errors.Average().ToString("F8"), 
                Errors.StandardDeviation().ToString("F6"));
            sb.AppendFormat("Test Set Direction guessed: M = {0}   SD = {1}\r\n", Directions.Average().ToString("F8"), 
                Directions.StandardDeviation().ToString("F6"));
            sb.AppendFormat("Iterations: M = {0}  SD = {1}\r\n", Iterations.Average().ToString("F1"),
                Iterations.StandardDeviation().ToString("F1"));
            sb.AppendLine();

            //sb.AppendFormat("Best error: {0}\r\n", Errors.Min().ToString());
            //sb.AppendFormat("Best direction guess factor: {0}\r\n", Directions.Max().ToString()); // the two best scores are not necessarily from the same run!
            //sb.AppendLine();

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
