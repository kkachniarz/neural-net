using LearningNN.DataSet;
using MathNet.Numerics.LinearAlgebra;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.Learning
{
    public class VSetLearningStrategy : LearningStrategy
    {
        public int IterLimit { get; set; }
        public int MaxBadIterations { get; set; }
        public int MinIterations { get; set; }

        private int badIterations = 0;
        private int iteration = 0;
        private float vSetPercentage;
        private int vSetStart;

        private double currentError = 1.0;
        private double previousError = 1.0;
        private double lowestError = double.MaxValue;

        private double minValidOutput; // this is network min activation + margin value, that is, the minimum value that could represent an output in the data.
        private double maxValidOutput;

        private object savedWeights = null;

        public VSetLearningStrategy(LearningSettings lSettings)
            : base(lSettings.LearningRate, lSettings.Momentum)
        {
            this.vSetPercentage = lSettings.ValidationSetSize;
            this.MaxBadIterations = lSettings.BadIterations;
            this.IterLimit = lSettings.MaxIterations;
            this.MinIterations = lSettings.MinIterations < lSettings.MaxIterations ?
                lSettings.MinIterations : lSettings.MaxIterations;
        }

        public override List<double> Train(INetwork network, IDataSet data, IStatusReporter statusHolder)
        {
            vSetStart = (int)((1 - vSetPercentage) * data.PatternCount);
            Normalizor.GetMinMaxActivationWithMargin(network.Activation.MinValue, network.Activation.MaxValue,
                out minValidOutput, out maxValidOutput);
            return base.Train(network, data, statusHolder);
        }

        protected override double RunEpoch()
        {
            DoTrainSetEpoch();
            currentError = CalculateMSEValidation();

            if (currentError < lowestError)
            {
                lowestError = currentError;
                savedWeights = network.SaveWeights();
            }

            if (currentError > previousError)
            {
                badIterations++;
            }
            else
            {
                badIterations = 0;
            }

            if (ShouldStop())
            {
                finished = true;
                // restore the best weights
                network.RestoreWeights(savedWeights);
                return lowestError; // was CalculateMSEValidation() // last error if calculated now would be off (overestimated) due to context layer saving - ideally it should be saved just after train set...
            }

            previousError = currentError;
            iteration++;
            return currentError;
        }

        private bool ShouldStop()
        {
            return (badIterations > MaxBadIterations && iteration >= MinIterations)
                || iteration >= IterLimit;
        }

        private void DoTrainSetEpoch()
        {
            for (int i = 0; i < vSetStart; i++) // train set (without validation set)
            {
                Pattern pattern = dataSet.GetPatternAt(i);
                Vector<double> networkAnswer = network.ComputeOutput(pattern.Input);
                Vector<double> modelAnswer = pattern.IdealOutput;

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(LearningRate, Momentum);
            }
        }

        protected override void UpdateStatus()
        {
            statusHolder.UpdateStatus(string.Format("iter {0} / {1}, VSet err: {2}",
                iteration, IterLimit, errorHistory[errorHistory.Count - 1].ToString("E2")));
        }

        private double CalculateMSEValidation()
        {
            double mse = 0.0;
            for (int i = vSetStart; i < dataSet.PatternCount; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return CalculateEpochMSE(mse, dataSet.PatternCount - vSetStart);
        }

        private double CalculateMSETrain()
        {
            double mse = 0.0;
            for (int i = 0; i < vSetStart; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return CalculateEpochMSE(mse, vSetStart);
        }

        private double CalculateEpochMSE(double mse, double count)
        {
            return MSECalculator.CalculateEpochMSE(mse, count, minValidOutput, maxValidOutput);
        }
    }
}
