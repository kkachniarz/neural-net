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

        private int badIterations = 0;
        private int iteration = 0;
        private float vSetPercentage;
        private int vSetStart;

        private double currentError = 1.0;
        private double previousError = 1.0;
        private double lowestError = double.MaxValue;

        private object savedWeights = null;

        public VSetLearningStrategy(LearningSettings lSettings)
            : base(lSettings.LearningRate, lSettings.Momentum)
        {
            this.vSetPercentage = lSettings.ValidationSetSize;
            this.MaxBadIterations = lSettings.BadIterations;
            this.IterLimit = lSettings.MaxIterations;
        }

        public override List<double> Train(INetwork network, IDataSet data, IStatusReporter statusHolder)
        {
            vSetStart = (int)((1 - vSetPercentage) * data.PatternCount);
            return base.Train(network, data, statusHolder);
        }

        protected override double RunEpoch()
        {            
            DoTrainSetEpoch();
            currentError = CalculateMSEValidation();

            if(currentError < lowestError)
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

            if (badIterations > MaxBadIterations
                || iteration > IterLimit)
            {
                finished = true;
                // restore the best weights
                network.RestoreWeights(savedWeights);
                return CalculateMSEValidation();
            }

            previousError = currentError;
            iteration++;
            return currentError;
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
                iteration, IterLimit, errorHistory[errorHistory.Count - 1]));
        }

        private double CalculateMSEValidation()
        {
            double mse = 0.0;
            for (int i = vSetStart; i < dataSet.PatternCount; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mse, dataSet.PatternCount - vSetStart + 1, network.Activation);
        }

        private double CalculateMSETrain()
        {
            double mse = 0.0;
            for (int i = 0; i < vSetStart; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mse, vSetStart, network.Activation);
        }
    }
}
