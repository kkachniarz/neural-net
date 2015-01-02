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
        private int vSetEnd;

        private double currentError = 1.0;
        private double previousError = 1.0;
        private double lowestError = double.MaxValue;

        private object savedWeights = null;
        private List<object> debugSavedWeights = new List<object>();

        public VSetLearningStrategy(LearningSettings lSettings)
            : base(lSettings.LearningRate, lSettings.Momentum)
        {
            this.vSetPercentage = lSettings.ValidationSetSize;
            this.MaxBadIterations = lSettings.BadIterations;
            this.IterLimit = lSettings.MaxIterations;
        }

        public override List<double> Train(INetwork network, IDataSet data, ILearningStatus statusHolder)
        {
            vSetEnd = (int)(vSetPercentage * data.PatternCount);
            return base.Train(network, data, statusHolder);
        }

        protected override double RunEpoch()
        {
            for (int i = vSetEnd; i < dataSet.PatternCount; i++) // train set (without validation set)
            {
                Pattern pattern = dataSet.GetPatternAt(i);
                Vector<double> networkAnswer = network.ComputeOutput(pattern.Input);
                Vector<double> modelAnswer = pattern.IdealOutput;

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(LearningRate, Momentum);
            }

            currentError = CalculateMSEValidation();
            debugSavedWeights.Add(network.SaveWeights());

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

        protected override void UpdateStatus()
        {
            statusHolder.SetStatusText(string.Format("iter: {0}, error: {1}", iteration, errorHistory[errorHistory.Count - 1]));
        }

        private double CalculateMSEValidation()
        {
            double mse = 0.0;
            for (int i = 0; i < vSetEnd; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mse, dataSet.PatternCount, network.Activation);
        }

        private double CalculateMSETrain()
        {
            double mse = 0.0;
            for (int i = vSetEnd; i < dataSet.PatternCount; i++)
            {
                Pattern p = dataSet.GetPatternAt(i);
                mse += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mse, dataSet.PatternCount, network.Activation);
        }
    }
}
