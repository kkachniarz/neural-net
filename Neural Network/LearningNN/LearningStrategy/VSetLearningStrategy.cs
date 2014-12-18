using LearningNN.DataSet;
using MathNet.Numerics.LinearAlgebra;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.LearningStrategy
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

        public VSetLearningStrategy(double learningRate, double momentum, float vSetSize)
            : base(learningRate, momentum)
        {
            this.vSetPercentage = vSetSize;
            this.MaxBadIterations = 5;
            IterLimit = 50000;
        }

        public override List<double> Train(INetwork network, IDataSet data)
        {
            vSetEnd = (int)(vSetPercentage * data.PatternCount);
            return base.Train(network, data);
        }

        protected override double RunEpoch()
        {
            for (int i = vSetEnd; i < dataSet.PatternCount; i++) // train set
            {
                Pattern pattern = dataSet.GetPatternAt(i);
                Vector<double> networkAnswer = network.ComputeOutput(pattern.Input);
                Vector<double> modelAnswer = pattern.IdealOutput;

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(LearningRate, Momentum);
            }

            currentError = CalculateMSEValidation();

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
                finished = true; // optimally, network should be restored to a previous state (serialization?)
            }

            previousError = currentError;
            iteration++;
            return CalculateMSEValidation(); // return validation set error
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
