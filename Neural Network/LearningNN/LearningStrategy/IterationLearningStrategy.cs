using MathNet.Numerics.LinearAlgebra;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.LearningStrategy
{
    public class IterationLearningStrategy : LearningStrategy
    {
        private int maxIterations;
        private int currentIteration = 0;

        public IterationLearningStrategy(double learningRate, double momentum, int maxIterations)
            : base(learningRate, momentum)
        {
            this.maxIterations = maxIterations;
        }

        protected override double RunEpoch()
        {
            foreach (Pattern pattern in dataSet.EnumeratePatterns())
            {
                Vector<double> networkAnswer = network.ComputeOutput(pattern.Input);
                Vector<double> modelAnswer = pattern.IdealOutput;

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(LearningRate, Momentum);
            }

            if(currentIteration++ >= maxIterations)
            {
                finished = true;
            }

            return CalculateMSEError(network, dataSet);
        }

        protected override void UpdateStatus()
        {
            statusHolder.SetText(string.Format("iter: {0}, error: {1}", currentIteration, errorHistory[errorHistory.Count - 1]));
        }
    }
}
