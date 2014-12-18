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
    public abstract class LearningStrategy : ILearningStrategy
    {
        public double LearningRate { get; set; }
        public double Momentum { get; set; }

        protected bool finished;
        protected INetwork network;
        protected IDataSet dataSet;

        public LearningStrategy(double learningRate, double momentum)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
        }

        public virtual List<double> Train(INetwork network, IDataSet data)
        {
            this.network = network;
            this.dataSet = data;
            List<double> errorHistory = new List<double>();
            while(!finished)
            {
                errorHistory.Add(RunEpoch());
            }

            return errorHistory;
        }

        protected abstract double RunEpoch();

        protected static double CalculateMSEError(INetwork network, IDataSet dataSet)
        {
            double mseSum = 0.0;
            foreach (Pattern p in dataSet.EnumeratePatterns())
            {
                mseSum += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mseSum, dataSet.PatternCount, network.Activation);
        }
    }
}
