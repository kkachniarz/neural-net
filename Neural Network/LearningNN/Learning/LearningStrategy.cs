using LearningNN.DataSet;
using MathNet.Numerics.LinearAlgebra;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.Learning
{
    public abstract class LearningStrategy : ILearningStrategy
    {
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        /// <summary>
        /// Total time taken by the last training.
        /// </summary>
        public TimeSpan TimeTaken { get; protected set; }
        public bool GotStuck { get; protected set; }

        protected bool finished;
        protected INetwork network;
        protected IDataSet dataSet;
        protected IStatusReporter statusHolder;
        protected List<double> errorHistory;

        private DateTime lastStatusUpdate = DateTime.MinValue;
        private TimeSpan minUpdateInterval = new TimeSpan(0, 0, 1);

        public LearningStrategy(double learningRate, double momentum)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
            this.TimeTaken = TimeSpan.Zero;
            this.GotStuck = false;
        }

        public virtual List<double> Train(INetwork network, IDataSet data, IStatusReporter statusHolder)
        {
            this.network = network;
            this.dataSet = data;
            this.statusHolder = statusHolder;

            errorHistory = new List<double>();
            DateTime startTime = DateTime.Now;
            while(!finished)
            {
                errorHistory.Add(RunEpoch());
                if (DateTime.Now - lastStatusUpdate > minUpdateInterval)
                {
                    UpdateStatus();
                    lastStatusUpdate = DateTime.Now;
                }
            }

            TimeTaken = DateTime.Now - startTime;
            return errorHistory;
        }

        protected abstract void UpdateStatus();

        protected abstract double RunEpoch();

        protected virtual double CalculateMSEError(INetwork network, IDataSet dataSet)
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
