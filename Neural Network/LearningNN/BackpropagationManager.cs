using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN;
using SharpNN.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RecursiveNN;
using MathNet.Numerics.LinearAlgebra;
using LearningNN.Learning;
using LearningNN.DataSet;

namespace LearningNN
{
    public static class BackpropagationManager
    {
        public static LearningResult Run(INetwork network, IDataSet trainData, IDataSet testData, 
            ILearningStrategy learningStrategy, ILearningStatus statusHolder)
        {
            //AssertArguments(trainData, testData, learningRate); // TODO: write assertions

            if (!network.IsInitialized)
            {
                network.Initialize(CreationModes.RandomizeWeights);
            }

            LearningResult learningResult = new LearningResult();
            learningResult.MSEHistory = learningStrategy.Train(network, trainData, statusHolder);
            FillOutput(network, testData);
            return learningResult;
        }

        private static void FillOutput(INetwork network, IDataSet data)
        {
            foreach(Pattern p in data.EnumeratePatterns())
            {
                p.NetworkAnswer = network.ComputeOutput(p.Input);
            }
        }

        private static double CalculateMSEError(INetwork network, IDataSet dataSet)
        {
            double mseSum = 0.0;
            foreach(Pattern p in dataSet.EnumeratePatterns())
            {
                mseSum += MSECalculator.CalculateRawMSE(p.IdealOutput - network.ComputeOutput(p.Input));
            }

            return MSECalculator.CalculateEpochMSE(mseSum, dataSet.PatternCount, network.Activation);
        }
    }
}
