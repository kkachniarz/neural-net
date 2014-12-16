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

namespace LearningNN
{
    public static class BackpropagationManager
    {
        public static LearningResult Run(INetwork network, IDataSet trainData, IDataSet testData, 
            int iterations, double learningRate, double momentum)
        {
            //AssertArguments(trainData, testData, learningRate); // TODO: write assertions
            NormalizeData(network, trainData, testData);

            if (!network.IsInitialized)
            {
                network.Initialize(CreationModes.RandomizeWeights);
            }

            LearningResult learningResult = new LearningResult();

            for (int currIter = 0; currIter < iterations; currIter++)
            {
                learningResult.MSEHistory.Add(LearnAge(network, trainData, learningRate, momentum));
            }

            FillOutput(network, testData);

            NormalizeDataBack(network, trainData, testData);
            return learningResult;
        }

        private static double LearnAge(INetwork network, IDataSet data,
            double learningRate, double momentum)
        {
            foreach(Pattern pattern in data.EnumeratePatterns())
            {
                Vector<double> networkAnswer = network.ComputeOutput(pattern.Input);
                Vector<double> modelAnswer = pattern.IdealOutput;

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(learningRate, momentum);
            }

            return CalculateMSEError(network, data);
        }

        private static void FillOutput(INetwork network, IDataSet data)
        {
            foreach(Pattern p in data.EnumeratePatterns())
            {
                p.NetworkAnswer = network.ComputeOutput(p.Input);
            }
        }

        private static void AssertArguments(CasesData trainData, CasesData testData, double learningRate)
        {
            if (trainData.ProblemType != testData.ProblemType)
            {
                throw new ArgumentException("Training data and test data should represent the same problem type.");
            }

            if (!trainData.HasOutput)
            {
                throw new ArgumentException("Output is obligatory in train data.");
            }

            if (learningRate <= 0)
            {
                throw new ArgumentException("leariningRate shoule be possitive.");
            };
        }

        private static void NormalizeData(INetwork network, IDataSet trainData, IDataSet testData)
        {
            double minValue = Math.Min(trainData.MinValue, testData.MinValue);
            double maxValue = Math.Max(trainData.MaxValue, testData.MaxValue);

            trainData.Normalize(minValue, maxValue, network.Activation.MinValue, network.Activation.MaxValue);
            testData.Normalize(minValue, maxValue, network.Activation.MinValue, network.Activation.MaxValue);
        }

        private static void NormalizeDataBack(INetwork network, IDataSet trainData, IDataSet testData)
        {
            testData.NormalizeBack();
            trainData.NormalizeBack();
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

        private static double CalculateClassificationError(INetwork network, CasesData trainingSet)
        {
            double correctness = 0.0;
            for (int i = 0; i < trainingSet.CasesCount; i++)
            {
                if (network.ComputeOutput(trainingSet.GetInput(i)).MaximumIndex() == trainingSet.GetIdealOutput(i).MaximumIndex())
                {
                    correctness += 1.0;
                }
            }

            return 1.0 - (correctness / (double)trainingSet.CasesCount);
        }
    }
}
