﻿using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN;
using SharpNN.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RecursiveNN;

namespace LearningNN
{
    public static class BackpropagationManager
    {
        public static LearningResult Run(INetwork network, CasesData trainData, CasesData testData, 
            BackpropagationRunMode runMode, int iterations, double learningRate, double momentum)
        {
            AssertArguments(trainData, testData, learningRate);
            NormalizeData(network, trainData, testData);

            if (!network.IsInitialized)
            {
                network.Initialize(CreationModes.RandomizeWeights);
            }

            LearningResult learningResult = new LearningResult();

            for (int currIter = 0; currIter < iterations; currIter++)
            {
                learningResult.MSEHistory.Add(
                    LearnAge(network, trainData, runMode, learningRate, momentum));
            }

            FillOutput(network, testData);

            NormalizeDataBack(network, trainData, testData);
            return learningResult;
        }

        private static double LearnAge(INetwork network, CasesData data,
            BackpropagationRunMode runMode, double learningRate, double momentum)
        {
            if (runMode == BackpropagationRunMode.Stochastic)
            {
                data.Permutate(); 
            }

            for (int i = 0; i < data.CasesCount; i++)
            {
                var networkAnswer = network.ComputeOutput(data.GetInput(i));
                var modelAnswer = data.GetIdealOutput(i);

                network.CalculateAndPropagateError(modelAnswer);
                network.ImproveWeights(learningRate, momentum);
            }

            if (data.ProblemType == ProblemType.Classification)
            {
                return CalculateClassificationError(network, data);
            }

            return CalculateMSEError(network, data);
        }

        private static void FillOutput(INetwork network, CasesData data)
        {
            for (int i = 0; i < data.CasesCount; i++)
            {
                data.SaveNetworkAnswer(i, network.ComputeOutput(data.GetInput(i)));
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
        private static void NormalizeData(INetwork network, CasesData trainData, CasesData testData)
        {
            double minValue = Math.Min(trainData.MinValue, testData.MinValue);
            double maxValue = Math.Max(trainData.MaxValue, testData.MaxValue);

            trainData.Normalize(maxValue, minValue, network.Activation.MaxValue, network.Activation.MinValue);
            testData.Normalize(maxValue, minValue, network.Activation.MaxValue, network.Activation.MinValue);
        }

        private static void NormalizeDataBack(INetwork network, CasesData trainData, CasesData testData)
        {
            testData.NormalizeBack(true);
            trainData.NormalizeBack(false);
        }

        private static double CalculateMSEError(INetwork network, CasesData trainingSet)
        {
            double mseSum = 0.0;
            for(int i=0; i<trainingSet.CasesCount; i++)
            {
                mseSum += MSECalculator.CalculateRawMSE(trainingSet.GetIdealOutput(i) - network.ComputeOutput(trainingSet.GetInput(i)));
            }

            return MSECalculator.CalculateEpochMSE(mseSum, trainingSet.CasesCount, network.Activation);
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

    public enum BackpropagationRunMode
    {
        Online,
        Stochastic
    }

    public enum PartIIProblemType
    {
        CTS,
        Stock
    }

    public enum ProblemType
    {
        Classification,
        Regression
    }
}
