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
            ILearningStrategy learningStrategy, IStatusReporter statusHolder)
        {
            //AssertArguments(trainData, testData, learningRate); // TODO: write assertions

            if (!network.IsInitialized)
            {
                network.Initialize(CreationModes.RandomizeWeights);
            }

            LearningResult learningResult = new LearningResult();
            learningResult.MSEHistory = learningStrategy.Train(network, trainData, statusHolder);
            learningResult.TestSetError = AnswerTestSet(network, testData);
            learningResult.DirectionGuessRate = CalculateDirectionGuessed(testData);
            learningResult.TimeTaken = learningStrategy.TimeTaken;
            learningResult.GotStuck = learningStrategy.GotStuck;
            return learningResult;
        }

        private static double AnswerTestSet(INetwork network, IDataSet testDataSet)
        {
            foreach (Pattern p in testDataSet.EnumeratePatterns())
            {
                p.NetworkAnswer = network.ComputeOutput(p.Input);
            }

            double mseSum = 0.0;
            foreach (Pattern p in testDataSet.EnumeratePatterns())
            {
                mseSum += MSECalculator.CalculateRawMSE(p.IdealOutput - p.NetworkAnswer);
            }

            return MSECalculator.CalculateEpochMSE(mseSum, testDataSet.PatternCount, network.Activation);
        }

        private static double CalculateDirectionGuessed(IDataSet testDataSet) // assumes 1 output. 
        {
            if (testDataSet.PatternCount <= 1)
            {
                throw new ArgumentException("Pattern count of the test set must be greater than 1");
            }

            if (testDataSet.GetPatternAt(0).IdealOutput.Count > 1)
            {
                throw new ArgumentException("To calculate how often the direction of change is guessed " +
                "correctly, exactly 1 output value is required");
            }

            return CalculateDirectionImplementation(testDataSet.EnumeratePatterns().ToList());
        }

        public static double CalculateDirectionImplementation(List<Pattern> patterns) // Public for unit tests
        {
            double guessCount = 0;
            Pattern prev = patterns[0];
            Pattern next = null;
            for (int i = 1; i < patterns.Count; i++)
            {
                next = patterns[i];
                int idealDir = Math.Sign(next.IdealOutput[0] - prev.IdealOutput[0]);
                int predictedDir = Math.Sign(next.NetworkAnswer[0] - prev.NetworkAnswer[0]);
                if (idealDir == predictedDir)
                {
                    guessCount += 1.0;
                }

                prev = next;
            }

            return guessCount / (double)(patterns.Count - 1);
        }
    }
}
