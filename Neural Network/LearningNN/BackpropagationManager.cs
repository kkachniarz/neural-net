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
using MathNet.Numerics.LinearAlgebra;
using LearningNN.Learning;
using LearningNN.DataSet;

namespace LearningNN
{
    public class BackpropagationManager
    {
        private INetwork network;
        private IDataSet trainSet;
        private IDataSet testSet;

        public BackpropagationManager(INetwork network, IDataSet train, IDataSet test)
        {
            this.network = network;
            this.trainSet = train;
            this.testSet = test;
        }

        public LearningResult Run(ILearningStrategy learningStrategy, 
            LearningSettings lSettings, IStatusReporter statusHolder)
        {
            //AssertArguments(trainData, testData, learningRate); // TODO: write assertions

            if (!network.IsInitialized)
            {
                network.Initialize(CreationModes.RandomizeWeights);
            }

            LearningResult learningResult = new LearningResult();
            learningResult.MSEHistory = learningStrategy.Train(network, trainSet, statusHolder);
            learningResult.TestSetError = AnswerTestSet();
            learningResult.DirectionGuessRate = CalculateDirectionGuessed(DirectionGuess.NetworkToNetwork);
            learningResult.DirectionGuessVer2 = CalculateDirectionGuessed(DirectionGuess.IdealToNetwork);
            learningResult.TimeTaken = learningStrategy.TimeTaken;
            learningResult.GotStuck = learningStrategy.GotStuck;
            return learningResult;
        }

        private double AnswerTestSet()
        {
            foreach (Pattern p in testSet.EnumeratePatterns())
            {
                p.NetworkAnswer = network.ComputeOutput(p.Input);
            }

            double mseSum = 0.0;
            foreach (Pattern p in testSet.EnumeratePatterns())
            {
                double vectorMSE = 0.0;
                for(int i = 0; i < testSet.Extremums.OutputExtremums.Count; i++)
                {
                    DataExtremum extr = testSet.Extremums.OutputExtremums[i];
                    double denormAns = extr.Normalizor.NormalizeBack(p.NetworkAnswer[i]);
                    double denormIdeal = extr.Normalizor.NormalizeBack(p.IdealOutput[i]);
                    vectorMSE += (denormAns - denormIdeal) * (denormAns - denormIdeal);
                }

                mseSum += vectorMSE / testSet.Extremums.OutputExtremums.Count;
            }            

//            double min;
 //           double max;
//            Normalizor.GetMinMaxActivationWithMargin(network.Activation.MinValue, network.Activation.MaxValue, out min, out max);
            return MSECalculator.CalculateEpochMSEDenormalized(mseSum, testSet.PatternCount);
        }

        private double CalculateDirectionGuessed(DirectionGuess dirGuessType) // assumes 1 output. 
        {
            if (testSet.PatternCount <= 1)
            {
                throw new ArgumentException("Pattern count of the test set must be greater than 1");
            }

            if (testSet.GetPatternAt(0).IdealOutput.Count > 1)
            {
                throw new ArgumentException("To calculate how often the direction of change is guessed " +
                "correctly, exactly 1 output value is required");
            }

            if (dirGuessType == DirectionGuess.NetworkToNetwork)
            {
                return CalculateDirNetToNet(testSet.EnumeratePatterns().ToList());
            }

            return CalculateDirIdealToNet(testSet.EnumeratePatterns().ToList());
        }

        public static double CalculateDirNetToNet(List<Pattern> patterns)
        {
            return CalculateDirectionImplementation(patterns,
                    (next, prev) => next.NetworkAnswer[0] - prev.NetworkAnswer[0]);
        }

        public static double CalculateDirIdealToNet(List<Pattern> patterns)
        {
            return CalculateDirectionImplementation(patterns,
                (next, prev) => next.NetworkAnswer[0] - prev.IdealOutput[0]);
        }

        private static double CalculateDirectionImplementation(List<Pattern> patterns, 
            Func<Pattern, Pattern, double> diffFunc) 
        {
            double guessCount = 0;
            Pattern prev = patterns[0];
            Pattern next = null;
            for (int i = 1; i < patterns.Count; i++)
            {
                next = patterns[i];
                int idealDir = Math.Sign(next.IdealOutput[0] - prev.IdealOutput[0]);
                int predictedDir = Math.Sign(diffFunc(next, prev));

                if (idealDir == predictedDir)
                {
                    guessCount += 1.0;
                }

                prev = next;
            }

            return guessCount / (double)(patterns.Count - 1);
        }

        private enum DirectionGuess
        {
            NetworkToNetwork, // see if consecutive network predictions have the correct direction of change
            IdealToNetwork // see if the change from last ideal output to next network answer has the right direction.
        }
    }
}
