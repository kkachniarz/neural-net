using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using LearningNN;
using LearningNN.DataSet;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;

namespace UnitTests
{
    [TestClass]
    public class DirectionErrorTests
    {
        [TestMethod]
        public void TestDirectionGuessedPerfectly()
        {
            List<Pattern> patterns = new List<Pattern>();
            for (int i = 0; i < 20; i++)
            {
                patterns.Add(new Pattern(i));
                patterns[i].IdealOutput = new DenseVector(new double[] { (double)i });
                patterns[i].NetworkAnswer = new DenseVector(new double[] { (double)i+20 }); //(i - 10) + (i % 5) * 100 
            }

            double directionGuessed = BackpropagationManager.CalculateDirectionImplementation(patterns);
            Assert.AreEqual(1.0, directionGuessed, 0.0001);
        }

        [TestMethod]
        public void TestDirectionGuessed80Percent()
        {
            List<Pattern> patterns = new List<Pattern>();
            for (int i = 0; i < 101; i++)
            {
                patterns.Add(new Pattern(i));
                patterns[i].IdealOutput = new DenseVector(new double[] { (double)i });
                patterns[i].NetworkAnswer = new DenseVector(new double[] { (double)(i - 10) + (i % 5) * 100});  
            }

            double directionGuessed = BackpropagationManager.CalculateDirectionImplementation(patterns);
            Assert.AreEqual(0.8, directionGuessed, 0.0001);
        }
    }
}
