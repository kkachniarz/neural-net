using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN.ActivationFunctions;
using LearningNN;

namespace UnitTests
{
    [TestClass]
    public class MSETests
    {
        private IActivation Unipolar;
        private IActivation Bipolar;

        public MSETests()
        {
            Unipolar = new UnipolarSigmoidActivation();
            Bipolar = new BipolarTanhActivation();
        }

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }
        [TestMethod]
        public void TestEpochMSEFor3Elements()
        {
            Vector<double> error1 = new DenseVector(new double[] { 0.5, 0.0, 0.0 });
            Vector<double> error2 = new DenseVector(new double[] { -1.0, -1.0, 1.0 });
            Vector<double> error3 = new DenseVector(new double[] { 1.0, 1.0, 0.0 });
            double mse1 = MSECalculator.CalculateRawAverageMSE(error1);
            double mse2 = MSECalculator.CalculateRawAverageMSE(error2);
            double mse3 = MSECalculator.CalculateRawAverageMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, Unipolar.MinValue, Unipolar.MaxValue);
            Assert.AreEqual(5.25 / 9.0, epochMSE, 0.00001);
        }

        [TestMethod]
        public void TestEpochCompletelyWrongEquals1Bipolar()
        {
            Vector<double> error1 = new DenseVector(new double[] { 2.0, 2.0, 2.0 });
            Vector<double> error2 = new DenseVector(new double[] { -2.0, -2.0, 2.0 });
            Vector<double> error3 = new DenseVector(new double[] { 2.0, -2.0, 2.0 });
            double mse1 = MSECalculator.CalculateRawAverageMSE(error1);
            double mse2 = MSECalculator.CalculateRawAverageMSE(error2);
            double mse3 = MSECalculator.CalculateRawAverageMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, Bipolar.MinValue, Bipolar.MaxValue);
            Assert.AreEqual(1.0, epochMSE, 0.00001);
        }

        [TestMethod]
        public void TestEpochCompletelyWrongEquals1Unipolar()
        {
            Vector<double> error1 = new DenseVector(new double[] { 1.0, 1.0, 1.0 });
            Vector<double> error2 = new DenseVector(new double[] { -1.0, -1.0, 1.0 });
            Vector<double> error3 = new DenseVector(new double[] { 1.0, -1.0, 1.0 });
            double mse1 = MSECalculator.CalculateRawAverageMSE(error1);
            double mse2 = MSECalculator.CalculateRawAverageMSE(error2);
            double mse3 = MSECalculator.CalculateRawAverageMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, Unipolar.MinValue, Unipolar.MaxValue);
            Assert.AreEqual(1.0, epochMSE, 0.00001);
        }

        [TestMethod]
        public void TestEpochCompletelyWrongEquals1UnipolarWithMargin()
        {
            Vector<double> error1 = new DenseVector(new double[] { 0.6, 0.6, 0.6 });
            Vector<double> error2 = new DenseVector(new double[] { -0.6, -0.6, 0.6 });
            Vector<double> error3 = new DenseVector(new double[] { 0.6, -0.6, 0.6});
            double mse1 = MSECalculator.CalculateRawAverageMSE(error1);
            double mse2 = MSECalculator.CalculateRawAverageMSE(error2);
            double mse3 = MSECalculator.CalculateRawAverageMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, 0.2, 0.8);
            Assert.AreEqual(1.0, epochMSE, 0.00001);
        }

        /// <summary>
        /// If the data set is normalized to outputs in range [0.2, 0.8], 
        /// then the error can be up to 1.0 - 0.2 = 0.8 (0.0 - 0.8 = -0.8)
        /// We can call this an "overflow" - the network output is outside the range of outputs allowed for the data,
        /// so in such cases the normalized MSE can exceed 1. This is never possible if the margin is 0.
        /// This phenomenon, however, allows for a more fair comparison of network performance across different margin values.
        /// </summary>
        [TestMethod]
        public void TestEpochWrongOverflowUnipolarWithMargin()
        {
            Vector<double> error1 = new DenseVector(new double[] { 0.8, 0.6, 0.6 });
            Vector<double> error2 = new DenseVector(new double[] { -0.6, -0.75, 0.6 });
            Vector<double> error3 = new DenseVector(new double[] { 0.6, -0.6, 0.6 });
            double mse1 = MSECalculator.CalculateRawAverageMSE(error1);
            double mse2 = MSECalculator.CalculateRawAverageMSE(error2);
            double mse3 = MSECalculator.CalculateRawAverageMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, 0.2, 0.8);
            Assert.IsTrue(epochMSE > 1.0);
        }

        [TestMethod]
        public void TestNormalizorMinMaxGetter()
        {
            double min;
            double max;
            double marginFactor = Normalizor.MARGIN_FACTOR;
            Normalizor.GetMinMaxActivationWithMargin(Unipolar.MinValue, Unipolar.MaxValue, out min, out max);

            double span = (Unipolar.MaxValue - Unipolar.MinValue);
            double desiredMin = Unipolar.MinValue + marginFactor * span;
            double desiredMax = Unipolar.MaxValue - marginFactor * span;

            Assert.AreEqual(desiredMin, min);
            Assert.AreEqual(desiredMax, max);
        }
    }
}
