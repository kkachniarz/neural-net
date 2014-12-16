using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN.ActivationFunctions;

namespace UnitTests
{
    [TestClass]
    public class MSETests
    {
        public MSETests()
        {
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
            double mse1 = MSECalculator.CalculateRawMSE(error1);
            double mse2 = MSECalculator.CalculateRawMSE(error2);
            double mse3 = MSECalculator.CalculateRawMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, new UnipolarSigmoidActivation());
            Assert.AreEqual(5.25 / 9.0, epochMSE, 0.00001);
        }

        [TestMethod]
        public void TestEpochCompletelyWrongEquals1Bipolar()
        {
            Vector<double> error1 = new DenseVector(new double[] { 2.0, 2.0, 2.0 });
            Vector<double> error2 = new DenseVector(new double[] { -2.0, -2.0, 2.0 });
            Vector<double> error3 = new DenseVector(new double[] { 2.0, -2.0, 2.0 });
            double mse1 = MSECalculator.CalculateRawMSE(error1);
            double mse2 = MSECalculator.CalculateRawMSE(error2);
            double mse3 = MSECalculator.CalculateRawMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, new BipolarTanhActivation());
            Assert.AreEqual(1.0, epochMSE, 0.00001);
        }

        [TestMethod]
        public void TestEpochCompletelyWrongEquals1Unipolar()
        {
            Vector<double> error1 = new DenseVector(new double[] { 1.0, 1.0, 1.0 });
            Vector<double> error2 = new DenseVector(new double[] { -1.0, -1.0, 1.0 });
            Vector<double> error3 = new DenseVector(new double[] { 1.0, -1.0, 1.0 });
            double mse1 = MSECalculator.CalculateRawMSE(error1);
            double mse2 = MSECalculator.CalculateRawMSE(error2);
            double mse3 = MSECalculator.CalculateRawMSE(error3);

            double rawMSESum = mse1 + mse2 + mse3;
            double epochMSE = MSECalculator.CalculateEpochMSE(rawMSESum, 3.0, new UnipolarSigmoidActivation());
            Assert.AreEqual(1.0, epochMSE, 0.00001);
        }
    }
}
