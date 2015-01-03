using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Shell;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace UnitTests
{
    [TestClass]
    public class StandardDeviationExtensionTests
    {
        [TestMethod]
        public void BasicStdevTest()
        {
            Vector<double> vec = new DenseVector(new double[] { 1, 2 });
            Assert.AreEqual(Math.Sqrt(2) / 2.0, vec.StandardDeviation(), 0.000001);
        }

        [TestMethod]
        public void SimpleStdevTest()
        {
            Vector<double> vec = new DenseVector(new double[]{1, 4, 5, 5, 6, 7});
            Assert.AreEqual(2.0656, vec.StandardDeviation(), 0.001);
        }
    }
}
