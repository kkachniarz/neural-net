using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace UnitTests
{
    [TestClass]
    public class MathHelperTests
    {
        [TestMethod]
        public void TestRandomizeGivesNonZeroResults()
        {
            for(int i=0; i<10000; i++)
            {
                Assert.AreNotEqual(0.0, MathHelper.RandomExceptZero(0.0001, 0.2));
            }
        }

        [TestMethod]
        public void TestRandomizeSeemsToHaveMean0()
        {
            int count = 100000;
            Vector<double> vec = new DenseVector(count);
            for (int i = 0; i < count; i++)
            {
                vec[i] = MathHelper.RandomExceptZero(1, 0.2);
            }

            Assert.AreEqual(0, vec.Average(), 0.002);
        }

        [TestMethod]
        public void TestRandomizeEpsilonIsApplied()
        {
            int count = 1000;
            double epsFactor = 0.2;
            double maxAbs = 1.0;
            Vector<double> vec = new DenseVector(count);
            for (int i = 0; i < count; i++)
            {
                vec[i] = MathHelper.RandomExceptZero(maxAbs, epsFactor);
            }

            double min = vec.AbsoluteMinimum();
            double max = vec.AbsoluteMaximum();
            Assert.IsTrue(min >= epsFactor * maxAbs);
            Assert.IsTrue(max <= maxAbs);
        }
    }
}
