using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;

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
                Assert.AreNotEqual(0.0, MathHelper.RandomExceptZero(0.0001));
            }
        }
    }
}
