using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN.ActivationFunctions;
using SharpNN;
using System.Collections.Generic;

namespace UnitTests
{
    [TestClass]
    public class ActivationFunctionsTests
    {
        private const double EPSILON = 0.00000001;
        private List<Tuple<double, double>> unipolarSigmoidTestCases = new List<Tuple<double, double>>()
        {
            new Tuple<double, double>(-100.0, 0.0),
            new Tuple<double, double>(-50.0, 0.0),
            new Tuple<double, double>(-10.0, 0.0000453978),
            new Tuple<double, double>(-2.0, 0.1192029220),
            new Tuple<double, double>(0.0, 0.5),
            new Tuple<double, double>(2.0, 0.8807970779),
            new Tuple<double, double>(10.0, 0.9999546021),
            new Tuple<double, double>(50.0, 1),
            new Tuple<double, double>(100.0, 1),
        };

        private List<Tuple<double, double>> bipolarTanhTestCases = new List<Tuple<double, double>>()
        {
            new Tuple<double, double>(-100.0, -1.0),
            new Tuple<double, double>(-50.0, -1.0),
            new Tuple<double, double>(-10.0, -0.999999995877),
            new Tuple<double, double>(-2.0, -0.964027580075),
            new Tuple<double, double>(0.0, 0.0),
            new Tuple<double, double>(2.0, 0.964027580075),
            new Tuple<double, double>(10.0, 0.999999995877),
            new Tuple<double, double>(50.0, 1.0),
            new Tuple<double, double>(100.0, 1.0),
        };

        private List<Tuple<double, double>> linearTestCases = new List<Tuple<double, double>>()
        {
            new Tuple<double, double>(-100.0, -100.0),
            new Tuple<double, double>(-2.0, -2.0),
            new Tuple<double, double>(0.0, 0.0),
            new Tuple<double, double>(2.0, 2.0),
            new Tuple<double, double>(100.0, 100.0),
        };

        private List<Tuple<double, double>> stepTestCases = new List<Tuple<double, double>>()
        {
            new Tuple<double, double>(-100.0, 0),
            new Tuple<double, double>(-2.0, 0),
            new Tuple<double, double>(0.0, 1.0),
            new Tuple<double, double>(2.0, 1.0),
            new Tuple<double, double>(100.0, 1.0),
        };

        [TestMethod]
        public void TestUnipolarSigmoid()
        {
            IActivation sigmoid = new UnipolarSigmoidActivation();
            foreach(Tuple<double, double> tup in unipolarSigmoidTestCases)
            {
                Assert.AreEqual(tup.Item2, sigmoid.Calc(tup.Item1), EPSILON);
            }
        }

        [TestMethod]
        public void TestBipolarTanh()
        {
            IActivation bipolarTanh = new BipolarTanhActivation();
            foreach (Tuple<double, double> tup in bipolarTanhTestCases)
            {
                Assert.AreEqual(tup.Item2, bipolarTanh.Calc(tup.Item1), EPSILON);
            }
        }

        [TestMethod]
        public void TestLinear()
        {
            IActivation linear = new LinearActivation();
            foreach (Tuple<double, double> tup in linearTestCases)
            {
                Assert.AreEqual(tup.Item2, linear.Calc(tup.Item1), EPSILON);
            }
        }

        [TestMethod]
        public void TestStep()
        {
            IActivation step = new StepActivation();
            foreach (Tuple<double, double> tup in stepTestCases)
            {
                Assert.AreEqual(tup.Item2, step.Calc(tup.Item1), EPSILON);
            }
        }
    }
}
