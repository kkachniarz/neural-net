using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using SharpNN.ActivationFunctions;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using RecursiveNN;

namespace UnitTests
{
    [TestClass]
    public class SaveWeightsTests
    {
        [TestMethod]
        public void TestDoesSaveMLP()
        {
            INetwork nn = new NeuralNetwork(new UnipolarSigmoidActivation(), true, 1, 4, 1);
            TestSaveInternal(nn);
        }

        [TestMethod]
        public void TestDoesSaveElman()
        {
            INetwork nn = new RecursiveNetwork(RecursiveNetwork.Type.Elman, 
                new UnipolarSigmoidActivation(), true, 1, 4, 1);
            TestSaveInternal(nn);
        }

        [TestMethod]
        public void TestDoesSaveJordan()
        {
            INetwork nn = new RecursiveNetwork(RecursiveNetwork.Type.Jordan, 
                new UnipolarSigmoidActivation(), true, 1, 4, 1);
            TestSaveInternal(nn);
        }

        private void TestSaveInternal(INetwork nn)
        {
            nn.Initialize(CreationModes.RandomizeWeights);

            Vector<double> v = new DenseVector(new double[] { 2.0 });   

            object save = nn.SaveWeights();
            Vector<double> original = nn.ComputeOutput(v);

            nn.Initialize(CreationModes.RandomizeWeights);
            Vector<double> after = nn.ComputeOutput(v);

            nn.RestoreWeights(save);
            Vector<double> restored = nn.ComputeOutput(v);
            Assert.AreEqual(original, restored);
            Assert.AreNotEqual(after, restored);
        }
    }
}
