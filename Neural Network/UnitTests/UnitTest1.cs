using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using SharpNN.ActivationFunctions;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace UnitTests
{
    [TestClass]
    public class SaveWeightsTests
    {
        [TestMethod]
        public void TestDoesSaveMLP()
        {
            NeuralNetwork nn = new NeuralNetwork(new UnipolarSigmoidActivation(), true, 1, 4, 1);
            nn.Initialize(CreationModes.RandomizeWeights);
            nn.SetIncomingWeightsForNeuron(1, 0, new double[] { 2.0 });
            Vector<double> v = new DenseVector(new double[] { 2.0 });   

            object save = nn.SaveWeights();
            Vector<double> original = nn.ComputeOutput(v);

            nn.Initialize(CreationModes.RandomizeWeights);
            nn.SetIncomingWeightsForNeuron(1, 0, new double[] { 1.0 });
            Vector<double> after = nn.ComputeOutput(v);

            nn.RestoreWeights(save);
            Vector<double> restored = nn.ComputeOutput(v);
            Assert.AreEqual(original, restored);
            Assert.AreNotEqual(after, restored);
        }
    }
}
