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
            Random r = new Random();
            nn.Initialize(CreationModes.RandomizeWeights);

            Vector<double> v = new DenseVector(new double[] { 1.0 });
            for(int i = 0; i < 10; i++)
            {
                nn.ComputeOutput(new DenseVector(new double[] {r.NextDouble()}));
            }

            object save = nn.SaveWeights(); // to make it easy for recurrent nets (memory would change after computation - save would be imperfect)
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
