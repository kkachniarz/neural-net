using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN.Layers;
using SharpNN.ActivationFunctions;
using MathNet.Numerics.LinearAlgebra;

namespace UnitTests
{
    [TestClass]
    public class LayerTests
    {
        [TestMethod]
        public void TestRandomizeLeavesNoZerosInConnections()
        {
            InputLayer il = new InputLayer(3);
            WeightedLayer wl = new WeightedLayer(5, new UnipolarSigmoidActivation(), false);
            il.ConnectTo(wl);
            wl.RandomizeWeights(0.0001);
            Assert.IsTrue(wl.IncomingWeights.Enumerate().All(x => x != 0.0));
        }

        [TestMethod]
        public void TestRandomizeLeavesNoZerosInBias()
        {
            InputLayer il = new InputLayer(3);
            WeightedLayer wl = new WeightedLayer(5, new UnipolarSigmoidActivation(), true);
            il.ConnectTo(wl);
            wl.RandomizeWeights(0.05);
            Assert.IsTrue(wl.BiasWeights.Enumerate().All(x => x != 0.0));
        }

        [TestMethod]
        public void TestPreviousLayerReferenceIsCorrect()
        {
            InputLayer il = new InputLayer(3);
            WeightedLayer wl = new WeightedLayer(5, new UnipolarSigmoidActivation(), true);
            il.ConnectTo(wl);
            Assert.AreEqual(il.NeuronCount, wl.Previous.NeuronCount);
        }

        [TestMethod]
        public void TestNextLayerReferenceIsCorrect()
        {
            InputLayer il = new InputLayer(3);
            WeightedLayer wl = new WeightedLayer(5, new UnipolarSigmoidActivation(), true);
            il.ConnectTo(wl);
            Assert.AreEqual(wl.NeuronCount, il.Next.NeuronCount);
        }

        [TestMethod]
        public void TestLastOutputInNetworkContext()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation(), false);
            nn.AddLayer(3, new LinearActivation(), false);
            nn.AddLayer(1, new LinearActivation(), false);
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { 
                new double[] {1, -2 }, new double[] { 4, 0 }, new double[] { -1, 3 } });
            nn.SetIncomingWeightsForLayer(2, new List<double[]>() { new double[] { -1, -2, 3 } });
            Vector<double> result = nn.ComputeOutput(new double[] { 1.0, 2.0 });
            double[] lastOutput = nn.LayerAt(1).LastOutput.ToArray();
            // TODO: write assert
        }
    }
}
