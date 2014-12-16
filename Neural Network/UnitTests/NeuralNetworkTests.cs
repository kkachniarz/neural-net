using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using SharpNN.ActivationFunctions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using SharpNN.Layers;

namespace UnitTests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        private const double EPSILON = 0.00000001;
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
        public void TestNeuralNetworkIsCreated()
        {
            NeuralNetwork nn = new NeuralNetwork();
            Assert.IsNotNull(nn);
        }

        [TestMethod]
        public void TestLayerIsCreated()
        {
            Layer inputLayer = new InputLayer(2);
            Assert.IsNotNull(inputLayer);
        }

        [TestMethod]
        public void TestNeuralNetworkIsCreatedWith2Layers()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            nn.Initialize(CreationModes.RandomizeWeights);
            Assert.AreEqual(2, nn.LayerCount);
        }

        [TestMethod]
        public void TestNeuralNetworkIsCreatedWith3Layers()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation());
            nn.AddLayer(4, new UnipolarSigmoidActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            nn.Initialize(CreationModes.RandomizeWeights);
            Assert.AreEqual(3, nn.LayerCount);
        }

        [TestMethod]
        public void TestNeuralNetworkAttributes5Layers()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation());
            nn.AddLayer(4, new UnipolarSigmoidActivation());
            nn.AddLayer(7, new UnipolarSigmoidActivation());
            nn.AddLayer(4, new UnipolarSigmoidActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            nn.Initialize(CreationModes.RandomizeWeights);
            Assert.AreEqual(5, nn.LayerCount);
            Assert.AreEqual(3, nn.HiddenLayerCount);
            Assert.AreEqual(2, nn.InputCount);
            Assert.AreEqual(1, nn.OutputCount);
            Assert.AreEqual(4, nn.LayerAt(3).NeuronCount);
        }

        [TestMethod]
        public void TestNeuralNetworkAttributes5LayersRichConstructor()
        {
            NeuralNetwork nn = new NeuralNetwork(new UnipolarSigmoidActivation(), false, 2, 4, 7, 4, 1);
            nn.Initialize(CreationModes.RandomizeWeights);
            Assert.AreEqual(5, nn.LayerCount);
            Assert.AreEqual(3, nn.HiddenLayerCount);
            Assert.AreEqual(2, nn.InputCount);
            Assert.AreEqual(1, nn.OutputCount);
            Assert.AreEqual(4, nn.LayerAt(3).NeuronCount);
        }

        [TestMethod]
        public void TestNeuralNetworkAttributes5LayersRichConstructorArrayInitialize()
        {
            int[] layerSizes = new int[] { 2, 4, 7, 4, 1 };
            NeuralNetwork nn = new NeuralNetwork(new UnipolarSigmoidActivation(), false, layerSizes);
            nn.Initialize(CreationModes.RandomizeWeights);
            Assert.AreEqual(5, nn.LayerCount);
            Assert.AreEqual(3, nn.HiddenLayerCount);
            Assert.AreEqual(2, nn.InputCount);
            Assert.AreEqual(1, nn.OutputCount);
            Assert.AreEqual(4, nn.LayerAt(3).NeuronCount);
        }

        [TestMethod]
        public void TestSimpleProcessing1()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { new double[] { 1 } });
            Vector<double> result = nn.ComputeOutput(new double[] { 2 });
            TestContext.WriteLine(result[0].ToString());
            Assert.AreEqual(2, result[0]);
        }

        [TestMethod]
        public void TestSimpleProcessing2()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { new double[] { 2 } });
            Vector<double> result = nn.ComputeOutput(new double[] { 4 });
            TestContext.WriteLine(result[0].ToString());
            Assert.AreEqual(8, result[0]);
        }

        [TestMethod]
        public void TestSimpleProcessing3()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { new double[] { 2 } });
            Vector<double> result = nn.ComputeOutput(new double[] { 0 });
            TestContext.WriteLine(result[0].ToString());
            Assert.AreEqual(0.5, result[0]);
        }

        [TestMethod]
        public void TestSeveralNeuronsPerceptronProcessing()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(3, new LinearActivation());
            nn.AddLayer(2, new LinearActivation());
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() {
                new double[]{ 1, 0, 2 }, new double[]{0, 4, 1}});
            Vector<double> result = nn.ComputeOutput(new double[] { 1, 2, 3 });
            Assert.AreEqual(7.0, result[0], EPSILON);
            Assert.AreEqual(11.0, result[1], EPSILON);
        }

        [TestMethod]
        public void TestSeveralNeuronsPerceptronProcessingIndividualWeightModifications()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(3, new LinearActivation());
            nn.AddLayer(2, new LinearActivation());
            nn.Initialize(CreationModes.RandomizeWeights);
            nn.SetIncomingWeightsForNeuron(1, 0, new double[] { 1, 0, 2 });
            nn.SetIncomingWeightsForNeuron(1, 1, new double[] { 0, 4, 1 });
            Vector<double> result = nn.ComputeOutput(new double[] { 1, 2, 3 });
            Assert.AreEqual(7.0, result[0], EPSILON);
            Assert.AreEqual(11.0, result[1], EPSILON);
        }

        [TestMethod]
        public void TestSimpleMLPProcessing1()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { new double[] { 2 } });
            nn.SetIncomingWeightsForLayer(2, new List<double[]>() { new double[] { 2 } });
            Vector<double> result = nn.ComputeOutput(new double[] { 3 });
            Assert.AreEqual(12, result[0]);
        }

        [TestMethod]
        public void TestSimpleMLPProcessingWithBiasLinear()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation(), false);
            nn.AddLayer(1, new LinearActivation(), true);
            nn.AddLayer(1, new LinearActivation(), true);
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { new double[] { 2 } });
            nn.SetIncomingWeightsForLayer(2, new List<double[]>() { new double[] { 2 } });
            nn.SetBiasForLayer(1, new double[] { 10 });
            nn.SetBiasForLayer(2, new double[] { -3 });
            Vector<double> result = nn.ComputeOutput(new double[] { 1 });
            Assert.AreEqual(21, result[0]);
        }

        [TestMethod]
        public void TestComplexMLPProcessingWithBiasBipolarSigmoid1()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation(), false);
            nn.AddLayer(3, new BipolarTanhActivation(), true);
            nn.AddLayer(1, new BipolarTanhActivation(), true);
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() 
            { new double[] { 2, 1 }, new double[] {3, -2}, new double[]{-2, 7}});
            nn.SetIncomingWeightsForLayer(2, new List<double[]>() 
            { new double[] { 17, 13, 20 } });
            nn.SetBiasForLayer(1, new double[] { -100.0, -100.0, -100.0 });
            nn.SetBiasForLayer(2, new double[] { 50.0 });
            Vector<double> result = nn.ComputeOutput(new double[] { 0.8, 0.5 });
            Assert.AreEqual(0.0, result[0], 0.0001);
        }

        [TestMethod]
        public void TestComplexMLPProcessingWithBiasBipolarSigmoidRichConstructor()
        {
            NeuralNetwork nn = new NeuralNetwork(new BipolarTanhActivation(), true, 2, 3, 1);
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForLayer(1, new List<double[]>() { 
                new double[] { 2, 1 }, new double[] { 3, -2 }, new double[] { -2, 7 } });
            nn.SetIncomingWeightsForLayer(2, new List<double[]>() { new double[] { 17, 13, 20 } });
            nn.SetBiasForLayer(1, new double[] { -100.0, -100.0, -100.0 });
            nn.SetBiasForLayer(2, new double[] { 50.0 });
            Vector<double> result = nn.ComputeOutput(new double[] { 0.8, 0.5 });
            Assert.AreEqual(0.0, result[0], 0.0001);
        }

        //[TestMethod]
        //public void TestComplexMLPProcessingWithBiasBipolarSigmoid2()
        //{
        //    NeuralNetwork nn = new NeuralNetwork();
        //    nn.AddLayer(2, new LinearActivation(), false);
        //    nn.AddLayer(3, new BipolarSigmoidActivation(), true);
        //    nn.AddLayer(1, new BipolarSigmoidActivation(), true);
        //    nn.Initialize(CreationModes.NoAction);
        //    nn.SetIncomingWeightsForLayer(1, new List<double[]>() { 
        //        new double[] { 2, 1 }, new double[] { 3, -2 }, new double[] { -2, 7 } });
        //    nn.SetIncomingWeightsForLayer(2, new List<double[]>() { new double[] { 17, 10, 20 } });
        //    nn.SetBiasForLayer(1, new double[] { -100.0, -100.0, -100.0 });
        //    nn.SetBiasForLayer(2, new double[] { 50.0 });
        //    Vector<double> result = nn.ComputeOutput(new double[] { 0.8, 0.5 });
        //    Assert.AreEqual(0.905148, result[0], 0.0001);
        //}

        [TestMethod]
        public void TestSettingLayerWeights()
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
            Assert.AreEqual(10.0, result[0], 0.0001);
        }

        [TestMethod]
        public void TestSettingNeuronWeights()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(2, new LinearActivation(), false);
            nn.AddLayer(3, new LinearActivation(), false);
            nn.AddLayer(1, new LinearActivation(), false);
            nn.Initialize(CreationModes.NoAction);
            nn.SetIncomingWeightsForNeuron(1, 0, new double[] {1, -2 });
            nn.SetIncomingWeightsForNeuron(1, 1, new double[] { 4, 0 });
            nn.SetIncomingWeightsForNeuron(1, 2, new double[] { -1, 3 });
            nn.SetIncomingWeightsForNeuron(2, 0, new double[] { -1, -2, 3 });
            Vector<double> result = nn.ComputeOutput(new double[] { 1.0, 2.0 });
            Assert.AreEqual(10.0, result[0], 0.0001);
        }

        [TestMethod]
        public void TestSimpleDeepNetProcessing1()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new LinearActivation());
            nn.Initialize(CreationModes.NoAction);
            for (int i = 1; i < 5; i++)
            {
                nn.SetIncomingWeightsForLayer(i, new List<double[]>() { new double[] { 2 } });
            }

            Vector<double> result = nn.ComputeOutput(new double[] { 3 });
            TestContext.WriteLine(result[0].ToString());
            Assert.AreEqual(48, result[0]);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void TestUninitializedNetworkThrowsInvalidOperationExceptionOnCompute()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            Vector<double> result = nn.ComputeOutput(new double[] { 0 });
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void TestIncorrectInputLengthThrowsInvalidOperationExceptionOnCompute()
        {
            NeuralNetwork nn = new NeuralNetwork();
            nn.AddLayer(1, new LinearActivation());
            nn.AddLayer(1, new UnipolarSigmoidActivation());
            nn.Initialize(CreationModes.RandomizeWeights);
            Vector<double> result = nn.ComputeOutput(new double[] { 0, 1 });
        }
    }
}
