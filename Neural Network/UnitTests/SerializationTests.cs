using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNN;
using SharpNN.ActivationFunctions;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RecursiveNN;

namespace UnitTests
{
    [TestClass]
    public class SerializationTests
    {
        [TestMethod]
        public void SerializeMLPWorksBothWays()
        {
            NeuralNetwork nn = new NeuralNetwork(new UnipolarSigmoidActivation(), true, 2, 10, 3, 3);
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new MemoryStream();
            formatter.Serialize(stream, nn);
            stream.Seek(0, SeekOrigin.Begin);

            object result = formatter.Deserialize(stream);
            Assert.IsTrue(result is NeuralNetwork);
            NeuralNetwork resNet = (NeuralNetwork)result;

            Assert.AreEqual(2, resNet.InputCount);
            Assert.AreEqual(3, resNet.OutputCount);
            Assert.AreEqual(2, resNet.HiddenLayerCount);
        }

        [TestMethod]
        public void SerializeElmanWorksBothWays()
        {
            RecursiveNetwork nn = new RecursiveNetwork(RecursiveNetwork.Type.Elman, new UnipolarSigmoidActivation(), true, 2, 10, 3);
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new MemoryStream();
            formatter.Serialize(stream, nn);
            stream.Seek(0, SeekOrigin.Begin);

            object result = formatter.Deserialize(stream);
            Assert.IsTrue(result is RecursiveNetwork);
            RecursiveNetwork resNet = (RecursiveNetwork)result;

            Assert.AreEqual(RecursiveNetwork.Type.Elman, resNet.NetworkType);
        }


        [TestMethod]
        public void SerializeJordanorksBothWays()
        {
            RecursiveNetwork nn = new RecursiveNetwork(RecursiveNetwork.Type.Jordan, new UnipolarSigmoidActivation(), true, 2, 10, 3);
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new MemoryStream();
            formatter.Serialize(stream, nn);
            stream.Seek(0, SeekOrigin.Begin);

            object result = formatter.Deserialize(stream);
            Assert.IsTrue(result is RecursiveNetwork);
            RecursiveNetwork resNet = (RecursiveNetwork)result;

            Assert.AreEqual(RecursiveNetwork.Type.Jordan, resNet.NetworkType);
        }
    }
}
