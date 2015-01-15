using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN;
using SharpNN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecursiveNN
{
    [Serializable]
    public class RecursiveNetwork : INetwork
    {
        private const double DEFAULT_RND_RANGE = 0.5;
        public IActivation Activation { get; private set; }
        public Type NetworkType { get; private set; }
        public bool IsInitialized { get; private set; }
        
        private InputLayer inputLayer;
        private WeightedLayer hiddenLayer;
        private WeightedLayer outputLayer;
        private ContextLayer contextLayer;

        public RecursiveNetwork(Type type, IActivation activation, bool useBiases, int inputNeurons, int hiddenNeurons, int outputNeurons)
        {
            IsInitialized = false;
            Activation = activation;
            NetworkType = type;

            inputLayer = new InputLayer(inputNeurons);
            hiddenLayer = new WeightedLayer(hiddenNeurons, activation, useBiases);
            contextLayer = new ContextLayer(NetworkType == Type.Jordan ? outputNeurons : hiddenNeurons);
            outputLayer = new WeightedLayer(outputNeurons, activation, useBiases);

            hiddenLayer.ConnectFrom(inputLayer, contextLayer);
            outputLayer.ConnectFrom(hiddenLayer);
        }

        public Vector<double> ComputeOutput(Vector<double> signal)
        {
            signal = inputLayer.CalculateOutput(signal);
            signal = ConcatSignals(signal, contextLayer.Memory);
            signal = hiddenLayer.CalculateOutput(signal);

            if (NetworkType == Type.Elman)
            {
                contextLayer.Memory = signal;
            }

            signal = outputLayer.CalculateOutput(signal);

            if (NetworkType == Type.Jordan)
            {
                contextLayer.Memory = signal;
            }
            
            return signal.Clone();
        }

        public void Initialize(CreationModes mode)
        {
            hiddenLayer.RandomizeWeights();
            outputLayer.RandomizeWeights();
            IsInitialized = true;
        }

        public void CalculateAndPropagateError(Vector<double> modelAnswer)
        {
            outputLayer.CalculateError(modelAnswer);
            outputLayer.PropagateErrorsBack();
            hiddenLayer.PropagateErrorsBack();
        }

        public void ImproveWeights(double learningRate, double momentum)
        {
            hiddenLayer.ImproveWeights(learningRate, momentum);
            outputLayer.ImproveWeights(learningRate, momentum);
        }

        public void ResetMemory()
        {
            contextLayer.Memory.Clear();
        }

        private Vector<double> ConcatSignals(Vector<double> x, Vector<double> y)
        {
 	        return new DenseVector(x.Concat(y).ToArray());
        }

        [Serializable]
        public enum Type
        {
            Jordan,
            Elman,
        }

        public object SaveWeights()
        {
            SavedWeights save = new SavedWeights();
            save.HiddenWeights = hiddenLayer.IncomingWeights.Clone();
            save.OutputWeights = outputLayer.IncomingWeights.Clone();

            save.HiddenBias = hiddenLayer.BiasWeights.Clone();
            save.OutputBias = outputLayer.BiasWeights.Clone();

            save.Memory = contextLayer.Memory.Clone();

            return save;
        }

        public void RestoreWeights(object savedWeights)
        {
            SavedWeights save = (SavedWeights)savedWeights;

            hiddenLayer.IncomingWeights = save.HiddenWeights.Clone();
            outputLayer.IncomingWeights = save.OutputWeights.Clone();

            hiddenLayer.BiasWeights = save.HiddenBias.Clone();
            outputLayer.BiasWeights = save.OutputBias.Clone();

            contextLayer.Memory = save.Memory.Clone();
        }

        private class SavedWeights
        {
            public Matrix<double> HiddenWeights;
            public Matrix<double> OutputWeights;

            public Vector<double> HiddenBias;
            public Vector<double> OutputBias;
            public Vector<double> Memory;
        }
    }
}
