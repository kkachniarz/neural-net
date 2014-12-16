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
    public class RecursiveNetwork : INetwork
    {
        public IActivation Activation { get; private set; }
        public Type NetworkType { get; private set; }
        public bool IsInitialized { get { return true; } }

        private InputLayer inputLayer;
        private WeightedLayer hiddenLayer;
        private WeightedLayer outputLayer;
        private ContextLayer contextLayer;

        public RecursiveNetwork(Type type, IActivation activation, bool useBiases, int inputNeurons, int hiddenNeurons, int outputNeurons)
        {
            Activation = activation;
            NetworkType = type;

            inputLayer = new InputLayer(inputNeurons);
            hiddenLayer = new WeightedLayer(hiddenNeurons, activation, useBiases);
            contextLayer = new ContextLayer(hiddenNeurons);
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
        { }

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

        private Vector<double> ConcatSignals(Vector<double> x, Vector<double> y)
        {
 	        return new DenseVector(x.Concat(y).ToArray());
        }

        public enum Type
        {
            Jordan,
            Elman,
        }


    }
}
