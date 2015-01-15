using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using SharpNN.Layers;
using SharpNN.ActivationFunctions;
using System.Diagnostics;
using System.Runtime.Serialization;

namespace SharpNN
{
    [Serializable]
    public partial class NeuralNetwork : INetwork
    {
        private const double DEFAULT_RND_RANGE = 0.5;//0.05;
        private const bool DEFAULT_HAS_BIAS = false;
        private List<Layer> layers;
        private InputLayer inputLayer;
        private List<WeightedLayer> weightedLayers;
        private Vector<double> lastOutput;

        public bool IsInitialized { get; private set; }
        public int LayerCount { get { return layers.Count; } }
        public int InputCount { get { return inputLayer.NeuronCount; } }
        public int OutputCount { get { return layers[layers.Count - 1].NeuronCount; } }
        public int HiddenLayerCount { get { return weightedLayers.Count - 1; } }

        public Vector<double> LastOutput { get { return lastOutput; } }
        public IActivation Activation { get; private set; }

        [Obsolete("Obsolete due to activation function constrains - same for all layers")]
        public NeuralNetwork()
        {
            IsInitialized = false;
            layers = new List<Layer>();
            weightedLayers = new List<WeightedLayer>();
        }

        public NeuralNetwork(IActivation activation, bool useBiases, params int[] neuronCounts) : this()
        {
            Activation = activation;
            AddInputLayer(neuronCounts[0]);
            for(int i=1; i<neuronCounts.Length; i++)
            {
                AddWeightedLayer(neuronCounts[i], activation, useBiases);
            }
        }

        [Obsolete("Obsolete due to activation function constrains - same for all layers")]
        private void AddInputLayer(int neuronCount)
        {
            inputLayer = new InputLayer(neuronCount);
            layers.Add(inputLayer);
        }

        [Obsolete("Obsolete due to activation function constrains - same for all layers")]
        private void AddWeightedLayer(int neuronCount, IActivation activation, bool hasBias)
        {
            WeightedLayer adding = new WeightedLayer(neuronCount, activation, hasBias);
            weightedLayers.Add(adding);
            layers.Add(adding);
        }

        /// <summary>
        /// Connects layers in a feed-forward fashion, creating weights that connect them.
        /// </summary>
        /// <param name="randomizeWeights">If set to true, the weights will be given small random values</param>
        public void Initialize(CreationModes mode)
        {
            AssertCanInitialize();
            ConnectLayersForward();

            if(mode == CreationModes.RandomizeWeights)
            {
                RandomizeWeights();
            }
            else if(mode == CreationModes.SetWeightToHalf)
            {
                SetWeights(0.5);
            }

            IsInitialized = true;
        }

        private void AssertCanInitialize()
        {
            if (LayerCount < 2)
            {
                throw new ArgumentException("Too few layers to build the network");
            }
        }

        private void ConnectLayersForward()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                WeightedLayer currentLayer = (WeightedLayer)layers[i]; // cast or throw exception if it's not the right type
                layers[i - 1].ConnectTo(currentLayer);
            }
        }

        public Vector<double> ComputeOutput(double[] networkInput)
        {
            Vector<double> currentSignal = new DenseVector(networkInput);
            return ComputeOutputInternal(currentSignal);
        }

        public Vector<double> ComputeOutput(Vector<double> networkInput)
        {
            Vector<double> currentSignal = networkInput.Clone();
            return ComputeOutputInternal(currentSignal);
        }

        private Vector<double> ComputeOutputInternal(Vector<double> networkInput)
        {
            AssertIsInitialized();
            AssertCanCompute(networkInput);
            for (int i = 0; i < layers.Count; i++)
            {
                networkInput = layers[i].CalculateOutput(networkInput);
            }

            this.lastOutput = networkInput;
            return this.lastOutput.Clone();
        }

        private void AssertCanCompute(Vector<double> networkInput)
        {
            if (networkInput.Count != layers[0].NeuronCount)
            {
                throw new InvalidOperationException("Input length: " + networkInput.Count +
                    " does not match neuron count at input layer: " + layers[0].NeuronCount);
            }
        }

        private void AssertIsInitialized()
        {
            if (!IsInitialized)
            {
                throw new InvalidOperationException(
                    "Cannot proceed, the network has not been initialized yet");
            }
        }

        public void AddLayer(int neuronCount, IActivation activation)
        {
            if (layers.Count == 0)
            {
                AddInputLayer(neuronCount);
            }
            else
            {
                AddWeightedLayer(neuronCount, activation, DEFAULT_HAS_BIAS);
            }
        }

        public void AddLayer(int neuronCount, IActivation activation, bool hasBias)
        {
            if (layers.Count == 0 && hasBias)
            {
                throw new InvalidOperationException("Input layer cannot have bias.");
            }
            
            if(layers.Count == 0)
            {
                AddInputLayer(neuronCount);
            }
            else
            {
                AddWeightedLayer(neuronCount, activation, hasBias);
            }
        }

        public Layer LayerAt(int index)
        {
            return layers[index];
        }

        public void RandomizeWeights()
        {
            foreach (WeightedLayer weightedLayer in layers.OfType<WeightedLayer>())
            {
                weightedLayer.RandomizeWeights();
            }
        }

        public void SetWeights(double value)
        {
            foreach (WeightedLayer weightedLayer in layers.OfType<WeightedLayer>())
            {
                weightedLayer.SetWeights(value);
            }
        }

        public void SetIncomingWeightsForNeuron(int layerIndex, int neuronIndex, double[] weights)
        {
            WeightedLayer weightedLayer = (WeightedLayer)layers[layerIndex];
            weightedLayer.SetIncomingWeightsForNeuron(neuronIndex, weights);
        }

        public void SetIncomingWeightsForLayer(int layerIndex, List<double[]> incomingWeights)
        {
            // important: the matrices in Math.NET are column-major!
            WeightedLayer weightedLayer = (WeightedLayer)layers[layerIndex];

            if (incomingWeights.Count != weightedLayer.NeuronCount)
            {
                throw new ArgumentException("Number of weight vectors passed " +
                    "is different than the neuron count in this layer");
            }

            weightedLayer.SetIncomingWeights(incomingWeights);
        }

        public void SetBiasForLayer(int layerIndex, double[] bias)
        {
            WeightedLayer weightedLayer = (WeightedLayer)layers[layerIndex];

            if (bias.Length != weightedLayer.NeuronCount)
            {
                throw new ArgumentException("Number of bias values provided does not match neuron count");
            }

            weightedLayer.SetBias(bias);
        }

        public void SetBiasForNeuron(int layerIndex, int neuronIndex, double bias) // TODO: write test
        {
            WeightedLayer weightedLayer = (WeightedLayer)layers[layerIndex];
            weightedLayer.SetBiasForNeuron(neuronIndex, bias);
        }


        public void CalculateAndPropagateError(Vector<double> modelAnswer)
        {
            ((WeightedLayer)LayerAt(LayerCount - 1)).CalculateError(modelAnswer);

            for (int k = LayerCount - 1; k > 0; k--)
            {
                ((WeightedLayer)LayerAt(k)).PropagateErrorsBack();
            }
        }

        public void ImproveWeights(double learningRate, double momentum)
        {
            for (int k = 1; k < LayerCount; k++)
            {
                ((WeightedLayer)LayerAt(k)).ImproveWeights(learningRate, momentum);
            }
        }

        public object SaveWeights()
        {
            SavedWeights save = new SavedWeights(weightedLayers.Count);
            for(int i = 0; i < weightedLayers.Count; i++)
            {
                save.WeightMatrices[i] = weightedLayers[i].IncomingWeights.Clone();
                if(weightedLayers[i].HasBias)
                {
                    save.Biases[i] = weightedLayers[i].BiasWeights.Clone();
                }
            }

            return save;
        }

        public void RestoreWeights(object savedWeights)
        {
            SavedWeights save = (SavedWeights)savedWeights;
            if(save.WeightMatrices.Length != weightedLayers.Count)
            {
                throw new ArgumentException("The number of saved matrices is different than the number of weight layers");
            }

            for(int i = 0; i < weightedLayers.Count; i++)
            {
                weightedLayers[i].IncomingWeights = save.WeightMatrices[i].Clone();
                if (weightedLayers[i].HasBias)
                {
                    weightedLayers[i].BiasWeights = save.Biases[i].Clone();
                }
            }
        }

        public class SavedWeights
        {
            public Matrix<double>[] WeightMatrices;
            public Vector<double>[] Biases;

            public SavedWeights(int count)
            {
                WeightMatrices = new Matrix<double>[count];
                Biases = new Vector<double>[count];
            }
        }
    }
}
