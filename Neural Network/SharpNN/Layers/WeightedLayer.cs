using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.Layers
{
    /// <summary>
    /// Represents a fully-connected layer with weighted input. 
    /// </summary>
    public class WeightedLayer : Layer
    {
        private Vector<double> LastBiasWeightsModification { get; set; }
        public Matrix<double> LastWeightsModification { get; set; }
        /// <summary>
        /// Stores weights of all connections coming to this layer.
        /// i-th row represents incoming weights of the i-th neuron.
        /// Weight (i, j) is the weight of the connection from the j-th neuron 
        /// of previous layer to the i-th neuron of the current layer.
        /// </summary>
        public Matrix<double> IncomingWeights { get; set; }
        public Vector<double> BiasWeights { get; set; }
        public bool HasBias { get; private set; }
        public Vector<double> Error { get; set; }
        public Vector<double> LastSignal { get; set; }

        public WeightedLayer(int neuronCount, IActivation activation)
            : base(neuronCount, activation)
        {
            this.HasBias = false;
        }

        public WeightedLayer(int neuronCount, IActivation activation, bool hasBias)
            : base(neuronCount, activation)
        {
            this.HasBias = hasBias;
            if(hasBias)
            {
                CreateBiasWeights();
            }
        }

        private void CreateBiasWeights()
        {
            BiasWeights = new DenseVector(NeuronCount);
            LastBiasWeightsModification = new DenseVector(NeuronCount);
        }

        public override Vector<double> CalculateOutput(Vector<double> signal)
        {
            LastSignal = signal;

            LastOutput = IncomingWeights.Multiply(signal); // weighted input
            if (HasBias)
            {
                LastOutput = LastOutput.Add(BiasWeights); // weighted input with bias
            }

            LastOutput.MapInplace(Activation.Calc); // neuron activations (neuron outputs)
            return LastOutput.Clone();
        }

        public void RandomizeWeights()
        {
            AssertIncomingWeightsExist();
            double maxAbsValue = 1.0 / Math.Sqrt(IncomingWeights.ColumnCount); // based on http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
            RandomizeConnectionWeights(maxAbsValue);
            if (HasBias)
            {
                RandomizeBiasWeights(maxAbsValue);
            }
        }

        public void SetWeights(double value)
        {
            AssertIncomingWeightsExist();
            SetConnectionWeights(value);
            if (HasBias)
            {
                SetBiasWeights(value);
            }
        }
        private void SetConnectionWeights(double value)
        {
            IncomingWeights.MapInplace(x => value);
        }
        private void SetBiasWeights(double value)
        {
            BiasWeights.MapInplace(x => MathHelper.RandomExceptZero(value));
        }
        private void RandomizeConnectionWeights(double maxAbsValue)
        {
            IncomingWeights.MapInplace(x => MathHelper.RandomExceptZero(maxAbsValue));
        }

        private void RandomizeBiasWeights(double maxAbsValue)
        {
            BiasWeights.MapInplace(x => MathHelper.RandomExceptZero(maxAbsValue));
        }

        private void AssertIncomingWeightsExist()
        {
            if (IncomingWeights == null)
            {
                throw new InvalidOperationException("Cannot randomize weights. " +
                    "This layer has no incoming connections set, therefore weight matrix is unknown");
            }
        }

        public void SetIncomingWeights(List<double[]> weights)
        {
            for (int i = 0; i < weights.Count; i++)
            {
                IncomingWeights.SetRow(i, weights[i]);
            }
        }

        public void SetBias(double[] biasWeights)
        {
            BiasWeights = new DenseVector(biasWeights);
        }

        public void SetIncomingWeightsForNeuron(int neuronIndex, double[] weights)
        {
            IncomingWeights.SetRow(neuronIndex, weights);
        }

        public void SetBiasForNeuron(int neuronIndex, double bias) // TODO: write test
        {
            BiasWeights.Storage[neuronIndex] = bias;
        }      
  
        public void CalculateError(Vector<double> modelOutput)
        {
            Error = modelOutput - LastOutput;
        }

        public void ImproveWeights(double learningFactor, double momentum)
        {
            var output = LastOutput.Clone();
            output.MapInplace(Activation.CalcDerivative);

            var learningErrorOutput = learningFactor * Vector.op_DotMultiply(output, Error);
            var modification = learningErrorOutput.OuterProduct(LastSignal);

            IncomingWeights += modification + (LastWeightsModification * momentum);

            if (HasBias)
            {
                BiasWeights += learningErrorOutput + (LastBiasWeightsModification * momentum);
            }

            LastWeightsModification = modification;
            LastBiasWeightsModification = learningErrorOutput;
        }

        public void PropagateErrorsBack()
        {
            if(Previous is WeightedLayer)
            {
                var prev = Previous as WeightedLayer;
                prev.Error = new DenseVector(IncomingWeights.ColumnCount);

                for (int i = 0; i < IncomingWeights.RowCount; i++)
                {
                    for (int j = 0; j < IncomingWeights.ColumnCount; j++)
                    {
                        prev.Error[j] += Error[i] * IncomingWeights[i, j];
                    }
                }
            }
        }

        public void ConnectFrom(Layer prevLayer, Layer contextLayer = null)
        {
            int prevNeurons = prevLayer.NeuronCount + (contextLayer != null ? contextLayer.NeuronCount : 0);

            this.IncomingWeights = new DenseMatrix(this.NeuronCount, prevNeurons);
            this.LastWeightsModification = new DenseMatrix(this.NeuronCount, prevNeurons);

            prevLayer.Next = this;

            if (contextLayer != null)
            {
                contextLayer.Next = this;
            }

            this.Previous = prevLayer;
        }
    }
}
