using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.Layers
{
    /// <summary>
    /// Represents a neural network layer.
    /// </summary>
    public abstract class Layer
    {
        public Vector<double> LastOutput { get; protected set; }
        public int NeuronCount { get; protected set; }
        public IActivation Activation {get; protected set;}
        public Layer Previous { get; set; }
        public Layer Next { get; set; }

        public Layer(int neuronCount, IActivation activation)
        {
            this.NeuronCount = neuronCount;
            this.Activation = activation;
        }

        public void ConnectTo(WeightedLayer nextLayer)
        {
            nextLayer.IncomingWeights = new DenseMatrix(nextLayer.NeuronCount, this.NeuronCount); // column-major, wektory są pionowe, mnozymy M x v, nie v x M, stąd taki wymiar macierzy
            nextLayer.LastWeightsModification = new DenseMatrix(nextLayer.NeuronCount, this.NeuronCount);
            Next = nextLayer;
            nextLayer.Previous = this;
        }

        public abstract Vector<double> CalculateOutput(Vector<double> signal);
    }
}
