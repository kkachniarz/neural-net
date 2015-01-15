using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN.ActivationFunctions;
using SharpNN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace RecursiveNN
{
    [Serializable]
    public class ContextLayer : Layer
    {
        public Vector<double> Memory;

        public ContextLayer(int neuronCount)
            : base(neuronCount, null)
        {
            Memory = new DenseVector(neuronCount);
        }

        [Obsolete("Use 'Memory' field to manipulate ContextLayer state.")]
        public override Vector<double> CalculateOutput(Vector<double> signal)
        {
            throw new InvalidOperationException();
        }
    }
}
