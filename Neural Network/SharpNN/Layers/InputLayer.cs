using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.Layers
{
    /// <summary>
    /// Represents an input layer where each neuron takes a single value as input. 
    /// There are no incoming weights.
    /// </summary>
    [Serializable]
    public class InputLayer : Layer
    {
        public InputLayer(int neuronCount) : base(neuronCount, new LinearActivation())
        {
        }

        public override Vector<double> CalculateOutput(Vector<double> signal)
        {
            LastOutput = signal.Map(Activation.Calc);
            return LastOutput.Clone();
        }
    }
}
