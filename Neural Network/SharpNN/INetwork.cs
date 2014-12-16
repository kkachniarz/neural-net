using MathNet.Numerics.LinearAlgebra;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN
{
    public interface INetwork
    {
        IActivation Activation { get; }

        bool IsInitialized { get; }

        void Initialize(CreationModes mode);

        Vector<double> ComputeOutput(Vector<double> signal);

        void CalculateAndPropagateError(Vector<double> modelAnswer);

        void ImproveWeights(double learningRate, double momentum);
    }
}
