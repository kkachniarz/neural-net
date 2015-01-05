using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN
{
    public class MSECalculator
    {
        public static double CalculateRawMSE(Vector<double> error)
        {
            return (error.PointwiseMultiply(error)).Average();
        }

        public static double CalculateNormalizedMSE(Vector<double> error, IActivation activation)
        {
            return (error.PointwiseMultiply(error)).Average() 
                / ((activation.MaxValue - activation.MinValue) * (activation.MaxValue - activation.MinValue));
        }

        public static double CalculateEpochMSE(double rawMSESum, double dataSetSize, IActivation activation)
        {
            double activationRange = ((activation.MaxValue - activation.MinValue) * (activation.MaxValue - activation.MinValue));
            return rawMSESum / (activationRange * dataSetSize);
        }
    }
}
