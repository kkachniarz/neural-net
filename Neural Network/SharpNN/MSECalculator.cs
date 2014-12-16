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
            return Math.Sqrt(rawMSESum / (((activation.MaxValue - activation.MinValue) * (activation.MaxValue - activation.MinValue))
                * dataSetSize));
        }
    }
}
