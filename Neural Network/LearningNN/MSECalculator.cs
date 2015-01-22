using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class MSECalculator
    {
        public static double CalculateRawMSE(Vector<double> error)
        {
            return (error.PointwiseMultiply(error)).Average();
        }

        public static double CalculateEpochMSE(double rawMSESum, double dataSetSize, double dataMinOutput, double dataMaxOutput)
        {
            double span = ((dataMaxOutput - dataMinOutput) * (dataMaxOutput - dataMinOutput));
            return rawMSESum / (span * dataSetSize);
        }

        public static double CalculateEpochMSEDenormalized(double mseSum, double setSize)
        {
            return mseSum / setSize;
        }
    }
}
