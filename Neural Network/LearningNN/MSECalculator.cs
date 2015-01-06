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
        public static double CalculateRawAverageMSE(Vector<double> error)
        {
            return (error.PointwiseMultiply(error)).Average();
        }

        public static double CalculateEpochMSE(double rawAverageMSESum, double dataSetSize, double dataMinOutput, double dataMaxOutput)
        {
            double span = ((dataMaxOutput - dataMinOutput) * (dataMaxOutput - dataMinOutput));
            return rawAverageMSESum / (span * dataSetSize);
        }
    }
}
