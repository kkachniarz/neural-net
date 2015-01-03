using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using Microsoft.Win32;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace Shell
{
    public static class Extensions
    {
        public static List<T> RemoveRange<T>(this List<T> list, IEnumerable<T> data)
        {
            foreach (var elem in data)
            {
                list.Remove(elem);
            }

            return list;
        }

        public static string ToCSVString(this DenseVector vector)
        {
            var sb = new StringBuilder();

            for (int i = 0; i < vector.Count(); i++)
            {
                sb.Append(",");
                sb.Append(vector[i].ToString());
            }

            return sb.ToString();
        }

        public static double StandardDeviation(this Vector<double> vector)
        {
            if(vector.Count <=1)
            {
                return 0.0;
            }

            double average = vector.Average();
            Vector<double> diffs = vector - average;
            diffs = diffs.PointwiseMultiply(diffs);

            return Math.Sqrt(diffs.Sum() / (double)(diffs.Count - 1));
        }
    }
}
