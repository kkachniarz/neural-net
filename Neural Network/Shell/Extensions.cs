using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using Microsoft.Win32;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Neural_Network
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

            for(int i = 0; i < vector.Count(); i++)
            {
                sb.Append(",");
                sb.Append(vector[i].ToString());
            }

            return sb.ToString();
        }
    }
}
