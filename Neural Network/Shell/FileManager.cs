using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SharpNN;
using SharpNN.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace Neural_Network
{
    public static class FileManager
    {
        public static List<DenseVector> ReadDataFromCSV(string path)
        {
            var reader = new StreamReader(path);
            var list = new List<DenseVector>();

            if (!reader.EndOfStream)
                reader.ReadLine(); // header

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine(); // data
                var values = line.Split(',');

                var vector = new DenseVector(new double[values.Count()]);

                for (int i = 0; i < vector.Count(); i++)
                {
                    vector[i] = Double.Parse(values[i], CultureInfo.InvariantCulture);
                }

                list.Add(vector);
            }

            return list;
        }

        public static void AppendDataToCSV(string path, List<DenseVector> newData)
        {
            if(newData == null || newData.Count < 2) // header plus minimum one data row is two
            {
                throw new ArgumentException();
            }

            List<string> lines = File.ReadAllLines(path).ToList();

            if(lines.Count - 1 != newData.Count) // substraction due to header in lines
            {
                throw new ArgumentException();
            }

            // header
            for (int i = 0; i < newData[0].Count; i++)
            {
                lines[0] += ",output" + (i + 1).ToString();
            }

            // data
            for (int i = 1; i < lines.Count(); i++)
            {
                lines[i] += newData[i - 1].ToCSVString();
            }

            File.WriteAllLines(path, lines);
        }

    }
}
