using LearningNN.Learning;
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

namespace Shell
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

        /// <summary>
        /// Create a list of all possible configurations based on the file provided.
        /// That is, generate all possible combinations of settings
        /// by combining values that are specified in the file.
        /// </summary>
        /// <param name="path">Path to parameters file</param>
        /// <returns>All possible combinations of settings</returns>
        public static List<LearningSettings> RetrieveParameters(string path)
        {
            var reader = new StreamReader(path);
            List<string[]> splitLines = new List<string[]>();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine(); // data
                string[] values = ParseParamsLine(line);
                if(values == null)
                {
                    continue;
                }

                splitLines.Add(values);
            }

            reader.Close();
            return SettingsMixer.BuildSettings(splitLines);
        }

        public static string[] ParseParamsLine(string line)
        {
            int commentIndex = line.IndexOf("//");
            if (commentIndex >= 0)
            {
                line = line.Substring(0, commentIndex);
            }

            string[] values = line.Split(new string[] { ",", ":", " " }, StringSplitOptions.RemoveEmptyEntries);
            if (values.Length == 0)
            {
                return null;
            }

            return values;
        }

        public static void SaveTextFile(string path, string text)
        {
            using (FileStream fs = new FileStream(path, FileMode.CreateNew))
            {
                using (StreamWriter sw = new StreamWriter(fs))
                {
                    sw.Write(text);
                }
            }
        }

        public static string ReadTextFile(string path)
        {
            string ret;
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                using (StreamReader st = new StreamReader(fs))
                {
                    ret = st.ReadToEnd();
                }
            }

            return ret;
        }
    }
}
