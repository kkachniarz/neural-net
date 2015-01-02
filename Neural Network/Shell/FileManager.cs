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

        public static List<LearningSettings> RetrieveParameters(string path)
        {
            var reader = new StreamReader(path);
            List<string[]> splitLines = new List<string[]>();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine(); // data
                line = line.Replace(" ", "");
                string[] values = line.Split(',', ':');
                if(values.Length == 0)
                {
                    continue;
                }

                splitLines.Add(values);
            }
                       
            Dictionary<string, List<string>> groupedParameters = PrepareParameterLists(splitLines);            
            return BuildSettings(groupedParameters, LearningSettings.RequiredTitles);
        }

        public static void SaveLearningInfo(string path, string text)
        {
            using (FileStream fs = new FileStream(path, FileMode.CreateNew))
            {
                using (StreamWriter sw = new StreamWriter(fs))
                {
                    sw.Write(text);
                }
            }

        }

        // TODO: refactor - move these methods outside FileManager class
        private static List<LearningSettings> BuildSettings(Dictionary<string, List<string>> groupedParameters, 
            List<string> titles)
        {            
            Stack<string> titleStack = new Stack<string>(titles);
            List<LearningSettings> ret = new List<LearningSettings>();
            BuildSettingsImplementation(groupedParameters, titleStack, 
                new LearningSettings(), ret);
            return ret;
        }

        private static void BuildSettingsImplementation(Dictionary<string, List<string>> groupedParameters, 
            Stack<string> titleStack, LearningSettings currentSetting, List<LearningSettings> result)
        {
            if(titleStack.Count == 0)
            {
                // recursion end
                result.Add(currentSetting.Clone()); // add a snapshot of the current settings to the result
                return;
            }

            string paramTitle = titleStack.Pop();
            foreach(string paramValue in groupedParameters[paramTitle])
            {
                currentSetting.SetParamByTitle(paramTitle, paramValue);
                BuildSettingsImplementation(groupedParameters, titleStack, currentSetting, result);
            }

            titleStack.Push(paramTitle);
        }

        private static Dictionary<string, List<string>> PrepareParameterLists(List<string[]> splitLines)
        {
            Dictionary<string, List<string>> stringPresent = new Dictionary<string, List<string>>();
            foreach(string s in LearningSettings.RequiredTitles)
            {
                stringPresent.Add(s, new List<string>());
            }

            foreach(string[] line in splitLines)
            {
                string title = line[0];
                if(stringPresent.ContainsKey(title))
                {
                    for(int i = 1; i < line.Length; i++)
                    {
                        stringPresent[title].Add(line[i]);
                    }
                }
            }

            foreach(KeyValuePair<string, List<string>> kvp in stringPresent)
            {
                if(kvp.Value.Count == 0)
                {
                    throw new ArgumentException(string.Format(
                        "Values for parameter: {0} have not been provided in the parameters file", kvp.Key));
                }
            }

            return stringPresent;
        }
    }
}
