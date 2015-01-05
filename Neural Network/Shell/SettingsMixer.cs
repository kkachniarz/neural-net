using LearningNN.Learning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell
{
    public static class SettingsMixer
    {
        public static List<LearningSettings> BuildSettings(List<string[]> splitLines)
        {
            Dictionary<string, List<string>> groupedParameters = PrepareParameterLists(splitLines);
            Stack<string> titleStack = new Stack<string>(LearningSettings.RequiredTitles);
            List<LearningSettings> ret = new List<LearningSettings>();
            BuildSettingsImplementation(groupedParameters, titleStack,
                new LearningSettings(), ret);
            return ret;
        }

        private static void BuildSettingsImplementation(Dictionary<string, List<string>> groupedParameters,
            Stack<string> titleStack, LearningSettings currentSetting, List<LearningSettings> result)
        {
            if (titleStack.Count == 0)
            {
                // recursion end
                result.Add(currentSetting.Clone()); // add a snapshot of the current settings to the result
                return;
            }

            string paramTitle = titleStack.Pop();
            foreach (string paramValue in groupedParameters[paramTitle])
            {
                currentSetting.SetParamByTitle(paramTitle, paramValue);
                BuildSettingsImplementation(groupedParameters, titleStack, currentSetting, result);
            }

            titleStack.Push(paramTitle);
        }

        private static Dictionary<string, List<string>> PrepareParameterLists(List<string[]> splitLines)
        {
            Dictionary<string, List<string>> stringPresent = new Dictionary<string, List<string>>();
            foreach (string s in LearningSettings.RequiredTitles)
            {
                stringPresent.Add(s, new List<string>());
            }

            foreach (string[] line in splitLines)
            {
                string title = line[0];
                if (stringPresent.ContainsKey(title))
                {
                    for (int i = 1; i < line.Length; i++)
                    {
                        stringPresent[title].Add(line[i]);
                    }
                }
            }

            foreach (KeyValuePair<string, List<string>> kvp in stringPresent)
            {
                if (kvp.Value.Count == 0)
                {
                    throw new ArgumentException(string.Format(
                        "Values for parameter: {0} have not been provided in the parameters file", kvp.Key));
                }
            }

            return stringPresent;
        }
    }
}
