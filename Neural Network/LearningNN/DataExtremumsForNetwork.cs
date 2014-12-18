using System.Collections.Generic;

namespace LearningNN
{
    public class DataExtremumsForNetwork
    {
        public IList<DataExtremum> InputExtremums { get; set; }
        public IList<DataExtremum> OutputExtremums { get; set; }

        public DataExtremumsForNetwork()
        {
            InputExtremums = new List<DataExtremum>();
            OutputExtremums = new List<DataExtremum>();
        }

        public static DataExtremumsForNetwork Merge(DataExtremumsForNetwork x, DataExtremumsForNetwork y)
        {
            var ret = new DataExtremumsForNetwork();

            for (int i = 0; i < x.InputExtremums.Count; i++)
            {
                ret.InputExtremums.Add(DataExtremum.Merge(x.InputExtremums[i], y.InputExtremums[i]));
            }

            for (int i = 0; i < x.OutputExtremums.Count; i++)
            {
                ret.OutputExtremums.Add(DataExtremum.Merge(x.OutputExtremums[i], y.OutputExtremums[i]));
            }

            return ret;
        }
    }
}
