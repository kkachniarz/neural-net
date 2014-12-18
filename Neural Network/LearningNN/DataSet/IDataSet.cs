using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.DataSet
{
    public interface IDataSet
    {
        int PatternCount { get; }
        DataExtremumsForNetwork Extremums { get; }
        Pattern GetPatternAt(int i);
        IEnumerable<Pattern> EnumeratePatterns();
        IDataSet Clone();
        void UpdateExtremums();
        void Normalize(double minFrom, double maxFrom, DataExtremumsForNetwork extremums);
        void NormalizeBack();
    }
}
