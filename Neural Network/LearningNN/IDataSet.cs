using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public interface IDataSet
    {
        int PatternCount { get; }
        DataExtremumsForNetwork Extremums { get; }
        Pattern GetPatternAt(int i);
        IEnumerable<Pattern> EnumeratePatterns();
        IDataSet Clone();
        void UpdateExtremums();
        void Normalize(double minTo, double maxTo, DataExtremumsForNetwork extremums);
        void NormalizeBack();
    }
}
