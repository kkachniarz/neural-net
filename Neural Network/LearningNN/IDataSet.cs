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
        double MinValue { get; }
        double MaxValue { get; }

        Pattern GetPatternAt(int i);
        IEnumerable<Pattern> EnumeratePatterns();
        IDataSet Clone();

        void Normalize(double minFrom, double maxFrom, double minTo, double maxTo);
        void NormalizeBack();
    }
}
