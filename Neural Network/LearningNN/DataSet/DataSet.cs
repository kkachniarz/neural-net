using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace LearningNN.DataSet
{
    public abstract class DataSet : IDataSet
    {
        public int PatternCount { get { return patterns.Count; } }
        public DataExtremumsForNetwork Extremums { get; protected set; }

        protected IList<Pattern> patterns;
        protected int startTime; // starting time index. Needed for continuity between train and test sets

        public DataSet(int startTime)
        {
            Extremums = new DataExtremumsForNetwork();

            this.startTime = startTime;
        }

        public void UpdateExtremums()
        {
            if (Extremums.InputExtremums.Count == 0)
            {
                var modelPattern = patterns.First();
                for (int i = 0; i < modelPattern.Input.Count; i++)
                {
                    Extremums.InputExtremums.Add(new DataExtremum(patterns.Select(x => x.Input[i]).ToList()));
                }
            }
            else
            {
                foreach (var extremum in Extremums.InputExtremums)
                {
                    extremum.Update();
                }
            }

            if (Extremums.OutputExtremums.Count == 0)
            {
                var modelPattern = patterns.First();
                for (int i = 0; i < modelPattern.IdealOutput.Count; i++)
                {
                    Extremums.OutputExtremums.Add(new DataExtremum(patterns.Select(x => x.IdealOutput[i]).ToList()));
                }
            }
            else
            {
                foreach (var extremum in Extremums.OutputExtremums)
                {
                    extremum.Update();
                }
            }
        }

        public void Normalize(double minTo, double maxTo, DataExtremumsForNetwork extremumus = null)
        {
            if (extremumus == null)
                extremumus = Extremums;

            var n = patterns.First().Input.Count;
            for (int i = 0; i < n; i++)
            {
                Extremums.InputExtremums[i].Normalizor = new Normalizor(
                    extremumus.InputExtremums[i].MinValue,
                    extremumus.InputExtremums[i].MaxValue,
                    minTo,
                    maxTo);
            }

            n = patterns.First().IdealOutput.Count;
            for (int i = 0; i < n; i++)
            {
                Extremums.OutputExtremums[i].Normalizor = new Normalizor(
                    extremumus.OutputExtremums[i].MinValue,
                    extremumus.OutputExtremums[i].MaxValue,
                    minTo,
                    maxTo);
            }

            foreach (Pattern p in patterns)
            {
                for (int i = 0; i < p.Input.Count; i++)
                {
                    p.Input[i] = Extremums.InputExtremums[i].Normalizor.Normalize(p.Input[i]);
                }

                for (int i = 0; i < p.IdealOutput.Count; i++)
                {
                    p.IdealOutput[i] = Extremums.OutputExtremums[i].Normalizor.Normalize(p.IdealOutput[i]);
                }
            }
        }

        public void NormalizeBack()
        {
            foreach (Pattern p in patterns)
            {
                for (int i = 0; i < p.Input.Count; i++)
                {
                    p.Input[i] = Extremums.InputExtremums[i].Normalizor.NormalizeBack(p.Input[i]);
                }

                for (int i = 0; i < p.IdealOutput.Count; i++)
                {
                    p.IdealOutput[i] = Extremums.OutputExtremums[i].Normalizor.NormalizeBack(p.IdealOutput[i]);
                }

                if (p.NetworkAnswer != null)
                {
                    for (int i = 0; i < p.NetworkAnswer.Count; i++)
                    {
                        p.NetworkAnswer[i] = Extremums.OutputExtremums[i].Normalizor.NormalizeBack(p.NetworkAnswer[i]);
                    }
                }
            }
        }

        public IEnumerable<Pattern> EnumeratePatterns()
        {
            for (int i = 0; i < patterns.Count; i++)
            {
                yield return patterns[i];
            }
        }

        public Pattern GetPatternAt(int i)
        {
            return patterns[i]; // skip index checking
        }

        public abstract IDataSet Clone();
    }
}
