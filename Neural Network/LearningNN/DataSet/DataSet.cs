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
        public double MinValue { get; protected set; }
        public double MaxValue { get; protected set; }

        protected IList<Pattern> patterns;
        protected Normalizor normalizer;
        protected int startTime; // starting time index. Needed for continuity between train and test sets

        public DataSet(int startTime)
        {
            MinValue = double.MaxValue;
            MaxValue = double.MinValue;
            this.startTime = startTime;
        }

        protected void UpdateExtremaInternal(IList<Vector<double>> vectors)
        {
            vectors = vectors.Where(x => x != null).ToList();

            foreach (var v in vectors)
            {
                double currentMin = v.Min();
                if (currentMin < MinValue)
                {
                    MinValue = currentMin;
                }

                double currentMax = v.Max();
                if (currentMax > MaxValue)
                {
                    MaxValue = currentMax;
                }
            }
        }

        public void UpdateExtrema()
        {
            UpdateExtremaInternal(patterns.Select(x => x.Input).ToList());
            UpdateExtremaInternal(patterns.Select(x => x.IdealOutput).ToList());
            UpdateExtremaInternal(patterns.Select(x => x.NetworkAnswer).ToList());
        }

        public void Normalize(double minFrom, double maxFrom, double minTo, double maxTo)
        {
            normalizer = new Normalizor(minFrom, maxFrom, minTo, maxTo);

            foreach (Pattern p in patterns)
            {
                p.Input.MapInplace(x => normalizer.Normalize(x));
                p.IdealOutput.MapInplace(x => normalizer.Normalize(x));
            }
        }

        public void NormalizeBack()
        {
            if (normalizer == null)
            {
                throw new ArgumentException("Cannot normalize back. No normalization has been performed.");
            }

            foreach (Pattern p in patterns)
            {
                p.Input.MapInplace(x => normalizer.NormalizeBack(x));
                p.IdealOutput.MapInplace(x => normalizer.NormalizeBack(x));
                if (p.NetworkAnswer != null)
                {
                    p.NetworkAnswer.MapInplace(x => normalizer.NormalizeBack(x));
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
