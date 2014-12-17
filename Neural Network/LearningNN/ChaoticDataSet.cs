using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class ChaoticDataSet : DataSet
    {
        private int historyLength;

        public ChaoticDataSet(IList<DenseVector> chaoticSeries, int prevValuesCount, int startTime)
            : base(startTime)
        {
            AssertCTSValid(chaoticSeries, prevValuesCount);
            this.historyLength = prevValuesCount;

            patterns = new List<Pattern>(chaoticSeries.Count - prevValuesCount);

            AddPatterns(chaoticSeries, prevValuesCount);
            UpdateExtrema(chaoticSeries);
        }

        private void AddPatterns(IList<DenseVector> chaoticSeries, int startIndex)
        {
            for (int i = startIndex; i < chaoticSeries.Count; i++)
            {
                Pattern adding = new Pattern(startTime + i);
                DenseVector input = new DenseVector(historyLength);
                for (int j = 1; j <= historyLength; j++)
                {
                    input.At(j - 1, chaoticSeries[i - j].At(0)); // At(i) is likely faster than [i] in Math.NET
                }

                adding.Input = input;
                adding.IdealOutput = chaoticSeries[i];
                patterns.Add(adding);
            }
        }

        private void AssertCTSValid(IList<DenseVector> cts, int historyLength)
        {
            if(cts.Count <= historyLength)
            {
                throw new ArgumentException("Too few patterns for the history length specified");
            }

            if(cts[0].Count != 1)
            {
                throw new ArgumentException("Chaotic time series must consist of vectors of length 1");
            }
        }
    }
}
