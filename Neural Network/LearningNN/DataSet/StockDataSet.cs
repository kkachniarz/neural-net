using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.DataSet
{
    public class StockDataSet : DataSet
    {
        public StockDataSet(List<DenseVector> inputs, List<DenseVector> outputs, int startTime)
            : base(startTime)
        {
            AssertDataValid(inputs, outputs);
            patterns = new List<Pattern>(inputs.Count);

            AddPatterns(inputs, outputs);
            UpdateExtremums();
        }

        private StockDataSet(int startTime) : base(startTime)
        {
        }

        private void AddPatterns(List<DenseVector> inputs, List<DenseVector> outputs)
        {
            for(int i = 0; i < inputs.Count; i++)
            {
                Pattern adding = new Pattern(startTime + i);
                adding.Input = inputs[i];
                adding.IdealOutput = outputs[i];
                patterns.Add(adding);
            }
        }

        private void AssertDataValid(List<DenseVector> inputs, List<DenseVector> outputs)
        {
            if(inputs.Count != outputs.Count)
            {
                throw new ArgumentException("Inequal inputs and outputs count. Cannot construct data set");
            }
        }

        public override IDataSet Clone()
        {
            StockDataSet clone = new StockDataSet(this.startTime);
            clone.patterns = new List<Pattern>();
            foreach (Pattern p in this.patterns)
            {
                Pattern clonedPattern = p.Clone();
                clone.patterns.Add(clonedPattern);
            }

            clone.startTime = this.startTime;

            return clone;
        }
    }
}
