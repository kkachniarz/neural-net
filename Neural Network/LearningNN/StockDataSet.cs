﻿using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class StockDataSet : DataSet
    {
        public StockDataSet(List<DenseVector> inputs, List<DenseVector> outputs, int startTime)
            : base(startTime)
        {
            AssertDataValid(inputs, outputs);
            patterns = new List<Pattern>(inputs.Count);

            AddPatterns(inputs, outputs);
            UpdateExtrema(inputs);
            UpdateExtrema(outputs);
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
    }
}
