using LearningNN.DataSet;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.LearningStrategy
{
    public interface ILearningStrategy
    {
        double LearningRate { get; set; }
        double Momentum { get; set; }
        List<double> Train(INetwork network, IDataSet dataSet);
    }
}
