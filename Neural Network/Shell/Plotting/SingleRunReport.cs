using LearningNN.DataSet;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Plotting
{
    /// <summary>
    /// Contains data concerning a single execution (run) of learning.
    /// </summary>
    public class SingleRunReport
    {
        public DateTime Time;
        public LearningResult LearningResult;
        public IDataSet TrainSet;
        public IDataSet TestSet;
        public INetwork Network;

        public string Name { get; set; }

        public SingleRunReport(INetwork net, DateTime time, LearningResult lres)
        {
            LearningResult = lres;
            Network = net;
            Time = time;
        }

        public SingleRunReport(INetwork net, DateTime time, LearningResult lres, IDataSet train, IDataSet test) 
            : this(net, time, lres)
        {
            TrainSet = train;
            TestSet = test;
        }
    }
}
