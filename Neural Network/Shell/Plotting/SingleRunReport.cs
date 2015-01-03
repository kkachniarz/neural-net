using LearningNN.DataSet;
using SharpNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Plotting
{
    public class SingleRunReport
    {
        public SingleRunReport(LearningResult lres, IDataSet train, IDataSet test, INetwork net, List<int> lval, DateTime time)
        {
            LearningResult = lres;
            TrainSet = train;
            TestSet = test;
            Network = net;
            LayersVal = lval;
            Time = time;
        }

        public DateTime Time;
        public LearningResult LearningResult;
        public IDataSet TrainSet;
        public IDataSet TestSet;
        public INetwork Network;
        public List<int> LayersVal; // tmp, later should be moved to LearningSettings and required in parameters file

        public string Name { get; set; }
    }
}
