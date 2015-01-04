using LearningNN.Learning;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Containers
{
    public class EngineResult
    {
        public Dictionary<LearningSettings, List<SingleRunReport>> ResultsBySettings;
        public EngineInitData Eid;
    }
}
