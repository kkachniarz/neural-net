using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Containers
{
    public class ReportingOptions
    {
        public bool ShouldDisplayPlots { get; set; }
        public bool ShouldSavePlots { get; set; }
        public bool ShouldSaveRunInfos { get; set; }
        public bool ShouldSerialize { get; set; }
    }
}
