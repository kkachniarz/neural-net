using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.Statistics
{
    public class ClassificationPoint
    {
        public int ClassID { get; set; }
        public double X { get; set; }
        public double Y { get; set; }

        public ClassificationPoint(double x, double y, int classID)
        {
            this.ClassID = classID;
            this.X = x;
            this.Y = y;
        }
    }
}
