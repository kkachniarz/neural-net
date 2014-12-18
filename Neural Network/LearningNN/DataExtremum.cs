using System.Collections.Generic;

namespace LearningNN
{
    public class DataExtremum
    {
        public double MinValue { get; private set; }
        public double MaxValue { get; private set; }
        public Normalizor Normalizor { get; set; }

        private List<double> data;

        public DataExtremum(List<double> data)
        {
            MinValue = double.MaxValue;
            MaxValue = double.MinValue;

            this.data = data;
            Update();
        }

        public void Update()
        {
            foreach (var v in data)
            {
                if (v < MinValue)
                {
                    MinValue = v;
                }

                if (v > MaxValue)
                {
                    MaxValue = v;
                }
            }

            Normalizor = null;
        }

        public static DataExtremum Merge(DataExtremum x, DataExtremum y)
        {
            var data = new List<double>();
            data.AddRange(x.data);
            data.AddRange(y.data);
            return new DataExtremum(data);
        }
    }
}
