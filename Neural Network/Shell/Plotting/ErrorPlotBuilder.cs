using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Plotting
{
    public class ErrorPlotBuilder
    {
        private readonly double errorScale;

        public ErrorPlotBuilder(double errorScale)
        {
            this.errorScale = errorScale;
        }

        public PlotModel SetUpModel(List<double> validationSetErrors, List<double> testSetErrors = null)
        {
            PlotModel plotModel = new PlotModel();
            var errorAxis = new LogarithmicAxis();
            errorAxis.TickStyle = TickStyle.Outside;
            errorAxis.Position = AxisPosition.Left;
            errorAxis.Maximum = validationSetErrors.Max() * 1.1 * errorScale;
            double errMin = validationSetErrors.Min() * errorScale;
            errorAxis.Minimum = Math.Min(1.0, errMin);
            errorAxis.Title = string.Format("Error x {0}", errorScale.ToString("E0"));
            errorAxis.StringFormat = "E0";
            errorAxis.MajorGridlineStyle = LineStyle.Dot;

            plotModel.Axes.Add(errorAxis);

            var iterationAxis = new LinearAxis();
            iterationAxis.Position = AxisPosition.Bottom;
            iterationAxis.TickStyle = TickStyle.Outside;
            iterationAxis.Minimum = 0;
            iterationAxis.Maximum = validationSetErrors.Count;
            iterationAxis.Title = "Iteration";
            plotModel.Axes.Add(iterationAxis);

            var series1 = new LineSeries { Title = "Validation set error", MarkerType = MarkerType.None };

            series1.Points.AddRange(CreateDataPoints(validationSetErrors));
            plotModel.Series.Add(series1);
            if (testSetErrors != null)
            {
                var series2 = new LineSeries { Title = "Test set error", MarkerType = MarkerType.None };
                series2.Points.AddRange(CreateDataPoints(testSetErrors));
                plotModel.Series.Add(series2);
            }

            return plotModel;
        }

        private List<DataPoint> CreateDataPoints(List<double> mseHistory)
        {
            List<DataPoint> points = new List<DataPoint>();
            for (int i = 0; i < mseHistory.Count; i++)
            {
                points.Add(new DataPoint(i, mseHistory[i] * errorScale));
            }
            return points;
        }
    }
}
