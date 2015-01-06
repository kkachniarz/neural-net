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

        public PlotModel SetUpModel(List<double> trainingSetErrors, List<double> validationSetErrors = null)
        {
            PlotModel plotModel = new PlotModel();
            var errorAxis = new LogarithmicAxis();
            errorAxis.TickStyle = TickStyle.Outside;
            errorAxis.Position = AxisPosition.Left;
            errorAxis.Minimum = 0;
            errorAxis.Maximum = trainingSetErrors.Max() * 1.1 * errorScale;
            errorAxis.Title = string.Format("Error x {0}", errorScale.ToString("E0"));

            plotModel.Axes.Add(errorAxis);
            var iterationAxis = new LinearAxis();
            iterationAxis.Position = AxisPosition.Bottom;
            iterationAxis.TickStyle = TickStyle.Outside;
            iterationAxis.Minimum = 0;
            iterationAxis.Maximum = trainingSetErrors.Count;
            iterationAxis.Title = "Iteration";
            plotModel.Axes.Add(iterationAxis);

            var series1 = new LineSeries { Title = "Training set error", MarkerType = MarkerType.None };

            series1.Points.AddRange(CreateDataPoints(trainingSetErrors));
            plotModel.Series.Add(series1);
            if (validationSetErrors != null)
            {
                var series2 = new LineSeries { Title = "Validation set error", MarkerType = MarkerType.None };
                series2.Points.AddRange(CreateDataPoints(validationSetErrors));
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
