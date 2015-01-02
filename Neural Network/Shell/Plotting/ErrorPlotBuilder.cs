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
            var linearAxis1 = new LinearAxis();
            linearAxis1.TickStyle = TickStyle.Outside;
            linearAxis1.Position = AxisPosition.Left;
            linearAxis1.Minimum = 0;
            linearAxis1.Maximum = trainingSetErrors.Max() * 1.1 * errorScale;
            linearAxis1.Title = string.Format("Error x {0}", errorScale);
            plotModel.Axes.Add(linearAxis1);
            var linearAxis2 = new LinearAxis();
            linearAxis2.Position = AxisPosition.Bottom;
            linearAxis2.TickStyle = TickStyle.Outside;
            linearAxis2.Minimum = 0;
            linearAxis2.Maximum = trainingSetErrors.Count;
            linearAxis2.Title = "Iteration";
            plotModel.Axes.Add(linearAxis2);

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
