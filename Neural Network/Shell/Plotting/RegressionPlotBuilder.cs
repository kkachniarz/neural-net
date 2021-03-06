﻿using LearningNN.DataSet;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SharpNN.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shell.Plotting
{
    public class RegressionPlotBuilder
    {
        public PlotModel Build1DRegressionModel(IDataSet trainingSet, IDataSet testSet, bool plotAgainstInput) // if plotAgainstInput is true, use input as X axis, not time
        {
            List<RegressionPoint> trainPoints = new List<RegressionPoint>();
            List<RegressionPoint> testIdealPoints = new List<RegressionPoint>();
            List<RegressionPoint> networkAnswers = new List<RegressionPoint>();
            Func<Pattern, double> patternToDouble;
            if (plotAgainstInput)
            {
                patternToDouble = p => p.Input[0];
            }
            else
            {
                patternToDouble = p => p.TimeIndex;
            }

            foreach (Pattern p in trainingSet.EnumeratePatterns())
            {
                trainPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
            }

            foreach (Pattern p in testSet.EnumeratePatterns())
            {
                testIdealPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
                networkAnswers.Add(new RegressionPoint(patternToDouble(p), p.NetworkAnswer.At(0)));
            }

            PlotModel regressionPlotModel = SetUpModel(trainPoints, testIdealPoints, networkAnswers);
            return regressionPlotModel;
        }

        private PlotModel SetUpModel(List<RegressionPoint> trainingPoints, List<RegressionPoint> idealAnswer, List<RegressionPoint> networkAnswer)
        {
            PlotModel plotModel = new PlotModel();
            trainingPoints = trainingPoints.OrderBy(a => a.X).ToList();
            idealAnswer = idealAnswer.OrderBy(a => a.X).ToList();
            networkAnswer = networkAnswer.OrderBy(a => a.X).ToList();

            var series1 = new LineSeries { Title = "Training set", MarkerType = MarkerType.Circle, MarkerSize = 1.0, 
               StrokeThickness = 0.2, Color = OxyColors.LightBlue, MarkerFill = OxyColors.DarkCyan};
            trainingPoints.ForEach(x => series1.Points.Add(CreateDataPoint(x)));
            var series2 = new LineSeries { Title = "Test set", MarkerType = MarkerType.Circle, MarkerSize = 1.5, 
                 StrokeThickness = 0.5, Color = OxyColors.Navy, MarkerFill = OxyColors.Black };
            idealAnswer.ForEach(x => series2.Points.Add(CreateDataPoint(x)));
            var series3 = new LineSeries { Title = "Network answer", MarkerType = MarkerType.Diamond, MarkerSize = 1.5, 
                StrokeThickness = 1.0, Color = OxyColors.Orange, MarkerFill = OxyColors.Red };
            networkAnswer.ForEach(x => series3.Points.Add(CreateDataPoint(x)));

            double minX = Math.Min(trainingPoints.Min(p => p.X), networkAnswer.Concat(idealAnswer).Min(p => p.X));
            double maxX = Math.Max(trainingPoints.Max(p => p.X), networkAnswer.Concat(idealAnswer).Max(p => p.X));
            double minY = Math.Min(trainingPoints.Min(p => p.Y), networkAnswer.Concat(idealAnswer).Min(p => p.Y));
            double maxY = Math.Max(trainingPoints.Max(p => p.Y), networkAnswer.Concat(idealAnswer).Max(p => p.Y));

            var linearAxis1 = new LinearAxis();
            linearAxis1.TickStyle = TickStyle.Outside;
            linearAxis1.Position = AxisPosition.Left;
            linearAxis1.Minimum = minY;
            linearAxis1.Maximum = maxY;
            linearAxis1.Title = "Y";
            plotModel.Axes.Add(linearAxis1);

            var linearAxis2 = new LinearAxis();
            linearAxis2.Position = AxisPosition.Bottom;
            linearAxis2.TickStyle = TickStyle.Outside;
            linearAxis2.Minimum = minX;
            linearAxis2.Maximum = maxX;
            linearAxis2.Title = "X";
            plotModel.Axes.Add(linearAxis2);

            plotModel.Series.Add(series1);
            plotModel.Series.Add(series2);
            plotModel.Series.Add(series3);
            return plotModel;
        }

        private ScatterPoint CreateScatterPoint(RegressionPoint regressionDataPoint)
        {
            return new ScatterPoint(regressionDataPoint.X, regressionDataPoint.Y);
        }

        private DataPoint CreateDataPoint(RegressionPoint regressionDataPoint)
        {
            return new DataPoint(regressionDataPoint.X, regressionDataPoint.Y);
        }
    }
}
