using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace PlottingTest
{
    /// <summary>
    /// Interaction logic for ClassificationWindow.xaml
    /// </summary>
    public partial class ClassificationWindow : Window
    {
        private PlotModel plotModel;

        public ClassificationWindow(List<ClassificationPoint> learned, List<ClassificationPoint> ideal)
        {
            InitializeComponent();
            plotModel = new PlotModel();
            SetUpModel(learned, ideal);
            ClassificationPlot.Model = plotModel;
        }

        private void SetUpModel(List<ClassificationPoint> learned, List<ClassificationPoint> ideal)
        {
            Dictionary<int, ScatterSeries> learnedSeries = new Dictionary<int, ScatterSeries>();
            Dictionary<int, ScatterSeries> idealSeries = new Dictionary<int, ScatterSeries>();
            int classCount = Math.Max(learned.Max(x => x.ClassID) - learned.Min(x => x.ClassID) + 1,
                ideal.Max(x => x.ClassID) - ideal.Min(x => x.ClassID) + 1);
            for (int i = 0; i < classCount; i++)
            {
                learnedSeries[i] = GetSeries(i, true);
                idealSeries[i] = GetSeries(i, false);
            }

            learned.ForEach(p => learnedSeries[p.ClassID].Points.Add(CreateScatterPoint(p)));
            ideal.ForEach(p => idealSeries[p.ClassID].Points.Add(CreateScatterPoint(p)));

            double minX = Math.Min(learned.Min(p => p.X), ideal.Min(p => p.X));
            double maxX = Math.Max(learned.Max(p => p.X), ideal.Max(p => p.X));
            double minY = Math.Min(learned.Min(p => p.Y), ideal.Min(p => p.Y));
            double maxY = Math.Max(learned.Max(p => p.Y), ideal.Max(p => p.Y));

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

            for (int i = 0; i < classCount; i++)
            {
                plotModel.Series.Add(learnedSeries[i]);
            }
            for (int i = 0; i < classCount; i++)
            {
                plotModel.Series.Add(idealSeries[i]); // layer by layer
            }
        }

        private ScatterPoint CreateScatterPoint(ClassificationPoint classificationPoint)
        {
            return new ScatterPoint(classificationPoint.X, classificationPoint.Y);
        }

        private ScatterSeries GetSeries(int classID, bool isLearned)
        {
            if (isLearned)
            {
                return new ScatterSeries { MarkerType = MarkerType.Triangle, MarkerSize = 5.0, MarkerFill = GetColorForClass(classID, isLearned) };
            }

            return new ScatterSeries { MarkerType = MarkerType.Square, MarkerSize = 5.0, MarkerFill = GetColorForClass(classID, isLearned) };
        }

        private OxyColor GetColorForClass(int classID, bool isLearned)
        {
            byte alpha = (byte) (isLearned? 255 : 70);
            switch (classID)
            {
                case 0:
                    return OxyColor.FromAColor(alpha, OxyColors.Red);
                case 1:
                    return OxyColor.FromAColor(alpha, OxyColors.Blue);
                case 2:
                    return OxyColor.FromAColor(alpha, OxyColors.Green);
                case 3:
                    return OxyColor.FromAColor(alpha, OxyColors.SaddleBrown); 
                case 4:
                    return OxyColor.FromAColor(alpha, OxyColors.Indigo);
                case 5:
                    return OxyColor.FromAColor(alpha, OxyColors.Pink);
                case 6:
                    return OxyColor.FromAColor(alpha, OxyColors.Silver);
                case 7:
                    return OxyColor.FromAColor(alpha, OxyColors.SpringGreen);
                case 8:
                    return OxyColor.FromAColor(alpha, OxyColors.Purple);
                case 9:
                    return OxyColor.FromAColor(alpha, OxyColors.Yellow);                    
            }

            return OxyColor.FromAColor(alpha, OxyColors.Black);
        }
    }
}
