﻿using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SharpNN;
using SharpNN.Statistics;
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

namespace Shell.Plotting
{
    /// <summary>
    /// Interaction logic for RegressionWindow.xaml
    /// </summary>
    public partial class RegressionWindow : Window
    {
        public RegressionWindow(PlotModel plotModel, LearningResult learningResult, string identifier)
        {
            InitializeComponent();
            RegressionPlot.Model = plotModel;
            this.Title = string.Format("{0}: TS Dir {1} TS Err {2}",  identifier,
                learningResult.MPVer1.ToString("E2"), learningResult.TestSetError.ToString("E2"));
        }
    }
}
