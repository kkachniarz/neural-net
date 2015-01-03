using LearningNN;
using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.Win32;
using Neural_Network;
using Neural_Network.Plotting;
using OxyPlot;
using OxyPlot.Wpf;
using RecursiveNN;
using SharpNN;
using SharpNN.ActivationFunctions;
using SharpNN.Statistics;
using Shell.Enums;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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

namespace Neural_Network
{
    public partial class MainWindow : Window, ILearningStatus
    {
        public bool IsReady { get { return csvLines != null; } }

        private const double ERROR_SCALE = 1000.0;
        private int runCounter = 0;

        ReportingOptions reportingOptions;
        List<DenseVector> csvLines;
        List<LearningSettings> settingsToRun = new List<LearningSettings>();
        IDataSet testDataSet;
        IDataSet trainDataSet;
        ILearningStrategy learningStrategy;
        string dataSetPath;

        public MainWindow()
        {
            InitializeComponent();

            NetworkTypeCombobox.ItemsSource = Enum.GetValues(typeof(NetworkType)).Cast<NetworkType>();
            ProblemTypeCombobox.ItemsSource = Enum.GetValues(typeof(PartIIProblemType)).Cast<PartIIProblemType>();
            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>();
            BiasCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();

            ActivationCombobox.SelectedIndex = 0;
            BiasCombobox.SelectedIndex = 0;
        }

        private void ReadParameters(object sender, RoutedEventArgs e)
        {
            string shortName;
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".txt";
            csvDlg.Filter = "TXT documents (.txt)|*.txt";
            csvDlg.Title = "Select a .txt file containing network parameters";
            string paramsPath = ReadFile(out shortName, csvDlg);

            if (paramsPath == null)
                return;

            ToggleAutomationRelatedSettings(false);
            settingsToRun = FileManager.RetrieveParameters(paramsPath);
            LoadParametersLabel.Content = shortName;
            ShowPlotsCheckbox.IsChecked = false; // uncheck by default to avoid spamming windows
            // allow to unload parameters from UI
        }

        private void ToggleAutomationRelatedSettings(bool on)
        {
            LearningRate.IsEnabled = on;
            Momentum.IsEnabled = on;
            BadIterations.IsEnabled = on;
            MaxIterations.IsEnabled = on;
            // add more as more parameters are handled by automation
        }

        private void ReadDataSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".csv";
            csvDlg.Filter = "CSV documents (.csv)|*.csv";
            csvDlg.Title = "Select proper CSV file";
            dataSetPath = ReadFile(out shortName, csvDlg);

            if (dataSetPath == null)
                return;

            csvLines = FileManager.ReadDataFromCSV(dataSetPath);
            TrainSetLabel.Content = shortName;

            if (IsReady)
            {
                StartButton.IsEnabled = true;
            }
        }

        private string ReadFile(out string shortName, OpenFileDialog dialog)
        {
            var result = dialog.ShowDialog();

            if (result == true)
            {
                var fileName = dialog.FileName.ToString();
                shortName = dialog.SafeFileName;
                return fileName;
            }

            shortName = null;
            return null;
        }

        private void StartButtonClick(object sender, RoutedEventArgs e)
        {
            StartButton.IsEnabled = false;
            runCounter = 0;
            string[] layersText = LayersTextBox.Text.Split(new string[] { ",", " ", "-", "_", "." }, StringSplitOptions.RemoveEmptyEntries);
            bool bias = (YesNo)BiasCombobox.SelectedItem == YesNo.Yes;
            IActivation activation = ((ActivationFunction)ActivationCombobox.SelectedItem == ActivationFunction.Bipolar) ?
                (IActivation)new BipolarTanhActivation() : new UnipolarSigmoidActivation();

            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;
            int ctsPrevValuesCount = int.Parse(CTSPreviousValues.Text);

            PartIIProblemType problemType = (PartIIProblemType)ProblemTypeCombobox.SelectedItem;

            reportingOptions = GetReportingOptions();

            if (settingsToRun.Count == 0)
            {
                settingsToRun.Add(GetLearningSettingsFromUI());
            }            

            foreach (LearningSettings learningSettings in settingsToRun)
            {
                runCounter++;
                List<int> layersVal = new List<int>();
                foreach (var layer in layersText)
                {
                    layersVal.Add(int.Parse(layer)); // re-initialize layer counts -> Later layer counts should be also configurable during parameter sweeps.
                }

                if (problemType == PartIIProblemType.CTS)
                {
                    InitCTS(layersVal, trainSetPercentage, ctsPrevValuesCount);
                }
                else
                {
                    InitStock(layersVal, trainSetPercentage, outputCount);
                }

                layersVal.Add(outputCount);
                INetwork network = null;
                switch (networkType)
                {
                    case NetworkType.MLP:
                        network = new NeuralNetwork(activation, bias, layersVal.ToArray());
                        break;
                    case NetworkType.Jordan:
                        network = new RecursiveNetwork(RecursiveNetwork.Type.Jordan,
                        activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                        break;
                    case NetworkType.Elman:
                        network = new RecursiveNetwork(RecursiveNetwork.Type.Elman,
                        activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                        break;
                }

                NormalizeData(network, trainDataSet, testDataSet);
                CheckIfPerformPCA(network);
                learningStrategy = new VSetLearningStrategy(learningSettings);

                var learningResult = BackpropagationManager.Run(network, trainDataSet, testDataSet,
                    learningStrategy, this);

                NormalizeDataBack(network, trainDataSet, testDataSet);

                
                bool plotAgainstInput = false;
                if (problemType == PartIIProblemType.CTS)
                {
                    plotAgainstInput = true;
                }

                PlotModel regressionPlot = Build1DRegressionModel(trainDataSet, testDataSet, plotAgainstInput);
                ErrorPlotBuilder builder = new ErrorPlotBuilder(ERROR_SCALE);
                PlotModel errorPlot = builder.SetUpModel(learningResult.MSEHistory);
                if (reportingOptions.ShouldSave)
                {
                    SaveResultsToDisk(layersVal, learningSettings, learningResult, regressionPlot, errorPlot);
                }

                if (reportingOptions.ShouldDisplay)
                {
                    DisplayResults(regressionPlot, errorPlot, learningResult);                 
                }
            }

            StartButton.IsEnabled = true;
        }

        private static void DisplayResults(PlotModel regressionPlot, PlotModel errorPlot, LearningResult learningResult)
        {
            Window errorWindow = new NetworkErrorWindow(errorPlot);
            errorWindow.Show();
            Window regressionWindow = new RegressionWindow(regressionPlot, learningResult);
            regressionWindow.Show();
        }

        private void SaveResultsToDisk(List<int> layersVal, LearningSettings learningSettings, 
            LearningResult learningResult, PlotModel regressionPlot, PlotModel errorPlot)
        {
            DateTime now = DateTime.Now;

            string datePrefix = now.ToShortDateString() + "_" + now.ToLongTimeString().Replace(":", "-") + 
                string.Format("-{0}", now.Millisecond);
            string regressionFileName = datePrefix + "_regression.png";
            string regressionSavePath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), regressionFileName);
            using (FileStream fileStream = new FileStream(regressionSavePath, FileMode.CreateNew))
            {
                PngExporter.Export(regressionPlot, fileStream, 900, 900, OxyColors.White);
            }

            string errorFileName = datePrefix + "_error.png";
            string errorSavePath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), errorFileName);
            using (FileStream fileStream = new FileStream(errorSavePath, FileMode.CreateNew))
            {
                PngExporter.Export(errorPlot, fileStream, 900, 900, OxyColors.White);
            }

            string infoFileName = datePrefix + "_info.txt";
            string infoSavePath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), infoFileName);
            // TODO: calculating test set error, calculating how often the direction of change is predicted correctly
            // TODO: allow many executions for each configuration to calculate averages.
            // TODO: save execution data as a "capsule" -> later we can find the best score in a batch, the best parameters, compute averages etc.

            FileManager.SaveLearningInfo(infoSavePath,
                GetResultInfo(learningSettings, learningResult, layersVal, now));
        }

        private string GetResultInfo(LearningSettings settings, LearningResult result, List<int> neuronCounts, DateTime now)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(settings.ToString());
            sb.Append(now.ToLongDateString());
            sb.Append("  ");
            sb.AppendLine(now.ToLongTimeString());
            sb.AppendFormat("Iterations executed: {0}\r\n", result.IterationsExecuted);
            sb.AppendLine(System.IO.Path.GetFileName(dataSetPath));
            sb.AppendFormat("Layer counts: {0}\r\n", string.Join("-", neuronCounts));
            sb.AppendFormat("Error on validation set: {0}\r\n", result.MSEHistory[result.MSEHistory.Count - 1]);
            sb.AppendFormat("Error on test set: {0}\r\n", result.TestSetError);
            sb.AppendFormat("Direction guessed on test set: {0}\r\n", result.TestSetDirectionGuessed);
            return sb.ToString();
        }

        private ReportingOptions GetReportingOptions()
        {
            ReportingOptions options = new ReportingOptions();
            options.ShouldDisplay = ShowPlotsCheckbox.IsChecked.Value;
            options.ShouldSave = SavePlotsCheckbox.IsChecked.Value;
            return options;
        }

        private LearningSettings GetLearningSettingsFromUI()
        {
            LearningSettings learnSettings = new LearningSettings();
            learnSettings.MaxIterations = int.Parse(MaxIterations.Text);
            learnSettings.BadIterations = BadIterations.Text == "" ? learnSettings.MaxIterations : int.Parse(BadIterations.Text);
            learnSettings.LearningRate = double.Parse(LearningRate.Text);
            learnSettings.Momentum = double.Parse(Momentum.Text);
            learnSettings.ValidationSetSize = 0.2f;

            return learnSettings;
        }

        private void CheckIfPerformPCA(INetwork network)
        {
            int dimensionPca;

            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                LearningNN.PCA.Run(trainDataSet, dimensionPca, network.Activation.MinValue, network.Activation.MaxValue);
                LearningNN.PCA.Run(testDataSet, dimensionPca, network.Activation.MinValue, network.Activation.MaxValue);
            }
        }

        private static void NormalizeData(INetwork network, IDataSet trainData, IDataSet testData)
        {
            var extremums = DataExtremumsForNetwork.Merge(trainData.Extremums, testData.Extremums);

            trainData.Normalize(network.Activation.MinValue, network.Activation.MaxValue, extremums);
            testData.Normalize(network.Activation.MinValue, network.Activation.MaxValue, extremums);
        }

        private static void NormalizeDataBack(INetwork network, IDataSet trainData, IDataSet testData)
        {
            testData.NormalizeBack();
            trainData.NormalizeBack();
        }

        private void InitCTS(List<int> layersVal, float trainSetPercentage, int historyLength)
        {
            int dimensionPca;
            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                layersVal.Insert(0, Math.Min(dimensionPca, historyLength));
            }
            else
            {
                layersVal.Insert(0, historyLength); // network needs as many inputs as many historical values we feed it.
            }

            int trainSetEndIndex = (int)(trainSetPercentage * csvLines.Count);
            List<DenseVector> chaoticValues = csvLines; // no need for further parsing

            List<DenseVector> trainValues = chaoticValues.ExtractList(0, trainSetEndIndex);
            List<DenseVector> testValues = chaoticValues.ExtractList(trainSetEndIndex, chaoticValues.Count);

            trainDataSet = new ChaoticDataSet(trainValues, historyLength, 0);
            if (trainSetEndIndex >= chaoticValues.Count - 1)
            {
                testDataSet = trainDataSet.Clone();
            }
            else
            {
                testDataSet = new ChaoticDataSet(testValues, historyLength, trainSetEndIndex);
            }
        }

        private void InitStock(List<int> layersVal, float trainSetPercentage, int outputCount)
        {
            int dimensionPca;
            int displayInputCount = csvLines[0].Count - outputCount; // this way we can calculate number of display inputs

            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                layersVal.Insert(0, Math.Min(dimensionPca, displayInputCount));
            }
            else
            {
                layersVal.Insert(0, displayInputCount);
            }
            
            
            int trainSetEndIndex = (int)(trainSetPercentage * csvLines.Count);

            List<DenseVector> allInputs = csvLines.Select(v => v.CreateSubVector(0, displayInputCount)).ToList();
            List<DenseVector> allOutputs = csvLines.Select(v => v.CreateSubVector(displayInputCount, outputCount)).ToList();

            trainDataSet = new StockDataSet(allInputs.ExtractList(0, trainSetEndIndex), 
                allOutputs.ExtractList(0, trainSetEndIndex), 0);

            if (trainSetEndIndex >= allInputs.Count - 1)
            {
                testDataSet = trainDataSet.Clone();
            }
            else
            {
                testDataSet = new StockDataSet(allInputs.ExtractList(trainSetEndIndex, csvLines.Count),
                    allOutputs.ExtractList(trainSetEndIndex, csvLines.Count), trainSetEndIndex);
            }
        }

        private void AppendCSVile(string path, CasesData data)
        {
            var output = new List<DenseVector>();

            for (int i = 0; i < data.CasesCount; i++)
            {
                output.Add(
                    data.ProblemType == ProblemType.Classification ?
                    data.GetClasificationOutput(i) :
                    data.GetIdealOutput(i));
            }

            FileManager.AppendDataToCSV(path, output);
        }

        private PlotModel Build1DRegressionModel(IDataSet trainingSet, IDataSet testSet, bool plotAgainstInput) // if plotAgainstInput is true, use input as X axis, not time
        {
            List<RegressionPoint> trainPoints = new List<RegressionPoint>();
            List<RegressionPoint> testIdealPoints = new List<RegressionPoint>();
            List<RegressionPoint> networkAnswers = new List<RegressionPoint>();
            Func<Pattern, double> patternToDouble;
            if(plotAgainstInput)
            {
                patternToDouble = p => p.Input[0];
            }
            else
            {
               patternToDouble = p => p.TimeIndex;
            }

            foreach(Pattern p in trainingSet.EnumeratePatterns())
            {
                trainPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
            }

            foreach(Pattern p in testSet.EnumeratePatterns())
            {
                testIdealPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
                networkAnswers.Add(new RegressionPoint(patternToDouble(p), p.NetworkAnswer.At(0)));
            }

            RegressionPlotBuilder builder = new RegressionPlotBuilder();
            PlotModel regressionPlotModel = builder.SetUpModel(trainPoints, testIdealPoints, networkAnswers);
            return regressionPlotModel;
        }

        public void SetStatusText(string text)
        {           
            this.Title = string.Format("{0} / {1}: {2}", runCounter, settingsToRun.Count, text);
        }

        private class ReportingOptions
        {
            public bool ShouldDisplay { get; set; }
            public bool ShouldSave { get; set; }
        }
    }

    public enum YesNo
    {
        Yes,
        No
    }

    public enum ActivationFunction
    {
        Unipolar,
        Bipolar
    }

    public enum PartIIProblemType
    {
        CTS,
        Stock
    }
}
