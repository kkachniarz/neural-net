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

namespace Shell
{
    public partial class MainWindow : Window, ILearningStatus
    {
        public bool IsReady { get { return csvLines != null; } }

        private const double ERROR_SCALE = 1000.0;
        private const int DISPLAY_LIMIT = 10;
        private int runCounter = 0;
        private int runsPerSettings = 1;         
        private bool plotAgainstInput = false;

        private Dictionary<LearningSettings, List<SingleRunReport>> resultsBySettings = new Dictionary<LearningSettings, List<SingleRunReport>>();
        
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
            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationType)).Cast<ActivationType>();
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
            runsPerSettings = int.Parse(RunsTextBox.Text);
            string[] layersText = LayersTextBox.Text.Split(new string[] { ",", " ", "-", "_", "." }, StringSplitOptions.RemoveEmptyEntries);
            bool bias = (YesNo)BiasCombobox.SelectedItem == YesNo.Yes;

            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;
            int ctsPrevValuesCount = 1; // int.Parse(CTSPreviousValues.Text); <- obsolete, never used (CT series depend only on 1 previous value...)
            PartIIProblemType problemType = csvLines[0].Count == 1 ? PartIIProblemType.CTS : PartIIProblemType.Stock;
            reportingOptions = GetReportingOptions();

            if (settingsToRun.Count == 0)
            {
                settingsToRun.Add(GetLearningSettingsFromUI());
            }

            if (problemType == PartIIProblemType.CTS)
            {
                plotAgainstInput = true;
            }

            ConfirmReportingSettings();

            foreach (LearningSettings learningSettings in settingsToRun)
            {
                resultsBySettings[learningSettings] = new List<SingleRunReport>();
                for (int i = 0; i < runsPerSettings; i++) // repeat several times to average out the results
                {
                    runCounter++;
                    List<int> layersVal = new List<int>();
                    foreach (var layer in layersText)
                    {
                        layersVal.Add(int.Parse(layer)); // re-initialize layer counts -> TODO: Later layer counts should be also configurable in params file / learning settings
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
                            network = new NeuralNetwork(learningSettings.Activation, bias, layersVal.ToArray());
                            break;
                        case NetworkType.Jordan:
                            network = new RecursiveNetwork(RecursiveNetwork.Type.Jordan,
                            learningSettings.Activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                            break;
                        case NetworkType.Elman:
                            network = new RecursiveNetwork(RecursiveNetwork.Type.Elman,
                            learningSettings.Activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                            break;
                    }

                    NormalizeData(network, trainDataSet, testDataSet);
                    CheckIfPerformPCA(network);
                    learningStrategy = new VSetLearningStrategy(learningSettings);

                    var learningResult = BackpropagationManager.Run(network, trainDataSet, testDataSet,
                        learningStrategy, this);

                    NormalizeDataBack(network, trainDataSet, testDataSet);
                    resultsBySettings[learningSettings].Add(
                        new SingleRunReport(learningResult, trainDataSet, testDataSet, network, layersVal, DateTime.Now));

                }
            }

            SaveResults();
            StartButton.IsEnabled = true;
        }

        private void ConfirmReportingSettings()
        {
            if (reportingOptions.ShouldDisplay && settingsToRun.Count * runsPerSettings >= DISPLAY_LIMIT)
            {
                MessageBoxResult result = MessageBox.Show("Display results for " + (settingsToRun.Count * runsPerSettings).ToString() + "?"
                    , "Confirmation", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result == MessageBoxResult.No)
                {
                    reportingOptions.ShouldDisplay = false;
                }
            }
        }

        private void SaveResults()
        {
            List<AggregateResult> aggregates = new List<AggregateResult>();
            int lsID = 0;
            foreach(KeyValuePair<LearningSettings, List<SingleRunReport>> kvp in resultsBySettings)
            {
                lsID++;

                for(int i = 0; i < kvp.Value.Count; i++) 
                {
                    kvp.Value[i].Name = string.Format("{0}-{1}", lsID, i + 1);
                    ProcessSingleResultEntry(kvp.Key, kvp.Value[i]);
                }

                aggregates.Add(new AggregateResult(kvp.Value, kvp.Key));
            }

            aggregates.Sort((a, b) => Math.Sign(a.AverageError - b.AverageError));
            SaveBatchReport(aggregates);
        }

        private void ProcessSingleResultEntry(LearningSettings settings, SingleRunReport result)
        {
            PlotModel regressionPlot = Build1DRegressionModel(result.TrainSet, result.TestSet, plotAgainstInput);
            ErrorPlotBuilder builder = new ErrorPlotBuilder(ERROR_SCALE);
            PlotModel errorPlot = builder.SetUpModel(result.LearningResult.MSEHistory);
            if (reportingOptions.ShouldSave)
            {
                SaveResultsToDisk(result.LayersVal, settings, result, regressionPlot, errorPlot, result.Network);
            }

            if (reportingOptions.ShouldDisplay)
            {
                DisplayResults(regressionPlot, errorPlot, result.LearningResult);
            }
        }

        private void DisplayResults(PlotModel regressionPlot, PlotModel errorPlot, LearningResult learningResult)
        {
            Window errorWindow = new NetworkErrorWindow(errorPlot);
            errorWindow.Show();
            Window regressionWindow = new RegressionWindow(regressionPlot, learningResult);
            regressionWindow.Show();
        }

        private void SaveResultsToDisk(List<int> layersVal, LearningSettings learningSettings, 
            SingleRunReport report, PlotModel regressionPlot, PlotModel errorPlot, INetwork network) // could be refactored -> use MainWindow fields or create a class
        {
            DateTime time = report.Time;

            string datePrefix = time.ToLongTimeString().Replace(":", "-") + "_" + report.Name;
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
                GetResultInfo(learningSettings, report.LearningResult, layersVal, network, time));
        }

        private string GetResultInfo(LearningSettings settings, LearningResult result, List<int> neuronCounts, INetwork network, DateTime now)
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

        private void SaveBatchReport(List<AggregateResult> sortedAverages)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Data set: {0}  Date {1}, time {2}\r\n", System.IO.Path.GetFileName(dataSetPath),
                DateTime.Now.ToLongDateString(), DateTime.Now.ToLongTimeString());
            sb.AppendLine();

            foreach(AggregateResult ar in sortedAverages)
            {
                sb.AppendLine(ar.ToString());
                sb.AppendLine();
            }

            DateTime time = DateTime.Now;
            string datePrefix = time.ToShortDateString() + "_" + time.ToLongTimeString().Replace(":", "-") +
            string.Format("-{0}", time.Millisecond);
            string reportName = datePrefix + "_REPORT.txt";
            string reportPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), reportName);
            using (FileStream fileStream = new FileStream(reportPath, FileMode.CreateNew))
            {
                using(StreamWriter sw = new StreamWriter(fileStream))
                {
                    sw.Write(sb.ToString());
                }
            }
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
            LearningSettings lSettings = new LearningSettings();
            lSettings.MaxIterations = int.Parse(MaxIterations.Text);
            lSettings.BadIterations = BadIterations.Text == "" ? lSettings.MaxIterations : int.Parse(BadIterations.Text);
            lSettings.LearningRate = double.Parse(LearningRate.Text);
            lSettings.Momentum = double.Parse(Momentum.Text);
            lSettings.ValidationSetSize = 0.2f;
            lSettings.Activation = ((ActivationType)ActivationCombobox.SelectedItem == ActivationType.Bipolar) ?
                (IActivation)new BipolarTanhActivation() : new UnipolarSigmoidActivation();

            return lSettings;
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
            this.Title = string.Format("{0} / {1}: {2}", runCounter, settingsToRun.Count * runsPerSettings, text);
        }

        private class ReportingOptions
        {
            public bool ShouldDisplay { get; set; }
            public bool ShouldSave { get; set; }
        }
    }
}
