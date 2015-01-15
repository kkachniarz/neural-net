using LearningNN;
using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.Win32;
using Neural_Network.Plotting;
using OxyPlot;
using OxyPlot.Wpf;
using SharpNN;
using Shell.Containers;
using Shell.Enums;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Windows;
using System.Windows.Documents;

namespace Shell
{
    public partial class MainWindow : Window
    {
        public bool IsReady { get { return csvLines != null; } }

        private const double ERROR_SCALE = 1E6;
        private const double DISCARD_FACTOR = 0.2;
        private const int DISPLAY_LIMIT = 10;

        private int runsPerSettings = 1;
        private bool plotAgainstInput = false;

        List<DenseVector> csvLines;
        List<LearningSettings> settingsToRun = new List<LearningSettings>();

        /// <summary>
        /// Contains results of learning
        /// </summary>
        EngineResult engineResult;

        /// <summary>
        /// Contains configuration data for the learning engine
        /// </summary>
        EngineInitData eid;
        BackgroundWorker worker;

        private string dataSetPath;
        private string dataSetShortName;
        private string resultsDirectoryPath;
        private string innerResultsPath;
        private string paramsFileName = null;
        private string paramsInputText;

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

            string paramsFilePath = ReadFile(out shortName, csvDlg);
            paramsFileName = shortName;

            if (paramsFilePath == null)
                return;

            settingsToRun = FileManager.RetrieveParameters(paramsFilePath);
            paramsInputText = FileManager.ReadTextFile(paramsFilePath);
            LoadParametersLabel.Content = string.Format("{0} ({1})", shortName, settingsToRun.Count);
            ToggleAutomationRelatedSettings(false);
        }

        private void HandleUnloadParametersClick(object sender, RoutedEventArgs e)
        {
            UnloadParameters();
        }

        private void UnloadParameters()
        {
            settingsToRun = new List<LearningSettings>(); // to create a new reference (safe for background worker)
            paramsFileName = null;
            paramsInputText = "";
            ToggleAutomationRelatedSettings(true);
            LoadParametersLabel.Content = "...";
        }

        private void ToggleAutomationRelatedSettings(bool on)
        {
            Momentum.IsEnabled = on;
            LearningRate.IsEnabled = on;
            MaxIterations.IsEnabled = on;
            BadIterations.IsEnabled = on;
            ActivationCombobox.IsEnabled = on;
            LayersTextBox.IsEnabled = on;
        }

        private void ReadDataSet(object sender, RoutedEventArgs e)
        {
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".csv";
            csvDlg.Filter = "CSV documents (.csv)|*.csv";
            csvDlg.Title = "Select proper CSV file";
            dataSetPath = ReadFile(out dataSetShortName, csvDlg);

            if (dataSetPath == null)
                return;

            csvLines = FileManager.ReadDataFromCSV(dataSetPath);
            TrainSetLabel.Content = dataSetShortName;

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
            ToggleSensitiveGUIParts(false);
            SetResultsLink(false, null);
            ToggleAutomationRelatedSettings(true); // user can prepare params for the next run

            runsPerSettings = int.Parse(RunsTextBox.Text);

            bool bias = (YesNo)BiasCombobox.SelectedItem == YesNo.Yes;

            int pcaDimensions = 0;
            int.TryParse(PCA.Text, out pcaDimensions); // TODO: test

            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;

            PartIIProblemType problemType = csvLines[0].Count == 1 ? PartIIProblemType.CTS : PartIIProblemType.Stock;
            ReportingOptions reportingOptions = GetReportingOptions();

            if (settingsToRun.Count == 0)
            {
                settingsToRun = GetLearningSettingsFromUI();
            }

            SetPlottingMethod(problemType);

            eid = new EngineInitData();

            eid.ErrorScale = ERROR_SCALE;
            eid.DiscardWorstFactor = DISCARD_FACTOR;
            eid.TrainSetPercentage = trainSetPercentage;

            eid.CtsPrevValuesCount = 1; // always 1
            eid.OutputCount = outputCount;
            eid.InputCount = problemType == PartIIProblemType.CTS ? 1 : csvLines[0].Count - outputCount;
            if (pcaDimensions > 0)
            {
                eid.InputCount = Math.Min(pcaDimensions, eid.InputCount);
            }

            eid.PcaDimensions = pcaDimensions;
            eid.RunsPerSettings = runsPerSettings;

            eid.UseBiases = bias;
            eid.PlotAgainstInput = plotAgainstInput;

            eid.NetworkType = networkType;
            eid.ProblemType = problemType;

            eid.ReportingOptions = reportingOptions;

            eid.CsvLines = csvLines;
            eid.SettingsToRun = settingsToRun;

            eid.DataSetName = dataSetPath;
            eid.ParametersFileName = paramsFileName;

            ConfirmReportingSettings();
            CreateResultDirectories(DateTime.Now);

            worker = new BackgroundWorker();
            worker.DoWork += worker_DoWork;
            worker.RunWorkerCompleted += worker_RunWorkerCompleted;

            worker.RunWorkerAsync();
        }

        private void SetPlottingMethod(PartIIProblemType problemType)
        {
            if (problemType == PartIIProblemType.CTS)
            {
                plotAgainstInput = true;
            }
            else
            {
                plotAgainstInput = false;
            }
        }

        private void ToggleSensitiveGUIParts(bool on)
        {
            StartButton.IsEnabled = on;
            LoadParamsBtn.IsEnabled = on;
            UnloadParamsBtn.IsEnabled = on;
        }

        void worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                Title = "ERROR";
                MessageBox.Show(e.Error.ToString());
            }
            else
            {
                Title = "Saving results...";
                SaveResults();
            }

            CleanUpAfterWorkerCompleted();
        }

        private void CleanUpAfterWorkerCompleted()
        {
            ToggleSensitiveGUIParts(true);
            SetResultsLink(true, new Uri(resultsDirectoryPath));
            if (paramsFileName == null) // if params from file weren't used
            {
                settingsToRun.Clear();
            }

            eid = null;
        }

        private void SetResultsLink(bool on, System.Uri resultsUri)
        {
            ResultsLink.NavigateUri = resultsUri;
            ResultsLink.IsEnabled = on;
        }

        void worker_DoWork(object sender, DoWorkEventArgs e)
        {
            Engine engine = new Engine(eid, this);
            engineResult = engine.Run();
        }

        private void SaveResults()
        {
            List<AggregateResult> aggregates = new List<AggregateResult>();
            int lsID = 0;
            foreach (KeyValuePair<LearningSettings, List<SingleRunReport>> kvp in engineResult.ResultsBySettings)
            {
                lsID++;

                kvp.Value.Sort((a, b) => Math.Sign(a.LearningResult.TestSetError - b.LearningResult.TestSetError)); // sort by errror asc
                for (int i = 0; i < kvp.Value.Count; i++)
                {
                    kvp.Value[i].Name = string.Format("{0}-{1}", lsID, i + 1);
                    ProcessSingleResultEntry(kvp.Key, kvp.Value[i]);
                }

                aggregates.Add(new AggregateResult(kvp.Value, kvp.Key));
            }

            aggregates.Sort((a, b) => Math.Sign(a.AverageError - b.AverageError));
            SafeSerializeNetworkToFile(aggregates[0].BestReport.Network, string.Format("{0}_{1}_{2}", eid.NetworkType,
                Path.GetFileNameWithoutExtension(dataSetShortName), 
                aggregates[0].BestReport.LearningResult.TestSetError.ToString("E2")));
            SaveBatchReport(aggregates);
            ReactToResultsSaved();
        }

        private void SafeSerializeNetworkToFile(object net, string networkID)
        {
            if (!eid.ReportingOptions.ShouldSerialize)
            {
                return;
            }

            try
            {
                string path = Path.Combine(resultsDirectoryPath, string.Format("{0}.bin", networkID));
                IFormatter formatter = new BinaryFormatter();
                using (Stream stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    formatter.Serialize(stream, net);
                }
            }
            catch(Exception e)
            {
                MessageBox.Show("Error serializing network " + e.ToString());
            }
        }

        private void ReactToResultsSaved()
        {
            Title = "Done.";
        }

        /// <summary>
        /// Saves the main report, a comparison of all learning settings, to a text file.
        /// </summary>
        /// <param name="sortedAverages"></param>
        private void SaveBatchReport(List<AggregateResult> sortedAverages)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat(@"Data set: {0}
Params text (file name: {1}): 
{2}
Date {3}, {4}
Runs per settings: {5}, discarding {6} = {7} worst runs per each settings
",
                System.IO.Path.GetFileName(eid.DataSetName), paramsFileName ?? "",
                paramsInputText,
                DateTime.Now.ToLongDateString(), DateTime.Now.ToLongTimeString(),
                eid.RunsPerSettings, eid.DiscardWorstFactor.ToString("P1"), engineResult.WorstDiscardedCount);

            sb.AppendFormat(@"Network type: {0}, inputs: {1}, outputs: {2}
Outputs normalized within {3} - {4} of activation range.
Total time taken: {5}s.",
            eid.NetworkType.ToString(), eid.InputCount, eid.OutputCount,
            Normalizor.MARGIN_FACTOR.ToString("P1"), (1.0 - Normalizor.MARGIN_FACTOR).ToString("P1"),
            sortedAverages.Sum(a => a.AverageSecondsTaken * a.RunCount).ToString("F1"));
            sb.AppendLine();
            sb.AppendLine("-------------------------------------------------------------------");
            sb.AppendLine();

            foreach (AggregateResult ar in sortedAverages)
            {
                sb.AppendLine(ar.ToString());
                sb.AppendLine();
            }

            string reportName = "REPORT.txt";
            string reportPath = System.IO.Path.Combine(resultsDirectoryPath, reportName);
            using (FileStream fileStream = new FileStream(reportPath, FileMode.CreateNew))
            {
                using (StreamWriter sw = new StreamWriter(fileStream))
                {
                    sw.Write(sb.ToString());
                }
            }
        }

        /// <summary>
        /// Ask for confirmation if user requested a lot of windows.
        /// </summary>
        private void ConfirmReportingSettings()
        {
            if (eid.ReportingOptions.ShouldDisplayPlots && settingsToRun.Count * runsPerSettings >= DISPLAY_LIMIT)
            {
                MessageBoxResult result = MessageBox.Show("Display results for " + (settingsToRun.Count * runsPerSettings).ToString() + "?"
                    , "Confirmation", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result == MessageBoxResult.No)
                {
                    eid.ReportingOptions.ShouldDisplayPlots = false;
                }
            }
        }

        private ReportingOptions GetReportingOptions()
        {
            ReportingOptions options = new ReportingOptions();
            options.ShouldDisplayPlots = ShowPlotsCheckbox.IsChecked.Value;
            options.ShouldSavePlots = SavePlotsCheckbox.IsChecked.Value;
            options.ShouldSaveRunInfos = SaveRunInfosCheckbox.IsChecked.Value;
            options.ShouldSerialize = SerializeCheckbox.IsChecked.Value;
            return options;
        }

        private List<LearningSettings> GetLearningSettingsFromUI()
        {
            LearningSettings lSettings = new LearningSettings();
            string[] maxIters = GetSimulatedParamsTexts(LearningSettings.MaxIterationsTitle, MaxIterations.Text);
            string[] badIters = GetSimulatedParamsTexts(LearningSettings.BadIterationsTitle, 
                BadIterations.Text == ""? (int.MaxValue / 5).ToString() : BadIterations.Text);
            string[] lr = GetSimulatedParamsTexts(LearningSettings.LearningRateTitle, LearningRate.Text);
            string[] mnt = GetSimulatedParamsTexts(LearningSettings.MomentumTitle, Momentum.Text);
            ActivationType aType = (ActivationType)ActivationCombobox.SelectedItem;
            string aTypeText = "";
            if(aType == ActivationType.Unipolar) { aTypeText = "U"; }
            else if(aType == ActivationType.Bipolar) { aTypeText = "B"; }
            else { throw new Exception("Unrecognized activation type"); };

            string[] act = GetSimulatedParamsTexts(LearningSettings.ActivationFuncTitle, aTypeText);
            string[] hls = GetSimulatedParamsTexts(LearningSettings.HLTitle, LayersTextBox.Text);

            List<string[]> simulatedSplitLines = new List<string[]>()
                {
                    maxIters, badIters, lr, mnt, act, hls
                };

            StringBuilder inputText = new StringBuilder();
            simulatedSplitLines.ForEach(l => inputText.AppendLine(string.Join(",", l)));
            paramsInputText = inputText.ToString();
            return SettingsMixer.BuildSettings(simulatedSplitLines);
        }

        private string[] GetSimulatedParamsTexts(string title, string fieldText)
        {
            return FileManager.ParseParamsLine(string.Format("{0}: {1}", title, fieldText));
        }

        public void UpdateStatus(string text)
        {
            this.Title = text;
        }

        /// <summary>
        /// Save results of a single learning (a single run) - just 1 network, 1 learning.
        /// Saves error plot, regression plot, and info about the run.
        /// </summary>
        /// <param name="layersVal"></param>
        /// <param name="learningSettings"></param>
        /// <param name="report"></param>
        /// <param name="regressionPlot"></param>
        /// <param name="errorPlot"></param>
        /// <param name="network"></param>
        private void SaveSingleRunData(LearningSettings learningSettings,
            SingleRunReport report, PlotModel regressionPlot, PlotModel errorPlot, INetwork network) // could be refactored -> use MainWindow fields or create a class
        {
            DateTime time = report.Time;
            List<int> layersVal = RetrieveLayersVal(learningSettings);
            string prefix = report.Name;

            if (eid.ReportingOptions.ShouldSavePlots)
            {
                string regressionFileName = prefix + "_regression.png";
                string regressionSavePath = System.IO.Path.Combine(innerResultsPath, regressionFileName);
                using (FileStream fileStream = new FileStream(regressionSavePath, FileMode.CreateNew))
                {
                    PngExporter.Export(regressionPlot, fileStream, 900, 900, OxyColors.White);
                }

                string errorFileName = prefix + "_error.png";
                string errorSavePath = System.IO.Path.Combine(innerResultsPath, errorFileName);
                using (FileStream fileStream = new FileStream(errorSavePath, FileMode.CreateNew))
                {
                    PngExporter.Export(errorPlot, fileStream, 900, 900, OxyColors.White);
                }
            }

            if (eid.ReportingOptions.ShouldSaveRunInfos)
            {
                string infoFileName = prefix + "_info.txt";
                string infoSavePath = System.IO.Path.Combine(innerResultsPath, infoFileName);

                FileManager.SaveTextFile(infoSavePath,
                    GetResultInfo(learningSettings, report.LearningResult, layersVal, network, time));
            }
        }

        private List<int> RetrieveLayersVal(LearningSettings learningSettings)
        {
            List<int> layersVal = new List<int>() { eid.InputCount };
            layersVal.AddRange(learningSettings.HiddenNeuronCounts);
            layersVal.Add(eid.OutputCount);
            return layersVal;
        }

        private string GetResultInfo(LearningSettings settings, LearningResult result, List<int> neuronCounts, INetwork network, DateTime now)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(settings.ToString());
            sb.Append(now.ToLongDateString());
            sb.Append("  ");
            sb.AppendLine(now.ToLongTimeString());
            sb.AppendLine(System.IO.Path.GetFileName(eid.DataSetName));
            sb.AppendFormat("Layer counts: {0}\r\n", string.Join("-", neuronCounts));
            sb.AppendLine(result.ToString());
            return sb.ToString();
        }

        private void ProcessSingleResultEntry(LearningSettings settings, SingleRunReport result)
        {
            RegressionPlotBuilder regressionBuilder = new RegressionPlotBuilder();
            PlotModel regressionPlot = regressionBuilder.Build1DRegressionModel(result.TrainSet, result.TestSet, eid.PlotAgainstInput);
            ErrorPlotBuilder errorBuilder = new ErrorPlotBuilder(eid.ErrorScale);
            PlotModel errorPlot = errorBuilder.SetUpModel(result.LearningResult.MSEHistory);

            SaveSingleRunData(settings, result, regressionPlot, errorPlot, result.Network);

            if (eid.ReportingOptions.ShouldDisplayPlots)
            {
                DisplayResults(regressionPlot, errorPlot, result.LearningResult, result.Name);
            }
        }

        private void DisplayResults(PlotModel regressionPlot, PlotModel errorPlot, LearningResult learningResult, string identifier)
        {
            Window errorWindow = new NetworkErrorWindow(errorPlot, identifier);
            errorWindow.Show();
            Window regressionWindow = new RegressionWindow(regressionPlot, learningResult, identifier);
            regressionWindow.Show();
        }


        private void CreateResultDirectories(DateTime time)
        {
            string dirName = settingsToRun.Count + "x" + runsPerSettings.ToString() + "_" +
                time.ToLongDateString() + "_" + time.ToLongTimeString().Replace(":", "-");
            resultsDirectoryPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), dirName);
            Directory.CreateDirectory(resultsDirectoryPath);
            innerResultsPath = System.IO.Path.Combine(resultsDirectoryPath, "run_data");
            Directory.CreateDirectory(innerResultsPath);
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

        private void Hyperlink_RequestNavigate(object sender, System.Windows.Navigation.RequestNavigateEventArgs e)
        {
            Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri));
            e.Handled = true;
        }
    }
}
