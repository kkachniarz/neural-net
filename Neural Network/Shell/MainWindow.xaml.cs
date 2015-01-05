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
using Shell.Containers;
using Shell.Enums;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.ComponentModel;
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
    public partial class MainWindow : Window
    {
        public bool IsReady { get { return csvLines != null; } }

        private const double ERROR_SCALE = 1000.0;
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
        private string parametersFileName = "";
        private string resultsDirectoryPath;
        private string innerResultsPath;
        private string paramsFileText;

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
            parametersFileName = shortName;

            if (paramsFilePath == null)
                return;

            settingsToRun = FileManager.RetrieveParameters(paramsFilePath);
            paramsFileText = FileManager.ReadTextFile(paramsFilePath);
            LoadParametersLabel.Content = string.Format("{0} ({1})", shortName, settingsToRun.Count);
            ToggleAutomationRelatedSettings(false);
        }

        private void UnloadParameters(object sender, RoutedEventArgs e)
        {
            settingsToRun = new List<LearningSettings>(); // to create a new reference (safe for background worker)
            parametersFileName = "(not used)";
            paramsFileText = "";
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
                settingsToRun.Add(GetLearningSettingsFromUI());
            }

            if (problemType == PartIIProblemType.CTS)
            {
                plotAgainstInput = true;
            }

            eid = new EngineInitData();

            eid.ErrorScale = ERROR_SCALE;
            eid.DiscardWorstFactor = DISCARD_FACTOR;
            eid.TrainSetPercentage = trainSetPercentage;

            eid.CtsPrevValuesCount = 1; // always 1
            eid.OutputCount = outputCount;
            eid.InputCount = problemType == PartIIProblemType.CTS? 1 : csvLines[0].Count - outputCount;
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
            eid.ParametersFileName = parametersFileName;

            ConfirmReportingSettings();
            CreateResultDirectories(DateTime.Now);

            worker = new BackgroundWorker();
            worker.DoWork += worker_DoWork;
            worker.RunWorkerCompleted += worker_RunWorkerCompleted;

            worker.RunWorkerAsync();
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
                Title = "Done.";
                SaveResults();
            }

            StartButton.IsEnabled = true;
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
            SaveBatchReport(aggregates);
        }

        /// <summary>
        /// Saves the main report, a comparison of all learning settings, to a text file.
        /// </summary>
        /// <param name="sortedAverages"></param>
        private void SaveBatchReport(List<AggregateResult> sortedAverages)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat(@"Data set: {0}
Params file: 
{1}
Date {2}, {3}
Runs per settings: {4}, discarding {5}% = {6} worst runs per each settings
Network type: {7}, inputs: {8}, outputs: {9}",
                System.IO.Path.GetFileName(eid.DataSetName), paramsFileText,
                DateTime.Now.ToLongDateString(), DateTime.Now.ToLongTimeString(),
                eid.RunsPerSettings, (eid.DiscardWorstFactor * 100).ToString("F1"), engineResult.WorstDiscardedCount,
                eid.NetworkType.ToString(), eid.InputCount, eid.OutputCount);
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
            lSettings.SetParamByTitle("HL", LayersTextBox.Text);

            return lSettings;
        }

        public void UpdateStatus(string text)
        {
            this.Title = text;
        }

        private void DisplayResults(PlotModel regressionPlot, PlotModel errorPlot, LearningResult learningResult)
        {
            Window errorWindow = new NetworkErrorWindow(errorPlot);
            errorWindow.Show();
            Window regressionWindow = new RegressionWindow(regressionPlot, learningResult);
            regressionWindow.Show();
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
            sb.AppendFormat("Iterations executed: {0}\r\n", result.IterationsExecuted);
            sb.AppendLine(System.IO.Path.GetFileName(eid.DataSetName));
            sb.AppendFormat("Layer counts: {0}\r\n", string.Join("-", neuronCounts));
            sb.AppendFormat("Error on validation set: {0}\r\n", result.MSEHistory[result.MSEHistory.Count - 1].ToString("E2"));
            sb.AppendFormat("Error on test set: {0}\r\n", result.TestSetError.ToString("E2"));
            sb.AppendFormat("Direction misguess: {0}\r\n", result.DirectionMisguessRate.ToString("E2"));
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
                DisplayResults(regressionPlot, errorPlot, result.LearningResult);
            }
        }

        private void CreateResultDirectories(DateTime time)
        {
            string dirName = settingsToRun.Count + "x" + runsPerSettings.ToString() + "_runs_" + 
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
    }
}
