using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra.Double;
using OxyPlot;
using Shell.Containers;
using Shell.Plotting;
using LearningNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNN;
using SharpNN.Statistics;
using System.IO;
using Shell.Enums;
using RecursiveNN;
using OxyPlot.Wpf;
using System.Windows;
using Neural_Network.Plotting;

namespace Shell
{
    public class Engine
    {
        private EngineInitData eid;
        public Dictionary<LearningSettings, List<SingleRunReport>> resultsBySettings 
            = new Dictionary<LearningSettings, List<SingleRunReport>>();
        private int runCounter;
        private string resultsDirectoryPath;

        private IDataSet trainSet;
        private IDataSet testSet;
        private ILearningStrategy learningStrategy;

        public Engine(EngineInitData engineInitData)
        {
            this.eid = engineInitData;
        }

        public void Run()
        {
            CreateResultsDirectory(DateTime.Now);

            foreach (LearningSettings learningSettings in eid.SettingsToRun)
            {
                resultsBySettings[learningSettings] = new List<SingleRunReport>();
                for (int i = 0; i < eid.RunsPerSettings; i++) // repeat several times to average out the results
                {
                    runCounter++;
                    List<int> layersVal = new List<int>();

                    layersVal.Add(eid.InputCount);   
                    foreach (int neuronCount in eid.HiddenNeuronCounts)
                    {
                        layersVal.Add(neuronCount); // re-initialize layer counts -> TODO: Later layer counts should be also configurable in params file / learning settings
                    }

                    layersVal.Add(eid.OutputCount); // TODO: test if the numbers are correct here
                                    

                    if (eid.ProblemType == PartIIProblemType.CTS)
                    {
                        InitCTS(layersVal, eid.TrainSetPercentage);
                    }
                    else
                    {
                        InitStock(layersVal, eid.TrainSetPercentage);
                    }

                   
                    INetwork network = null;
                    switch (eid.NetworkType)
                    {
                        case NetworkType.MLP:
                            network = new NeuralNetwork(learningSettings.Activation, eid.UseBiases, layersVal.ToArray());
                            break;
                        case NetworkType.Jordan:
                            network = new RecursiveNetwork(RecursiveNetwork.Type.Jordan,
                            learningSettings.Activation, eid.UseBiases, layersVal[0], layersVal[1], layersVal[2]);
                            break;
                        case NetworkType.Elman:
                            network = new RecursiveNetwork(RecursiveNetwork.Type.Elman,
                            learningSettings.Activation, eid.UseBiases, layersVal[0], layersVal[1], layersVal[2]);
                            break;
                    }

                    NormalizeData(network, trainSet, testSet);
                    CheckIfPerformPCA(network);
                    learningStrategy = new VSetLearningStrategy(learningSettings);

                    var learningResult = BackpropagationManager.Run(network, trainSet, testSet,
                        learningStrategy, null); // TODO: fix reporting (text update)

                    NormalizeDataBack(network, trainSet, testSet);
                    resultsBySettings[learningSettings].Add(CreateSingleRunReport(layersVal, network, learningResult));
                }
            }

            SaveResults();
        }

        private void CreateResultsDirectory(DateTime time)
        {
            string dirName = time.ToLongDateString() + "_" + time.ToLongTimeString().Replace(":", "-");
            resultsDirectoryPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), dirName);
            Directory.CreateDirectory(resultsDirectoryPath);
        }

        private PlotModel Build1DRegressionModel(IDataSet trainingSet, IDataSet testSet, bool plotAgainstInput) // if plotAgainstInput is true, use input as X axis, not time
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

            RegressionPlotBuilder builder = new RegressionPlotBuilder();
            PlotModel regressionPlotModel = builder.SetUpModel(trainPoints, testIdealPoints, networkAnswers);
            return regressionPlotModel;
        }

        private void InitCTS(List<int> layersVal, float trainSetPercentage)
        {
            int historyLength = 1; // always historyLength = 1
            int trainSetEndIndex = (int)(trainSetPercentage * eid.CsvLines.Count);
            List<DenseVector> chaoticValues = eid.CsvLines; // no need for further parsing

            List<DenseVector> trainValues = chaoticValues.ExtractList(0, trainSetEndIndex);
            List<DenseVector> testValues = chaoticValues.ExtractList(trainSetEndIndex, chaoticValues.Count);

            trainSet = new ChaoticDataSet(trainValues, historyLength, 0);
            if (trainSetEndIndex >= chaoticValues.Count - 1)
            {
                testSet = trainSet.Clone();
            }
            else
            {
                testSet = new ChaoticDataSet(testValues, historyLength, trainSetEndIndex);
            }
        }

        private void InitStock(List<int> layersVal, float trainSetPercentage)
        {
            int trainSetEndIndex = (int)(trainSetPercentage * eid.CsvLines.Count);
            List<DenseVector> allInputs = eid.CsvLines.Select(v => v.CreateSubVector(0, eid.OutputCount)).ToList();
            List<DenseVector> allOutputs = eid.CsvLines.Select(v => v.CreateSubVector(eid.InputCount, eid.OutputCount)).ToList();

            trainSet = new StockDataSet(allInputs.ExtractList(0, trainSetEndIndex),
                allOutputs.ExtractList(0, trainSetEndIndex), 0);

            if (trainSetEndIndex >= allInputs.Count - 1)
            {
                testSet = trainSet.Clone();
            }
            else
            {
                testSet = new StockDataSet(allInputs.ExtractList(trainSetEndIndex, eid.CsvLines.Count),
                    allOutputs.ExtractList(trainSetEndIndex, eid.CsvLines.Count), trainSetEndIndex);
            }
        }

        private void CheckIfPerformPCA(INetwork network)
        {
            if (eid.PcaDimensions > 0)
            {
                LearningNN.PCA.Run(trainSet, eid.PcaDimensions, network.Activation.MinValue, network.Activation.MaxValue);
                LearningNN.PCA.Run(testSet, eid.PcaDimensions, network.Activation.MinValue, network.Activation.MaxValue);
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


            string prefix = report.Name;
            string regressionFileName = prefix + "_regression.png";
            string regressionSavePath = System.IO.Path.Combine(resultsDirectoryPath, regressionFileName);
            using (FileStream fileStream = new FileStream(regressionSavePath, FileMode.CreateNew))
            {
                PngExporter.Export(regressionPlot, fileStream, 900, 900, OxyColors.White);
            }

            string errorFileName = prefix + "_error.png";
            string errorSavePath = System.IO.Path.Combine(resultsDirectoryPath, errorFileName);
            using (FileStream fileStream = new FileStream(errorSavePath, FileMode.CreateNew))
            {
                PngExporter.Export(errorPlot, fileStream, 900, 900, OxyColors.White);
            }

            string infoFileName = prefix + "_info.txt";
            string infoSavePath = System.IO.Path.Combine(resultsDirectoryPath, infoFileName);
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
            sb.AppendLine(System.IO.Path.GetFileName(eid.DataSetName));
            sb.AppendFormat("Layer counts: {0}\r\n", string.Join("-", neuronCounts));
            sb.AppendFormat("Error on validation set: {0}\r\n", result.MSEHistory[result.MSEHistory.Count - 1]);
            sb.AppendFormat("Error on test set: {0}\r\n", result.TestSetError);
            sb.AppendFormat("Direction guessed on test set: {0}\r\n", result.TestSetDirectionGuessed);
            return sb.ToString();
        }

        private void SaveBatchReport(List<AggregateResult> sortedAverages)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("Data set: {0}\r\nParams file: {1}\r\nDate {2}\r\nTime {3}\r\nRuns per settings: {4}",
                System.IO.Path.GetFileName(eid.DataSetName), eid.ParametersFileName,
                DateTime.Now.ToLongDateString(), DateTime.Now.ToLongTimeString(),
                eid.RunsPerSettings);
            sb.AppendLine();
            sb.AppendLine();
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

        private void SaveResults()
        {
            List<AggregateResult> aggregates = new List<AggregateResult>();
            int lsID = 0;
            foreach (KeyValuePair<LearningSettings, List<SingleRunReport>> kvp in resultsBySettings)
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

        //private List<SingleRunReport> reports DiscardWorstRuns(List<SingleRunReport> reportsBestToWorst)
        //{
        //    HashSet<SingleRunReport> excludedReports;
        //    List<SingleRunReport>
        //}

        private void ProcessSingleResultEntry(LearningSettings settings, SingleRunReport result)
        {
            PlotModel regressionPlot = Build1DRegressionModel(result.TrainSet, result.TestSet, eid.PlotAgainstInput);
            ErrorPlotBuilder builder = new ErrorPlotBuilder(eid.ErrorScale);
            PlotModel errorPlot = builder.SetUpModel(result.LearningResult.MSEHistory);
            if (eid.ReportingOptions.ShouldSave)
            {
                SaveResultsToDisk(result.LayersVal, settings, result, regressionPlot, errorPlot, result.Network);
            }

            if (eid.ReportingOptions.ShouldDisplay)
            {
                DisplayResults(regressionPlot, errorPlot, result.LearningResult);
            }
        }

        private SingleRunReport CreateSingleRunReport(List<int> layersVal, INetwork network, LearningResult learningResult)
        {
            SingleRunReport report = eid.ReportingOptions.ShouldSave ?
                new SingleRunReport(network, layersVal, DateTime.Now, learningResult)
                : new SingleRunReport(network, layersVal, DateTime.Now, learningResult, trainSet, testSet);
            return report;
        }
    }
}
