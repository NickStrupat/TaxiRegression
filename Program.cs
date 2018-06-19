using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Threading.Tasks;
using System.IO;
using System.Linq;

namespace TaxiFarePrediction
{
    class Program
    {
        const string BasePath = @".\Data";
        const string _datapath = BasePath + @"\taxi-fare-train.csv";
        const string _testdatapath = BasePath + @"\taxi-fare-test.csv";
        const string _modelpath = BasePath + @"\Model.zip";

        static async Task Main(string[] args)
        {
            var regressorTypes = new [] {
                typeof(FastForestRegressor),
                typeof(FastTreeTweedieRegressor),
                typeof(GeneralizedAdditiveModelRegressor),
                // typeof(OnlineGradientDescentRegressor),
                // typeof(OrdinaryLeastSquaresRegressor),
                typeof(PoissonRegressor),
                typeof(StochasticDualCoordinateAscentRegressor),
            };

            var taxiTripTests = File.ReadAllLines(_testdatapath)
                                    .Skip(1)
                                    .Select(x => {
                                        var fields = x.Split(',');
                                        var taxiTrip = new TaxiTrip {
                                            VendorId = fields[0],
                                            RateCode = fields[1],
                                            PassengerCount = float.Parse(fields[2]),
                                            TripTime = float.Parse(fields[3]),
                                            TripDistance = float.Parse(fields[4]),
                                            PaymentType = fields[5],
                                            FareAmount = float.Parse(fields[6])
                                        };
                                        return (taxiTrip, taxiTripFarePrediction:(TaxiTripFarePrediction)null);
                                    })
                                    //.Take(500)
                                    .ToArray();

            Func<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> trainFunc = Train<FastForestRegressor>;
            foreach (var regressorType in regressorTypes)
            {
                var methodInfo = trainFunc.Method.GetGenericMethodDefinition().MakeGenericMethod(regressorType);
                Console.WriteLine();
                Console.WriteLine("--- " + regressorType.Name + " ---");
                var model = (PredictionModel<TaxiTrip, TaxiTripFarePrediction>) methodInfo.Invoke(null, null);
                Evaluate(model);
                var prediction = model.Predict(TestTrips.Trip1);
                Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);

                Console.WriteLine();
                Console.Write("Begin test...");
                var what = model.Predict(taxiTripTests.Select(x => x.taxiTrip)).ToList();
                Parallel.ForEach(what, (x,state, i) => taxiTripTests[(int)i].taxiTripFarePrediction = x);
                
                Console.WriteLine("Done.");
                var diffs = taxiTripTests.Select(x => Math.Abs(x.taxiTrip.FareAmount - x.taxiTripFarePrediction.FareAmount))
                                         .AsParallel()
                                         .OrderBy(x => x)
                                         .ToList();
                var okay = diffs.Take(diffs.Count - 2)
                                .Where((x, i) => x > 0.75 & x < 450)
                                .Where((x, i) => i % 1000 == 0)
                                .Select((x, i) => i + "," + x)
                                .ToList();
                File.WriteAllLines($@".\Data\{regressorType.Name}_ModelErrors.csv", okay);
                Console.WriteLine(taxiTripTests.Min(x => Math.Abs(x.taxiTrip.FareAmount - x.taxiTripFarePrediction.FareAmount)));
                Console.WriteLine(taxiTripTests.Max(x => Math.Abs(x.taxiTrip.FareAmount - x.taxiTripFarePrediction.FareAmount)));
                Console.WriteLine(taxiTripTests.Average(x => Math.Abs(x.taxiTrip.FareAmount - x.taxiTripFarePrediction.FareAmount)));
            }
        }

        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train<TRegressor>()
        where TRegressor : ILearningPipelineItem, new()
        {
            var modelLoaded = PredictionModel.ReadAsync<TaxiTrip, TaxiTripFarePrediction>(Path.Combine(BasePath, typeof(TRegressor).Name) + ".zip").Result;
            return modelLoaded;

            //var what = modelLoaded.Predict()
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                
                // new CategoricalOneHotVectorizer("VendorId",
                //     "RateCode",
                //     "PaymentType"
                // ),
                new ColumnConcatenator("Features"
                    // ,"VendorId"
                    // ,"RateCode"
                    ,nameof(TaxiTrip.PassengerCount)
                    ,nameof(TaxiTrip.TripTime)
                    ,nameof(TaxiTrip.TripDistance)
                    // ,"PaymentType"
                ),
                new TRegressor()
            };

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            model.WriteAsync(Path.Combine(BasePath, typeof(TRegressor).Name) + ".zip").Wait();
            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            // Rms should be around 2.795276
            Console.WriteLine("Rms=" + metrics.Rms);
            Console.WriteLine("RSquared = " + metrics.RSquared);
            Console.WriteLine();
        }
    }
}