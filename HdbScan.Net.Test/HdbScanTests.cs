using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using HdbScan.Net;

namespace HdbScan.Net.Test
{
    public class HdbScanTests
    {
        private static double EuclideanDistance(double[] a, double[] b)
        {
            var sum = 0.0;
            for (var i = 0; i < a.Length; i++)
            {
                var d = a[i] - b[i];
                sum += d * d;
            }
            return Math.Sqrt(sum);
        }

        /// <summary>
        /// Validates HDBSCAN implementation against scikit-learn HDBSCAN on the Iris dataset.
        /// Python reference (scikit-learn 1.3+):
        ///   from sklearn.cluster import HDBSCAN
        ///   hdb = HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_method='eom')
        ///   labels = hdb.fit_predict(X)
        /// Expected result:
        ///   - 2 clusters, 0 noise points
        ///   - First 50 points (setosa) in one cluster
        ///   - Last 100 points (versicolor + virginica) in another cluster
        /// </summary>
        [Test]
        public void Iris_MatchesScikitLearn()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 5,
                MinSamples = 5,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // scikit-learn produces 2 clusters with 0 noise
            Assert.That(model.ClusterCount, Is.EqualTo(2), "Expected 2 clusters (setosa vs versicolor+virginica)");

            var noiseCount = model.Labels.Count(l => l == -1);
            Assert.That(noiseCount, Is.EqualTo(0), "Expected 0 noise points");

            // First 50 points (setosa) should all have the same label
            var setosaLabels = model.Labels.Take(50).ToHashSet();
            Assert.That(setosaLabels.Count, Is.EqualTo(1), "All setosa should be in one cluster");
            Assert.That(setosaLabels.First(), Is.Not.EqualTo(-1), "Setosa should not be noise");

            // Last 100 points (versicolor + virginica) should all have the same label
            var otherLabels = model.Labels.Skip(50).ToHashSet();
            Assert.That(otherLabels.Count, Is.EqualTo(1), "All versicolor+virginica should be in one cluster");
            Assert.That(otherLabels.First(), Is.Not.EqualTo(-1), "Versicolor+virginica should not be noise");

            // The two clusters should have different labels
            Assert.That(setosaLabels.First(), Is.Not.EqualTo(otherLabels.First()),
                "Setosa and versicolor+virginica should be in different clusters");
        }

        /// <summary>
        /// Validates HDBSCAN EOM with multiple (>2) clusters against scikit-learn.
        /// Uses KPCA-transformed Iris data which produces finer-grained clustering.
        /// Python reference (scikit-learn 1.3+):
        ///   from sklearn.cluster import HDBSCAN
        ///   import pandas as pd
        ///   df = pd.read_csv("iris_kpca_poly_d2c1.csv")
        ///   X = df[["x", "y"]].values
        ///   hdb = HDBSCAN(min_cluster_size=3, min_samples=3, cluster_selection_method='eom')
        ///   labels = hdb.fit_predict(X)
        /// Expected result:
        ///   - 12 clusters, 24 noise points
        ///   - First 50 points (setosa) form one cluster (cluster 0)
        ///   - Remaining 100 points distributed across 11 clusters + noise
        /// </summary>
        [Test]
        public void IrisKpca_EomMultipleClusters_MatchesScikitLearn()
        {
            var xs = ReadXY("iris_kpca_poly_d2c1.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 3,
                MinSamples = 3,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // scikit-learn produces 12 clusters with 24 noise points
            Assert.That(model.ClusterCount, Is.EqualTo(12), "Expected 12 clusters");

            var noiseCount = model.Labels.Count(l => l == -1);
            Assert.That(noiseCount, Is.EqualTo(24), "Expected 24 noise points");

            // First 50 points (setosa) should all be in one cluster
            var setosaLabels = model.Labels.Take(50).ToHashSet();
            Assert.That(setosaLabels.Count, Is.EqualTo(1), "All setosa should be in one cluster");
            Assert.That(setosaLabels.First(), Is.Not.EqualTo(-1), "Setosa should not be noise");

            // Verify cluster size distribution matches Python
            // Python cluster sizes (excluding noise): 50, 20, 11, 8, 7, 6, 6, 5, 4, 3, 3, 3
            var clusterSizes = model.Labels
                .Where(l => l >= 0)
                .GroupBy(l => l)
                .Select(g => g.Count())
                .OrderByDescending(c => c)
                .ToArray();

            var expectedSizes = new[] { 50, 20, 11, 8, 7, 6, 6, 5, 4, 3, 3, 3 };
            Assert.That(clusterSizes, Is.EqualTo(expectedSizes),
                "Cluster size distribution should match Python");
        }

        /// <summary>
        /// Additional validation with different parameters.
        /// Python reference:
        ///   hdb = HDBSCAN(min_cluster_size=15, min_samples=10, cluster_selection_method='eom')
        /// Expected: Same 2-cluster structure (setosa separate from rest)
        /// </summary>
        [Test]
        public void Iris_DifferentParameters()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 15,
                MinSamples = 10,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Should still produce 2 clusters
            Assert.That(model.ClusterCount, Is.EqualTo(2));

            // Setosa (first 50) should be in one cluster
            var setosaLabel = model.Labels[0];
            Assert.That(setosaLabel, Is.Not.EqualTo(-1));
            Assert.That(model.Labels.Take(50).All(l => l == setosaLabel), Is.True,
                "All setosa should be in the same cluster");

            // Versicolor+virginica should be in a different cluster
            var otherLabel = model.Labels[50];
            Assert.That(otherLabel, Is.Not.EqualTo(-1));
            Assert.That(otherLabel, Is.Not.EqualTo(setosaLabel));
            Assert.That(model.Labels.Skip(50).All(l => l == otherLabel), Is.True,
                "All versicolor+virginica should be in the same cluster");
        }

        /// <summary>
        /// Validates HDBSCAN Leaf cluster selection against scikit-learn.
        /// Python reference:
        ///   from sklearn.cluster import HDBSCAN
        ///   hdb = HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_method='leaf')
        ///   labels = hdb.fit_predict(X)
        /// Expected result:
        ///   - 6 clusters, 85 noise points
        ///   - First 50 points (setosa) grouped into 2 clusters (clusters 0, 1)
        ///   - Versicolor and virginica split across remaining clusters
        /// Note: Minor point differences (1-2 points) may occur due to floating-point precision.
        /// </summary>
        [Test]
        public void Iris_LeafMethod_MatchesScikitLearn()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 5,
                MinSamples = 5,
                ClusterSelectionMethod = ClusterSelectionMethod.Leaf
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // scikit-learn produces 6 clusters with 85 noise points
            Assert.That(model.ClusterCount, Is.EqualTo(6), "Expected 6 leaf clusters");

            var noiseCount = model.Labels.Count(l => l == -1);
            Assert.That(noiseCount, Is.EqualTo(85), "Expected 85 noise points");

            // First 50 points (setosa) should be in exactly 2 clusters
            var setosaLabels = model.Labels.Take(50).Where(l => l != -1).Distinct().ToList();
            Assert.That(setosaLabels.Count, Is.EqualTo(2), "Setosa should span 2 leaf clusters");

            // Setosa clusters should each have 8 points (16 total non-noise in setosa)
            var setosaNonNoise = model.Labels.Take(50).Count(l => l != -1);
            Assert.That(setosaNonNoise, Is.EqualTo(16), "Setosa should have 16 non-noise points");

            // Verify the exact setosa cluster assignments match Python
            // Python: cluster 0 has indices [0, 4, 7, 17, 27, 28, 39, 49]
            // Python: cluster 1 has indices [1, 9, 12, 25, 29, 30, 34, 47]
            var cluster0Indices = new[] { 0, 4, 7, 17, 27, 28, 39, 49 };
            var cluster1Indices = new[] { 1, 9, 12, 25, 29, 30, 34, 47 };

            var cluster0Label = model.Labels[0];
            var cluster1Label = model.Labels[1];
            Assert.That(cluster0Label, Is.Not.EqualTo(-1));
            Assert.That(cluster1Label, Is.Not.EqualTo(-1));
            Assert.That(cluster0Label, Is.Not.EqualTo(cluster1Label), "Clusters 0 and 1 should be distinct");

            // All points in cluster0Indices should have the same label
            foreach (var idx in cluster0Indices)
            {
                Assert.That(model.Labels[idx], Is.EqualTo(cluster0Label),
                    $"Point {idx} should be in same cluster as point 0");
            }

            // All points in cluster1Indices should have the same label
            foreach (var idx in cluster1Indices)
            {
                Assert.That(model.Labels[idx], Is.EqualTo(cluster1Label),
                    $"Point {idx} should be in same cluster as point 1");
            }
        }

        /// <summary>
        /// Validates HDBSCAN on the pendigits (8x8 handwritten digits) dataset against scikit-learn.
        /// Python reference (scikit-learn 1.3+):
        ///   from sklearn.datasets import load_digits
        ///   from sklearn.cluster import HDBSCAN
        ///   X = load_digits().data  # 1797 samples, 64 features
        ///   hdb = HDBSCAN(min_cluster_size=15, min_samples=15, cluster_selection_method='eom')
        ///   labels = hdb.fit_predict(X)
        /// Expected result:
        ///   - 10 clusters, 894 noise points
        ///   - Cluster sizes (descending): [171, 146, 127, 119, 99, 83, 64, 42, 35, 17]
        /// </summary>
        [Test]
        public void Pendigits_MatchesScikitLearn()
        {
            var xs = ReadCsv("pendigits.csv", columns: 64).ToArray();
            Assert.That(xs.Length, Is.EqualTo(1797), "Pendigits dataset should have 1797 samples");

            var options = new HdbScanOptions
            {
                MinClusterSize = 15,
                MinSamples = 15,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // scikit-learn produces 10 clusters with 894 noise points
            Assert.That(model.ClusterCount, Is.EqualTo(10), "Expected 10 clusters");

            var noiseCount = model.Labels.Count(l => l == -1);
            Assert.That(noiseCount, Is.EqualTo(894), "Expected 894 noise points");

            // Verify cluster size distribution matches Python
            var clusterSizes = model.Labels
                .Where(l => l >= 0)
                .GroupBy(l => l)
                .Select(g => g.Count())
                .OrderByDescending(c => c)
                .ToArray();

            var expectedSizes = new[] { 171, 146, 127, 119, 99, 83, 64, 42, 35, 17 };
            Assert.That(clusterSizes, Is.EqualTo(expectedSizes),
                "Cluster size distribution should match scikit-learn");
        }

        private static IEnumerable<double[]> ReadCsv(string filename, int columns)
        {
            var path = Path.Combine("dataset", filename);
            foreach (var line in File.ReadLines(path).Skip(1))
            {
                var values = line.Split(',').Take(columns).Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                yield return values;
            }
        }

        private static IEnumerable<double[]> ReadIris(string filename)
        {
            var path = Path.Combine("dataset", filename);
            foreach (var line in File.ReadLines(path).Skip(1))
            {
                var values = line.Split(',').Take(4).Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                yield return values;
            }
        }

        private static IEnumerable<double[]> ReadXY(string filename)
        {
            var path = Path.Combine("dataset", filename);
            foreach (var line in File.ReadLines(path).Skip(1))
            {
                var values = line.Split(',').Take(2).Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                yield return values;
            }
        }

        [Test]
        public void BasicClustering()
        {
            double[][] xs =
            [
                // Cluster 1 (dense)
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],

                // Cluster 2 (dense)
                [10, 0], [10, 1], [11, 0], [11, 1], [10.5, 0.5],

                // Noise
                [100, 100],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Should find 2 clusters
            Assert.That(model.ClusterCount, Is.EqualTo(2));

            // Noise point should be labeled -1
            Assert.That(model.Labels[10], Is.EqualTo(-1));

            // Points in same cluster should have same label
            var cluster1Label = model.Labels[0];
            var cluster2Label = model.Labels[5];
            Assert.That(cluster1Label, Is.Not.EqualTo(-1));
            Assert.That(cluster2Label, Is.Not.EqualTo(-1));
            Assert.That(cluster1Label, Is.Not.EqualTo(cluster2Label));

            for (var i = 0; i < 5; i++)
            {
                Assert.That(model.Labels[i], Is.EqualTo(cluster1Label));
            }

            for (var i = 5; i < 10; i++)
            {
                Assert.That(model.Labels[i], Is.EqualTo(cluster2Label));
            }
        }

        [Test]
        public void AllNoise()
        {
            double[][] xs =
            [
                [0, 0],
                [100, 100],
                [200, 200],
                [300, 300],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // All points should be noise
            Assert.That(model.ClusterCount, Is.EqualTo(0));
            Assert.That(model.Labels.All(l => l == -1), Is.True);
        }

        [Test]
        public void SingleClusterNotAllowed()
        {
            double[][] xs =
            [
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = false };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // By default, single cluster not allowed
            Assert.That(model.ClusterCount, Is.EqualTo(0));
        }

        [Test]
        public void SingleClusterAllowed()
        {
            double[][] xs =
            [
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Single cluster allowed
            Assert.That(model.ClusterCount, Is.EqualTo(1));
            Assert.That(model.Labels.All(l => l == 0), Is.True);
        }

        [Test]
        public void PredictionData()
        {
            double[][] xs =
            [
                // Cluster 1
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],

                // Cluster 2
                [10, 0], [10, 1], [11, 0], [11, 1], [10.5, 0.5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

            Assert.That(model.HasPredictionData, Is.True);

            // Predict point near cluster 1
            double[] nearCluster1 = [0.25, 0.25];
            var (label1, prob1) = model.PredictWithProbability(nearCluster1);
            Assert.That(label1, Is.EqualTo(model.Labels[0]));
            Assert.That(prob1, Is.GreaterThan(0));

            // Predict point near cluster 2
            double[] nearCluster2 = [10.25, 0.25];
            var (label2, prob2) = model.PredictWithProbability(nearCluster2);
            Assert.That(label2, Is.EqualTo(model.Labels[5]));
            Assert.That(prob2, Is.GreaterThan(0));
        }

        [Test]
        public void PredictionDataNotStoredThrows()
        {
            double[][] xs =
            [
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: false);

            Assert.That(model.HasPredictionData, Is.False);

            double[] point = [0.5, 0.5];
            Assert.Throws<InvalidOperationException>(() => model.Predict(point));
        }

        [Test]
        public void LeafClusterSelection()
        {
            double[][] xs =
            [
                // Cluster 1
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],

                // Cluster 2
                [10, 0], [10, 1], [11, 0], [11, 1], [10.5, 0.5],
            ];

            var options = new HdbScanOptions
            {
                MinClusterSize = 3,
                ClusterSelectionMethod = ClusterSelectionMethod.Leaf
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // With leaf method should still find 2 clusters
            Assert.That(model.ClusterCount, Is.EqualTo(2));
        }

        [Test]
        public void MinSamples()
        {
            double[][] xs =
            [
                // Dense core
                [0, 0], [0, 1], [1, 0], [1, 1],

                // Less dense points
                [0.5, 5], [1, 5], [0, 5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, MinSamples = 2 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Should work without error
            Assert.That(model.Labels.Count, Is.EqualTo(7));
        }

        [Test]
        public void Probabilities()
        {
            double[][] xs =
            [
                // Cluster
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // All probabilities should be between 0 and 1
            foreach (var prob in model.Probabilities)
            {
                Assert.That(prob, Is.GreaterThanOrEqualTo(0));
                Assert.That(prob, Is.LessThanOrEqualTo(1));
            }
        }

        [Test]
        public void OutlierScores()
        {
            double[][] xs =
            [
                // Cluster
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],

                // Outlier
                [100, 100],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Outlier scores should be between 0 and 1
            foreach (var score in model.OutlierScores)
            {
                Assert.That(score, Is.GreaterThanOrEqualTo(0));
                Assert.That(score, Is.LessThanOrEqualTo(1));
            }

            // Noise point should have high outlier score
            Assert.That(model.OutlierScores[5], Is.GreaterThan(0.9));
        }

        [Test]
        public void CustomDistanceMetric()
        {
            string[] words = ["cat", "bat", "rat", "dog", "fog", "log"];

            Func<string, string, double> hammingDistance = (a, b) =>
            {
                var dist = 0;
                var len = Math.Min(a.Length, b.Length);
                for (var i = 0; i < len; i++)
                {
                    if (a[i] != b[i]) dist++;
                }
                dist += Math.Abs(a.Length - b.Length);
                return dist;
            };

            var options = new HdbScanOptions { MinClusterSize = 2 };
            var model = new HdbScan<string>(words, hammingDistance, options);

            // Should cluster similar words together
            Assert.That(model.Labels.Count, Is.EqualTo(6));
        }

        [Test]
        public void ThreeClusters()
        {
            double[][] xs =
            [
                // Cluster 1
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],

                // Cluster 2
                [10, 0], [10, 1], [11, 0], [11, 1], [10.5, 0.5],

                // Cluster 3
                [5, 10], [5, 11], [6, 10], [6, 11], [5.5, 10.5],

                // Noise
                [50, 50],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Should find 3 clusters
            Assert.That(model.ClusterCount, Is.EqualTo(3));

            // Noise should be -1
            Assert.That(model.Labels[15], Is.EqualTo(-1));

            // Verify clusters are distinct
            var distinctLabels = model.Labels.Where(l => l >= 0).Distinct().Count();
            Assert.That(distinctLabels, Is.EqualTo(3));
        }

        [Test]
        public void TooFewPoints()
        {
            double[][] xs = [[0, 0], [1, 1]];

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // All points should be noise
            Assert.That(model.ClusterCount, Is.EqualTo(0));
            Assert.That(model.Labels.All(l => l == -1), Is.True);
        }

        [Test]
        public void OptionsValidation()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var options = new HdbScanOptions { MinClusterSize = 1 };
            });

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var options = new HdbScanOptions { MinSamples = -1 };
            });

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var options = new HdbScanOptions { MinSamples = 0 };
            });

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var options = new HdbScanOptions { MaxDegreeOfParallelism = 0 };
            });

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var options = new HdbScanOptions { MaxDegreeOfParallelism = -2 };
            });

            // 1 is valid (as in scikit-learn): every point is a core point.
            Assert.That(new HdbScanOptions { MinSamples = 1 }.MinSamples, Is.EqualTo(1));
        }

        [Test]
        public void SinglePoint()
        {
            double[][] xs = [[0, 0]];

            var options = new HdbScanOptions { MinClusterSize = 2 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            Assert.That(model.ClusterCount, Is.EqualTo(0));
            Assert.That(model.Labels[0], Is.EqualTo(-1));
        }

        [Test]
        public void IdenticalPoints()
        {
            double[][] xs =
            [
                [5, 5], [5, 5], [5, 5], [5, 5], [5, 5],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // All identical points should form one cluster
            Assert.That(model.ClusterCount, Is.EqualTo(1));
            Assert.That(model.Labels.All(l => l == 0), Is.True);
        }

        [Test]
        public void EmptyInputThrows()
        {
            double[][] xs = [];

            var options = new HdbScanOptions { MinClusterSize = 2 };
            Assert.Throws<ArgumentException>(() => new HdbScan<double[]>(xs, EuclideanDistance, options));
        }

        [Test]
        public void NullInputThrows()
        {
            IReadOnlyList<double[]>? xs = null;

            Assert.Throws<ArgumentNullException>(() => new HdbScan<double[]>(xs!, EuclideanDistance));
        }

        [Test]
        public void LargeMinSamples()
        {
            double[][] xs =
            [
                [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            ];

            // minSamples larger than data size
            var options = new HdbScanOptions { MinClusterSize = 3, MinSamples = 10 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Should still work without error
            Assert.That(model.Labels.Count, Is.EqualTo(5));
        }

        [Test]
        public void TwoPoints()
        {
            double[][] xs = [[0, 0], [1, 1]];

            var options = new HdbScanOptions { MinClusterSize = 2, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // Two points can form a cluster if minClusterSize = 2
            Assert.That(model.Labels.Count, Is.EqualTo(2));
        }

        [Test]
        public void PredictOnNoiseModel()
        {
            double[][] xs =
            [
                [0, 0], [100, 100], [200, 200],
            ];

            var options = new HdbScanOptions { MinClusterSize = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

            // All points are noise
            Assert.That(model.ClusterCount, Is.EqualTo(0));

            // Prediction should return noise
            double[] point = [50, 50];
            var (label, prob) = model.PredictWithProbability(point);
            Assert.That(label, Is.EqualTo(-1));
            Assert.That(prob, Is.EqualTo(0));
        }
        [Test]
        public void OutlierScores_TwoDensityClusters()
        {
            // Dense cluster (spacing ~0.1) + sparse cluster (spacing ~2.0) + outlier.
            // The old bug used a global max lambda, inflating scores for the sparse cluster.
            var points = new List<double[]>();

            // Dense cluster: 20 points near origin, spacing ~0.1
            for (var i = 0; i < 20; i++)
            {
                points.Add([0.1 * (i % 5), 0.1 * (i / 5)]);
            }

            // Sparse cluster: 20 points far away, spacing ~2.0
            for (var i = 0; i < 20; i++)
            {
                points.Add([50 + 2.0 * (i % 5), 50 + 2.0 * (i / 5)]);
            }

            // Outlier
            points.Add([200, 200]);

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(points.ToArray(), EuclideanDistance, options);

            // Core points in BOTH clusters should have low scores (<= 0.5).
            // This is the key regression test — the old bug inflated sparse cluster scores.
            // The bound is inclusive: a grid corner leaves at twice the grid spacing, half
            // the peak density of its cluster, which is a score of exactly 0.5.
            for (var i = 0; i < 20; i++)
            {
                if (model.Labels[i] >= 0)
                {
                    Assert.That(model.OutlierScores[i], Is.LessThanOrEqualTo(0.5),
                        $"Dense cluster point {i} should have low outlier score");
                }
            }

            for (var i = 20; i < 40; i++)
            {
                if (model.Labels[i] >= 0)
                {
                    Assert.That(model.OutlierScores[i], Is.LessThanOrEqualTo(0.5),
                        $"Sparse cluster point {i} should have low outlier score");
                }
            }

            // Outlier should have high score
            Assert.That(model.OutlierScores[40], Is.GreaterThan(0.5));
        }

        [Test]
        public void OutlierScores_PerClusterNormalization()
        {
            // 10 very dense points + 10 moderate points.
            // All clustered points should have scores < 0.5.
            var points = new List<double[]>();

            // Very dense cluster: 10 points
            for (var i = 0; i < 10; i++)
            {
                points.Add([0.01 * (i % 5), 0.01 * (i / 5)]);
            }

            // Moderate cluster: 10 points
            for (var i = 0; i < 10; i++)
            {
                points.Add([20 + 0.5 * (i % 5), 20 + 0.5 * (i / 5)]);
            }

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(points.ToArray(), EuclideanDistance, options);

            for (var i = 0; i < 20; i++)
            {
                if (model.Labels[i] >= 0)
                {
                    Assert.That(model.OutlierScores[i], Is.LessThan(0.5),
                        $"Clustered point {i} should have low outlier score");
                }
            }
        }

        /// <summary>
        /// Validates that MinSamples includes the point itself, matching sklearn's convention.
        /// sklearn's HDBSCAN counts the point itself as one of its min_samples neighbors
        /// (unlike scikit-learn-contrib/hdbscan which does NOT include self).
        ///
        /// This test uses a configuration where the two interpretations produce different
        /// results: with self-inclusion (correct) → 0 noise; without self → point 118
        /// becomes noise. This catches an off-by-one error in core distance computation.
        ///
        /// Python reference (scikit-learn 1.3+):
        ///   df = pd.read_csv("iris_kpca_poly_d2c1.csv")
        ///   X = df[["x", "y"]].values
        ///   hdb = HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_method='eom')
        ///   labels = hdb.fit_predict(X)
        ///   # 2 clusters, 0 noise, labels[118] != -1
        ///
        ///   # Without self (equivalent to min_samples=6 in sklearn):
        ///   hdb2 = HDBSCAN(min_cluster_size=5, min_samples=6, cluster_selection_method='eom')
        ///   labels2 = hdb2.fit_predict(X)
        ///   # 2 clusters, 1 noise at index 118
        /// </summary>
        [Test]
        public void IrisKpca_MinSamplesIncludesSelf_MatchesScikitLearn()
        {
            var xs = ReadXY("iris_kpca_poly_d2c1.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 5,
                MinSamples = 5,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            Assert.That(model.ClusterCount, Is.EqualTo(2), "Expected 2 clusters");

            var noiseCount = model.Labels.Count(l => l == -1);
            Assert.That(noiseCount, Is.EqualTo(0),
                "Expected 0 noise points (if min_samples includes self). " +
                "If this fails with 1 noise point at index 118, core distances are computed " +
                "without self-inclusion, which does not match sklearn.");

            // Point 118 is the canary: it becomes noise if self is excluded from count.
            Assert.That(model.Labels[118], Is.Not.EqualTo(-1),
                "Point 118 should NOT be noise when min_samples includes self (matching sklearn)");
        }

        /// <summary>
        /// Validates HDBSCAN probabilities against scikit-learn on the Iris dataset.
        /// Python reference (scikit-learn 1.3+):
        ///   hdb = HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_method='eom')
        ///   hdb.fit(X)
        ///   # Setosa (first 50): min prob = 0.313625, 21 points at prob=1.0
        ///   # Other (last 100): min prob = 0.422159, 57 points at prob=1.0
        /// </summary>
        [Test]
        public void Probabilities_Iris_MatchesScikitLearn()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 5,
                MinSamples = 5,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            Assert.That(model.ClusterCount, Is.EqualTo(2));

            // Validate probabilities match scikit-learn.
            // When EOM selects non-leaf clusters, points from dense sub-clusters
            // get probability=1.0 (they are the most central members).
            var setosaProbs = model.Probabilities.Take(50).ToArray();
            Assert.That(setosaProbs.Min(), Is.EqualTo(0.313625).Within(1e-4),
                "Setosa min probability should match scikit-learn");
            Assert.That(setosaProbs.Count(p => Math.Abs(p - 1.0) < 1e-9), Is.EqualTo(21),
                "Setosa should have 21 points at prob=1.0 (matching scikit-learn)");

            var otherProbs = model.Probabilities.Skip(50).ToArray();
            Assert.That(otherProbs.Min(), Is.EqualTo(0.422159).Within(1e-4),
                "Versicolor+virginica min probability should match scikit-learn");
            Assert.That(otherProbs.Count(p => Math.Abs(p - 1.0) < 1e-9), Is.EqualTo(57),
                "Versicolor+virginica should have 57 points at prob=1.0 (matching scikit-learn)");
        }

        [Test]
        public void OutlierScores_Iris_MatchesScikitLearn()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions
            {
                MinClusterSize = 5,
                MinSamples = 5,
                ClusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass
            };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            // All scores should be in [0, 1]
            foreach (var score in model.OutlierScores)
            {
                Assert.That(score, Is.GreaterThanOrEqualTo(0));
                Assert.That(score, Is.LessThanOrEqualTo(1));
            }

            // Setosa (first 50 points) is well-separated; most points should have low scores.
            var setosaMedianScore = model.OutlierScores.Take(50).OrderBy(x => x).ElementAt(25);
            Assert.That(setosaMedianScore, Is.LessThan(0.5),
                "Setosa (well-separated cluster) should have low median outlier score");
        }

        /// <summary>
        /// Every single label must equal scikit-learn's, including the cluster numbering.
        /// Python reference (scikit-learn 1.8, algorithm='brute' and 'kd_tree' agree):
        ///   X = pd.read_csv("iris_kpca_poly_d2c1.csv")[["x", "y"]].values
        ///   HDBSCAN(min_cluster_size=3, min_samples=3).fit(X).labels_
        /// </summary>
        [Test]
        public void IrisKpca_ExactLabels_MatchScikitLearn()
        {
            var xs = ReadXY("iris_kpca_poly_d2c1.csv").ToArray();

            var options = new HdbScanOptions { MinClusterSize = 3, MinSamples = 3 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            int[] expected =
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                -1, 9, -1, 3, 9, 10, 9, 1, 9, 3, 1, -1, 5, 11, -1, 9, 10, 5, -1, 3, 11, 5, 11, 11, 11,
                9, -1, -1, 11, -1, 3, 3, 5, 11, 10, 11, -1, 11, 5, 3, 5, 11, 5, 1, 5, 5, 5, -1, 1, 5,
                -1, 11, -1, 4, 4, 2, -1, -1, 4, -1, 6, -1, 7, 8, 11, 6, 6, -1, -1, 8, 7, 8, 2, 11, 7,
                -1, 11, 11, 4, -1, -1, -1, 4, 11, -1, 2, 7, 6, 11, 7, 7, -1, 11, 7, 7, 6, 11, 6, 6, 11,
            ];
            Assert.That(model.Labels, Is.EqualTo(expected));
        }

        /// <summary>
        /// When the root is selected as the single cluster, scikit-learn keeps only the points
        /// that stay until the root's highest density; points that leave earlier are noise.
        /// Python reference (scikit-learn 1.8, algorithm='brute' and 'kd_tree' agree):
        ///   X = iris[:50]   # setosa only
        ///   hdb = HDBSCAN(min_cluster_size=10, min_samples=2, allow_single_cluster=True).fit(X)
        ///   np.where(hdb.labels_ == -1)[0]; hdb.probabilities_   # 1 for members, 0 for noise
        /// </summary>
        [Test]
        public void SingleCluster_NoisePoints_MatchScikitLearn()
        {
            var xs = ReadIris("iris.csv").Take(50).ToArray();

            var options = new HdbScanOptions { MinClusterSize = 10, MinSamples = 2, AllowSingleCluster = true };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

            int[] expectedNoise = [5, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 31, 32, 33, 36, 41, 44, 46];

            Assert.That(model.ClusterCount, Is.EqualTo(1));
            for (var i = 0; i < xs.Length; i++)
            {
                var noise = expectedNoise.Contains(i);
                Assert.That(model.Labels[i], Is.EqualTo(noise ? -1 : 0), $"Label of point {i}");
                Assert.That(model.Probabilities[i], Is.EqualTo(noise ? 0.0 : 1.0), $"Probability of point {i}");

                // Prediction applies the same rule.
                Assert.That(model.Predict(xs[i]), Is.EqualTo(model.Labels[i]), $"Prediction of point {i}");
            }
        }

        /// <summary>
        /// MinSamples = 1: every point is a core point (plain single linkage distances).
        /// Python reference (scikit-learn 1.8):
        ///   HDBSCAN(min_cluster_size=5, min_samples=1).fit(X)   # X = iris_kpca_poly_d2c1
        ///   9 clusters, 21 noise points
        /// </summary>
        [Test]
        public void IrisKpca_MinSamplesOne_MatchesScikitLearn()
        {
            var xs = ReadXY("iris_kpca_poly_d2c1.csv").ToArray();

            var options = new HdbScanOptions { MinClusterSize = 5, MinSamples = 1 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);

            Assert.That(model.ClusterCount, Is.EqualTo(9));
            Assert.That(model.Labels.Count(l => l == -1), Is.EqualTo(21));

            var clusterSizes = model.Labels.Where(l => l >= 0).GroupBy(l => l).Select(g => g.Count()).OrderByDescending(c => c).ToArray();
            Assert.That(clusterSizes, Is.EqualTo(new[] { 50, 25, 12, 10, 8, 7, 6, 6, 5 }));
        }

        /// <summary>
        /// HDBSCAN only compares distances, so the unit of measurement must not matter.
        /// The scales are powers of two: those rescale floating point numbers exactly, so
        /// equal distances stay equal and the results must be identical.
        /// </summary>
        [TestCase(1.0 / (1 << 30))]
        [TestCase(1 << 20)]
        public void ScaleInvariance(double scale)
        {
            var xs = ReadIris("iris.csv").ToArray();
            var scaled = xs.Select(x => x.Select(v => v * scale).ToArray()).ToArray();

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options);
            var scaledModel = new HdbScan<double[]>(scaled, EuclideanDistance, options);

            Assert.That(scaledModel.Labels, Is.EqualTo(model.Labels));
            Assert.That(scaledModel.Probabilities, Is.EqualTo(model.Probabilities).Within(1e-12));
            Assert.That(scaledModel.OutlierScores, Is.EqualTo(model.OutlierScores).Within(1e-12));
        }

        /// <summary>
        /// Duplicate points have distance 0, i.e. infinite density (lambda). Results must
        /// stay well-defined: no NaN, everything in [0, 1].
        /// </summary>
        [Test]
        public void DuplicatePoints_ResultsAreWellDefined()
        {
            var xs = new List<double[]>();
            for (var i = 0; i < 6; i++) xs.Add([0, 0]);          // 6 duplicates
            for (var i = 0; i < 6; i++) xs.Add([0.1 * i, 0.3]);  // spread out, same region
            for (var i = 0; i < 6; i++) xs.Add([50, 50]);        // 6 duplicates, far away
            for (var i = 0; i < 6; i++) xs.Add([50 + 0.1 * i, 50.3]);

            foreach (var method in new[] { ClusterSelectionMethod.ExcessOfMass, ClusterSelectionMethod.Leaf })
            {
                var options = new HdbScanOptions { MinClusterSize = 3, ClusterSelectionMethod = method };
                var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

                Assert.That(model.ClusterCount, Is.GreaterThanOrEqualTo(2));
                Assert.That(model.Probabilities, Is.All.InRange(0.0, 1.0));
                Assert.That(model.OutlierScores, Is.All.InRange(0.0, 1.0));

                // The duplicates are the densest points there are: full members, no outliers.
                foreach (var i in new[] { 0, 5, 12, 17 })
                {
                    Assert.That(model.Labels[i], Is.GreaterThanOrEqualTo(0));
                    Assert.That(model.Probabilities[i], Is.EqualTo(1.0));
                    Assert.That(model.OutlierScores[i], Is.EqualTo(0.0));
                }
                Assert.That(model.Labels[0], Is.Not.EqualTo(model.Labels[12]));

                Assert.That(model.PredictWithProbability([0, 0]), Is.EqualTo((model.Labels[0], 1.0)));
            }
        }

        [Test]
        public void InvalidDistancesThrow()
        {
            double[][] xs = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [10, 10]];
            var options = new HdbScanOptions { MinClusterSize = 3, AllowSingleCluster = true };

            foreach (var bad in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -1.0 })
            {
                // A metric that returns the bad value for the far-away point only.
                Func<double[], double[], double> metric = (a, b) => a[0] > 5 || b[0] > 5 ? bad : EuclideanDistance(a, b);

                Assert.Throws<ArgumentException>(() => new HdbScan<double[]>(xs, metric, options), $"Fitting with distance {bad}");

                var model = new HdbScan<double[]>(xs.Take(5).ToArray(), metric, options, predictionData: true);
                Assert.That(model.ClusterCount, Is.EqualTo(1));
                Assert.Throws<ArgumentException>(() => model.Predict([10, 10]), $"Predicting with distance {bad}");
            }
        }

        /// <summary>
        /// Without the stored distance matrix (the mode used automatically above 46,340 points)
        /// and with parallel distance computation, the result must be exactly the same.
        /// </summary>
        [TestCase(true, 4)]
        [TestCase(true, -1)]
        [TestCase(false, 1)]
        [TestCase(false, 4)]
        [TestCase(false, -1)]
        public void DistanceMatrixAndParallelism_DoNotChangeResults(bool storeDistanceMatrix, int maxDegreeOfParallelism)
        {
            foreach (var xs in new[] { ReadXY("iris_kpca_poly_d2c1.csv").ToArray(), ReadIris("iris.csv").ToArray(), ReadCsv("pendigits.csv", columns: 64).ToArray() })
            {
                var expected = new HdbScan<double[]>(xs, EuclideanDistance, new HdbScanOptions { MinClusterSize = 3 }, predictionData: true);

                var options = new HdbScanOptions
                {
                    MinClusterSize = 3,
                    StoreDistanceMatrix = storeDistanceMatrix,
                    MaxDegreeOfParallelism = maxDegreeOfParallelism
                };
                var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

                Assert.That(model.ClusterCount, Is.EqualTo(expected.ClusterCount));
                Assert.That(model.Labels, Is.EqualTo(expected.Labels));
                Assert.That(model.Probabilities, Is.EqualTo(expected.Probabilities));
                Assert.That(model.OutlierScores, Is.EqualTo(expected.OutlierScores));
                Assert.That(model.PredictWithProbability(xs[0]), Is.EqualTo(expected.PredictWithProbability(xs[0])));
            }
        }

        /// <summary>
        /// Without the distance matrix the metric is called 3 times as often (n² for the core
        /// distances + n²/2 for the MST, instead of n²/2), and nothing of size n² is needed.
        /// </summary>
        [Test]
        public void WithoutDistanceMatrix_MetricCallCount()
        {
            var xs = ReadIris("iris.csv").ToArray();
            var n = xs.Length;

            long Calls(bool storeDistanceMatrix)
            {
                var calls = 0L;
                var options = new HdbScanOptions { StoreDistanceMatrix = storeDistanceMatrix };
                _ = new HdbScan<double[]>(xs, (a, b) => { calls++; return EuclideanDistance(a, b); }, options);
                return calls;
            }

            Assert.That(Calls(true), Is.EqualTo(n * (n - 1) / 2));
            Assert.That(Calls(false), Is.EqualTo(n * (n - 1) + n * (n - 1) / 2));
        }

        [TestCase(true)]
        [TestCase(false)]
        public void InvalidDistanceInParallelThrowsArgumentException(bool storeDistanceMatrix)
        {
            var xs = ReadIris("iris.csv").ToArray();
            var options = new HdbScanOptions { StoreDistanceMatrix = storeDistanceMatrix, MaxDegreeOfParallelism = 4 };

            Assert.Throws<ArgumentException>(() => new HdbScan<double[]>(xs, (a, b) => double.NaN, options));
        }

        [Test]
        public void TooFewPoints_KeepsPredictionData()
        {
            double[][] xs = [[0, 0], [1, 1]];

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

            Assert.That(model.HasPredictionData, Is.True);
            Assert.That(model.PredictWithProbability([0, 0]), Is.EqualTo((-1, 0.0)));
        }

        /// <summary>
        /// A new point far away from all training data is noise, whatever the number of clusters.
        /// </summary>
        [TestCase(false)]
        [TestCase(true)]
        public void Predict_FarAwayPointIsNoise(bool singleCluster)
        {
            var xs = singleCluster ? ReadIris("iris.csv").Take(50).ToArray() : ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions { MinClusterSize = 5, AllowSingleCluster = singleCluster };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);
            Assert.That(model.ClusterCount, Is.EqualTo(singleCluster ? 1 : 2));

            Assert.That(model.PredictWithProbability([1e6, 1e6, 1e6, 1e6]), Is.EqualTo((-1, 0.0)));
            Assert.That(model.PredictWithProbability([8, 8, 8, 8]), Is.EqualTo((-1, 0.0)));
        }

        /// <summary>
        /// Predicting a training point must give back exactly its fitted label and probability,
        /// noise points included.
        /// </summary>
        [TestCase(ClusterSelectionMethod.ExcessOfMass)]
        [TestCase(ClusterSelectionMethod.Leaf)]
        public void Predict_TrainingPointsReproduceFit(ClusterSelectionMethod method)
        {
            var xs = ReadXY("iris_kpca_poly_d2c1.csv").ToArray();

            var options = new HdbScanOptions { MinClusterSize = 3, ClusterSelectionMethod = method };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);
            Assert.That(model.Labels, Has.Some.EqualTo(-1), "The data set should contain noise");

            for (var i = 0; i < xs.Length; i++)
            {
                Assert.That(model.PredictWithProbability(xs[i]), Is.EqualTo((model.Labels[i], model.Probabilities[i])), $"Point {i}");
            }
        }

        /// <summary>
        /// A point between a cluster's members is predicted into that cluster.
        /// </summary>
        [Test]
        public void Predict_PointInsideCluster()
        {
            var xs = ReadIris("iris.csv").ToArray();

            var options = new HdbScanOptions { MinClusterSize = 5 };
            var model = new HdbScan<double[]>(xs, EuclideanDistance, options, predictionData: true);

            // Midpoints of training points 0-1 (setosa) and 100-101 (virginica).
            var (setosaLabel, setosaProb) = model.PredictWithProbability(Midpoint(xs[0], xs[1]));
            Assert.That(setosaLabel, Is.EqualTo(model.Labels[0]));
            Assert.That(setosaProb, Is.InRange(0.0, 1.0).And.GreaterThan(0.0));

            var (otherLabel, otherProb) = model.PredictWithProbability(Midpoint(xs[100], xs[101]));
            Assert.That(otherLabel, Is.EqualTo(model.Labels[100]));
            Assert.That(otherProb, Is.InRange(0.0, 1.0).And.GreaterThan(0.0));

            static double[] Midpoint(double[] a, double[] b) => a.Zip(b, (x, y) => (x + y) / 2).ToArray();
        }
    }
}
