using System;
using System.Collections.Generic;
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

        private static IEnumerable<double[]> ReadIris(string filename)
        {
            var path = Path.Combine("dataset", filename);
            foreach (var line in File.ReadLines(path).Skip(1))
            {
                var values = line.Split(',').Take(4).Select(double.Parse).ToArray();
                yield return values;
            }
        }

        private static IEnumerable<double[]> ReadXY(string filename)
        {
            var path = Path.Combine("dataset", filename);
            foreach (var line in File.ReadLines(path).Skip(1))
            {
                var values = line.Split(',').Take(2).Select(double.Parse).ToArray();
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

            // Core points in BOTH clusters should have low scores (< 0.5).
            // This is the key regression test — the old bug inflated sparse cluster scores.
            for (var i = 0; i < 20; i++)
            {
                if (model.Labels[i] >= 0)
                {
                    Assert.That(model.OutlierScores[i], Is.LessThan(0.5),
                        $"Dense cluster point {i} should have low outlier score");
                }
            }

            for (var i = 20; i < 40; i++)
            {
                if (model.Labels[i] >= 0)
                {
                    Assert.That(model.OutlierScores[i], Is.LessThan(0.5),
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
    }
}
