using System;
using System.Collections.Generic;
using System.Linq;

namespace HdbScan.Net
{
    /// <summary>
    /// Provides the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm.
    /// </summary>
    /// <typeparam name="T">
    /// The type of the points.
    /// </typeparam>
    /// <remarks>
    /// <para>
    /// HDBSCAN extends DBSCAN by building a hierarchy of clusterings at all density levels
    /// and extracting a flat clustering based on cluster stability. Unlike k-means or GMM,
    /// it does not require specifying the number of clusters and can identify noise points.
    /// </para>
    /// <para>
    /// The algorithm computes mutual reachability distances, builds a minimum spanning tree,
    /// converts it to a hierarchical tree, condenses small clusters, and extracts the final
    /// clustering using either the Excess of Mass (EOM) or Leaf method.
    /// </para>
    /// <para>
    /// Reference: Campello, R.J.G.B., Moulavi, D., Zimek, A., Sander, J. (2015).
    /// "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection."
    /// ACM Trans. Knowl. Discov. Data 10, 1, Article 5 (July 2015).
    /// https://doi.org/10.1145/2733381
    /// </para>
    /// </remarks>
    public sealed class HdbScan<T>
    {
        private readonly int[] labels;
        private readonly double[] probabilities;
        private readonly double[] outlierScores;
        private readonly int clusterCount;

        private readonly T[]? points;
        private readonly double[]? coreDistances;
        private readonly Func<T, T, double>? distanceMetric;

        /// <summary>
        /// Applies the HDBSCAN algorithm to the given set of points.
        /// </summary>
        /// <param name="points">
        /// The collection of points to be clustered.
        /// </param>
        /// <param name="distanceMetric">
        /// The distance metric to calculate the distance between points.
        /// </param>
        /// <param name="options">
        /// Specifies options for HDBSCAN. If null, default options are used.
        /// </param>
        /// <param name="predictionData">
        /// If true, stores data needed for predicting cluster membership of new points.
        /// </param>
        public HdbScan(IReadOnlyList<T> points, Func<T, T, double> distanceMetric, HdbScanOptions? options = null, bool predictionData = false)
        {
            ArgumentNullException.ThrowIfNull(points);
            ArgumentNullException.ThrowIfNull(distanceMetric);

            if (points.Count == 0)
            {
                throw new ArgumentException("The sequence must contain at least one point.", nameof(points));
            }

            options ??= new HdbScanOptions();
            var minClusterSize = options.MinClusterSize;
            var minSamples = options.MinSamples;

            if (points.Count < minClusterSize)
            {
                labels = new int[points.Count];
                probabilities = new double[points.Count];
                outlierScores = new double[points.Count];
                Array.Fill(labels, -1);
                Array.Fill(outlierScores, 1.0);
                clusterCount = 0;
                return;
            }

            var n = points.Count;

            // Algorithm pipeline (Campello et al. 2015, ACM TKDD 10(1), doi:10.1145/2733381):
            //
            //   1. Compute core distances           (Definition 3.1)
            //   2. Build mutual reachability graph   (Definitions 3.2–3.3)
            //   3. Compute MST (Minimum Spanning Tree) of that graph (Proposition 3.4, Section 3.2)
            //   4. Build hierarchical clustering     (Algorithm 1, single-linkage dendrogram)
            //   5. Condense the dendrogram           (Section 3.3, Algorithm 2)
            //   6. Extract flat clustering           (Section 5.2, Algorithm 3)
            //   7. Compute outlier scores            (Section 6, Algorithm 4 — GLOSH)

            var distMatrix = ComputeDistanceMatrix(points, distanceMetric);
            var coreDistances = HdbScanAlgorithm.ComputeCoreDistances(distMatrix, n, minSamples);

            var mst = HdbScanAlgorithm.BuildMst(distMatrix, n, coreDistances);

            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var singleLinkageTree = HdbScanAlgorithm.BuildSingleLinkageTree(mst, n);

            var condensedTree = HdbScanAlgorithm.CondenseTree(singleLinkageTree, minClusterSize);

            var (clusterLabels, clusterProbs, numClusters) = HdbScanAlgorithm.ExtractClusters(
                condensedTree, n, options.ClusterSelectionMethod, options.AllowSingleCluster);

            var scores = HdbScanAlgorithm.ComputeOutlierScores(condensedTree, n);

            labels = clusterLabels;
            probabilities = clusterProbs;
            outlierScores = scores;
            clusterCount = numClusters;

            if (predictionData)
            {
                this.points = points.ToArray();
                this.coreDistances = coreDistances;
                this.distanceMetric = distanceMetric;
            }
        }

        /// <summary>
        /// Predicts the cluster membership of a new point.
        /// </summary>
        /// <param name="x">
        /// The point to classify.
        /// </param>
        /// <returns>
        /// The predicted cluster label, or -1 for noise.
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Prediction data was not stored during fitting.
        /// </exception>
        public int Predict(T x)
        {
            return PredictWithProbability(x).Label;
        }

        /// <summary>
        /// Predicts the cluster membership and probability of a new point.
        /// </summary>
        /// <param name="x">
        /// The point to classify.
        /// </param>
        /// <returns>
        /// The predicted cluster label and membership probability.
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Prediction data was not stored during fitting.
        /// </exception>
        /// <remarks>
        /// Prediction uses approximate soft clustering based on mutual reachability distance
        /// to the nearest point in each cluster.
        /// </remarks>
        public (int Label, double Probability) PredictWithProbability(T x)
        {
            if (points == null || coreDistances == null || distanceMetric == null)
            {
                throw new InvalidOperationException("Prediction data was not stored. Set predictionData=true when fitting.");
            }

            if (clusterCount == 0)
            {
                return (-1, 0);
            }

            // For each cluster, find the minimum mutual reachability distance from x
            // to any core point in that cluster. The new point's core distance is unknown,
            // so we use a one-sided MRD: max(core(training_point), d(x, training_point)).
            // This matches sklearn's approximate_predict approach.
            var clusterDistances = new double[clusterCount];
            for (var i = 0; i < clusterCount; i++)
            {
                clusterDistances[i] = double.MaxValue;
            }

            for (var i = 0; i < points.Length; i++)
            {
                if (labels[i] >= 0)
                {
                    var dist = distanceMetric(x, points[i]);
                    var mrd = Math.Max(coreDistances[i], dist);

                    if (mrd < clusterDistances[labels[i]])
                    {
                        clusterDistances[labels[i]] = mrd;
                    }
                }
            }

            var bestCluster = -1;
            var bestDist = double.MaxValue;
            for (var i = 0; i < clusterCount; i++)
            {
                if (clusterDistances[i] < bestDist)
                {
                    bestDist = clusterDistances[i];
                    bestCluster = i;
                }
            }

            if (bestCluster == -1)
            {
                return (-1, 0);
            }

            // Soft assignment: probability proportional to inverse distance.
            const double epsilon = 1e-10;
            var prob = 1.0;
            if (clusterCount > 1)
            {
                var sumInv = 0.0;
                for (var i = 0; i < clusterCount; i++)
                {
                    if (clusterDistances[i] < double.MaxValue)
                    {
                        sumInv += 1.0 / (clusterDistances[i] + epsilon);
                    }
                }
                prob = (1.0 / (bestDist + epsilon)) / sumInv;
            }

            return (bestCluster, prob);
        }

        /// <summary>
        /// Gets the cluster labels for each point.
        /// A label of -1 indicates noise.
        /// </summary>
        public IReadOnlyList<int> Labels => labels;

        /// <summary>
        /// Gets the membership probability for each point in its assigned cluster.
        /// </summary>
        public IReadOnlyList<double> Probabilities => probabilities;

        /// <summary>
        /// Gets the outlier score for each point.
        /// Higher values indicate more outlier-like points.
        /// </summary>
        public IReadOnlyList<double> OutlierScores => outlierScores;

        /// <summary>
        /// Gets the number of clusters found.
        /// </summary>
        public int ClusterCount => clusterCount;

        /// <summary>
        /// Gets whether prediction data is available.
        /// </summary>
        public bool HasPredictionData => points != null;

        /// <summary>
        /// Computes the full n×n pairwise distance matrix, exploiting symmetry (only calls
        /// dm(i,j) for i &lt; j, then mirrors). Stored as a flat double[] in row-major order
        /// for cache-friendly access. Diagonal entries are 0.
        /// </summary>
        private static double[] ComputeDistanceMatrix(IReadOnlyList<T> points, Func<T, T, double> dm)
        {
            var n = points.Count;
            var matrix = new double[n * n];

            for (var i = 0; i < n; i++)
            {
                for (var j = i + 1; j < n; j++)
                {
                    var d = dm(points[i], points[j]);
                    matrix[i * n + j] = d;
                    matrix[j * n + i] = d;
                }
            }

            return matrix;
        }
    }
}
