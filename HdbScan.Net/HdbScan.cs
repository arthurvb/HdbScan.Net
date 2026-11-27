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
        private readonly int minSamples;
        private readonly HdbScanAlgorithm.ClusterSelection? selection;

        /// <summary>
        /// Applies the HDBSCAN algorithm to the given set of points.
        /// </summary>
        /// <param name="points">
        /// The collection of points to be clustered.
        /// </param>
        /// <param name="distanceMetric">
        /// The distance metric to calculate the distance between points.
        /// It must be symmetric and return finite, non-negative values. It is called once
        /// per pair of distinct points; the distance from a point to itself is taken as 0.
        /// </param>
        /// <param name="options">
        /// Specifies options for HDBSCAN. If null, default options are used.
        /// </param>
        /// <param name="predictionData">
        /// If true, stores data needed for predicting cluster membership of new points.
        /// </param>
        /// <exception cref="ArgumentException">
        /// <paramref name="points"/> is empty, or <paramref name="distanceMetric"/> returned a
        /// NaN, infinite or negative distance.
        /// </exception>
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
            minSamples = options.MinSamples;

            if (predictionData)
            {
                this.points = points.ToArray();
                this.distanceMetric = distanceMetric;
            }

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

            // Steps 1-3 are the only ones that look at distances. Storing all n×n of them is
            // fastest; when that is switched off, or n×n is more than one array can hold
            // (n > 46,340), they are computed on demand instead. Both give the same result.
            var parallelism = options.MaxDegreeOfParallelism;
            HdbScanAlgorithm.Distances distances = options.StoreDistanceMatrix && (long)n * n <= Array.MaxLength
                ? new HdbScanAlgorithm.DistanceMatrix(ComputeDistanceMatrix(points, distanceMetric, parallelism), n)
                : new OnDemandDistances(points.ToArray(), distanceMetric);

            var coreDistances = HdbScanAlgorithm.ComputeCoreDistances(distances, minSamples, parallelism);

            var mst = HdbScanAlgorithm.BuildMst(distances, coreDistances, parallelism);

            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var singleLinkageTree = HdbScanAlgorithm.BuildSingleLinkageTree(mst, n);

            var condensedTree = HdbScanAlgorithm.CondenseTree(singleLinkageTree, minClusterSize);

            var (clusterLabels, clusterProbs, clusterSelection) = HdbScanAlgorithm.ExtractClusters(
                condensedTree, n, options.ClusterSelectionMethod, options.AllowSingleCluster);

            var scores = HdbScanAlgorithm.ComputeOutlierScores(condensedTree, n);

            labels = clusterLabels;
            probabilities = clusterProbs;
            outlierScores = scores;
            clusterCount = clusterSelection.ClusterCount;

            if (predictionData)
            {
                this.coreDistances = coreDistances;
                selection = clusterSelection;
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
        /// <exception cref="ArgumentException">
        /// The distance metric returned a NaN, infinite or negative distance.
        /// </exception>
        /// <remarks>
        /// See <see cref="PredictWithProbability"/> for how the prediction is made.
        /// </remarks>
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
        /// The predicted cluster label (-1 for noise) and membership probability (0 for noise).
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Prediction data was not stored during fitting.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// The distance metric returned a NaN, infinite or negative distance.
        /// </exception>
        /// <remarks>
        /// <para>
        /// The clustering itself is not changed. The new point is attached to the cluster
        /// hierarchy through its nearest training point in mutual reachability distance, and
        /// then labelled like a training point in that position: it gets a cluster if it is
        /// dense enough to be a member, and is noise otherwise. The probability is on the same
        /// scale as <see cref="Probabilities"/>.
        /// </para>
        /// <para>
        /// This is the approximate_predict method of the scikit-learn-contrib/hdbscan library.
        /// The result can differ from re-clustering with the new point included.
        /// </para>
        /// </remarks>
        public (int Label, double Probability) PredictWithProbability(T x)
        {
            if (points == null || distanceMetric == null)
            {
                throw new InvalidOperationException("Prediction data was not stored. Set predictionData=true when fitting.");
            }

            // No clusters (or too few points to cluster at all): every new point is noise.
            if (selection == null || coreDistances == null || clusterCount == 0)
            {
                return (-1, 0);
            }

            var distances = new double[points.Length];
            for (var i = 0; i < points.Length; i++)
            {
                distances[i] = CheckDistance(distanceMetric(x, points[i]));
            }

            return selection.Predict(distances, coreDistances, minSamples);
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
        private static double[] ComputeDistanceMatrix(IReadOnlyList<T> points, Func<T, T, double> dm, int parallelism)
        {
            var n = points.Count;
            var matrix = new double[n * n];

            // Every (i, j) pair is written by the iteration of its smaller index only,
            // so the iterations can run in parallel.
            HdbScanAlgorithm.For(n, parallelism, i =>
            {
                for (var j = i + 1; j < n; j++)
                {
                    var d = CheckDistance(dm(points[i], points[j]));
                    matrix[i * n + j] = d;
                    matrix[j * n + i] = d;
                }
            });

            return matrix;
        }

        /// <summary>
        /// Distances that are not stored but computed each time a row is asked for, so no
        /// memory of size n² is needed. Calls dm(i, j) with i &lt; j, exactly like
        /// <see cref="ComputeDistanceMatrix"/>, so both give the same values.
        /// </summary>
        private sealed class OnDemandDistances(T[] points, Func<T, T, double> dm)
            : HdbScanAlgorithm.Distances(points.Length)
        {
            public override ReadOnlySpan<double> Row(int i, double[] buffer, bool[]? skip = null, int parallelism = 1)
            {
                HdbScanAlgorithm.For(points.Length, parallelism, j =>
                {
                    if (skip == null || !skip[j])
                    {
                        buffer[j] = i == j ? 0.0 : CheckDistance(i < j ? dm(points[i], points[j]) : dm(points[j], points[i]));
                    }
                });

                return buffer;
            }
        }

        /// <summary>
        /// The algorithm is only defined for real distances. A NaN, infinite or negative value
        /// would not fail by itself but silently produce a meaningless clustering, so reject it.
        /// </summary>
        private static double CheckDistance(double d)
        {
            if (!double.IsFinite(d) || d < 0)
            {
                throw new ArgumentException(
                    $"The distance metric returned {d}. Distances must be finite and non-negative.",
                    nameof(distanceMetric));
            }

            return d;
        }
    }
}
