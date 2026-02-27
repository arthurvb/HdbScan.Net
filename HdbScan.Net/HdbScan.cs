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
    /// Reference: Campello, R.J.G.B., Moulavi, D., Sander, J. (2013).
    /// "Density-Based Clustering Based on Hierarchical Density Estimates."
    /// In: Pei, J., Tseng, V.S., Cao, L., Motoda, H., Xu, G. (eds) PAKDD 2013.
    /// Lecture Notes in Computer Science, vol 7819. Springer, Berlin, Heidelberg.
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
                for (var i = 0; i < points.Count; i++)
                {
                    labels[i] = -1;
                    probabilities[i] = 0;
                    outlierScores[i] = 1;
                }
                clusterCount = 0;
                return;
            }

            var n = points.Count;

            // Compute core distances.
            var coreDistances = ComputeCoreDistances(points, distanceMetric, minSamples);

            // Build minimum spanning tree of the mutual reachability graph.
            var mst = BuildMst(points, distanceMetric, coreDistances);

            // Build single-linkage dendrogram.
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var singleLinkageTree = BuildSingleLinkageTree(mst, n);

            // Condense the tree.
            var condensedTree = CondenseTree(singleLinkageTree, minClusterSize);

            // Extract flat clustering.
            var (clusterLabels, clusterProbs, numClusters) = ExtractClusters(
                condensedTree, n, options.ClusterSelectionMethod, options.AllowSingleCluster);

            // Compute outlier scores.
            var scores = ComputeOutlierScores(condensedTree, clusterLabels, n);

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

        private static double[] ComputeCoreDistances(IReadOnlyList<T> points, Func<T, T, double> dm, int k)
        {
            var n = points.Count;
            var coreDistances = new double[n];
            k = Math.Min(k, n);

            for (var i = 0; i < n; i++)
            {
                var distances = new double[n];
                for (var j = 0; j < n; j++)
                {
                    distances[j] = i == j ? 0 : dm(points[i], points[j]);
                }
                Array.Sort(distances);
                coreDistances[i] = distances[k - 1];
            }

            return coreDistances;
        }

        private static MstEdge[] BuildMst(IReadOnlyList<T> points, Func<T, T, double> dm, double[] coreDistances)
        {
            var n = points.Count;
            var inMst = new bool[n];
            var minDist = new double[n];
            var minEdge = new int[n];
            var edges = new List<MstEdge>(n - 1);

            for (var i = 0; i < n; i++)
            {
                minDist[i] = double.MaxValue;
                minEdge[i] = -1;
            }

            inMst[0] = true;
            for (var j = 1; j < n; j++)
            {
                var d = MutualReachabilityDistance(points, dm, coreDistances, 0, j);
                minDist[j] = d;
                minEdge[j] = 0;
            }

            for (var step = 0; step < n - 1; step++)
            {
                var minVal = double.MaxValue;
                var minIdx = -1;
                for (var i = 0; i < n; i++)
                {
                    if (!inMst[i] && minDist[i] < minVal)
                    {
                        minVal = minDist[i];
                        minIdx = i;
                    }
                }

                if (minIdx == -1) break;

                inMst[minIdx] = true;
                edges.Add(new MstEdge(minEdge[minIdx], minIdx, minVal));

                for (var j = 0; j < n; j++)
                {
                    if (!inMst[j])
                    {
                        var d = MutualReachabilityDistance(points, dm, coreDistances, minIdx, j);
                        if (d < minDist[j])
                        {
                            minDist[j] = d;
                            minEdge[j] = minIdx;
                        }
                    }
                }
            }

            return edges.ToArray();
        }

        private static double MutualReachabilityDistance(IReadOnlyList<T> points, Func<T, T, double> dm,
            double[] coreDistances, int i, int j)
        {
            var dist = dm(points[i], points[j]);
            return Math.Max(Math.Max(coreDistances[i], coreDistances[j]), dist);
        }

        private static SingleLinkageNode[] BuildSingleLinkageTree(MstEdge[] sortedMst, int n)
        {
            var uf = new UnionFind(n);
            var tree = new List<SingleLinkageNode>();
            var nextLabel = n;

            foreach (var edge in sortedMst)
            {
                var rootA = uf.Find(edge.A);
                var rootB = uf.Find(edge.B);

                if (rootA != rootB)
                {
                    var sizeA = uf.GetSize(rootA);
                    var sizeB = uf.GetSize(rootB);
                    var newNode = new SingleLinkageNode(
                        uf.GetLabel(rootA),
                        uf.GetLabel(rootB),
                        edge.Distance,
                        sizeA + sizeB);
                    tree.Add(newNode);
                    uf.Union(rootA, rootB, nextLabel);
                    nextLabel++;
                }
            }

            return tree.ToArray();
        }

        private static List<CondensedTreeEdge> CondenseTree(SingleLinkageNode[] singleLinkageTree, int minClusterSize)
        {
            const double epsilon = 1e-10;

            var n = singleLinkageTree.Length + 1;
            var condensed = new List<CondensedTreeEdge>();

            var nodeInfo = new Dictionary<int, (int Left, int Right, double Lambda, int Size)>();
            var nodeLabel = n;
            foreach (var node in singleLinkageTree)
            {
                nodeInfo[nodeLabel] = (node.Left, node.Right, 1.0 / (node.Distance + epsilon), node.Size);
                nodeLabel++;
            }

            var root = nodeLabel - 1;
            var relabel = new Dictionary<int, int> { [root] = n };
            var nextCondensedLabel = n + 1;

            void ProcessNode(int node, int parent)
            {
                if (node < n)
                {
                    var lambda = nodeInfo.ContainsKey(parent) ? nodeInfo[parent].Lambda : 0;
                    condensed.Add(new CondensedTreeEdge(relabel[parent], node, lambda, 1));
                    return;
                }

                var (left, right, lambdaVal, size) = nodeInfo[node];
                var leftSize = left < n ? 1 : nodeInfo[left].Size;
                var rightSize = right < n ? 1 : nodeInfo[right].Size;

                if (leftSize >= minClusterSize && rightSize >= minClusterSize)
                {
                    relabel[left] = nextCondensedLabel++;
                    relabel[right] = nextCondensedLabel++;
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[left], lambdaVal, leftSize));
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[right], lambdaVal, rightSize));
                    ProcessNode(left, left);
                    ProcessNode(right, right);
                }
                else if (leftSize >= minClusterSize)
                {
                    relabel[left] = relabel[parent];
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                    ProcessNode(left, parent);
                }
                else if (rightSize >= minClusterSize)
                {
                    relabel[right] = relabel[parent];
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    ProcessNode(right, parent);
                }
                else
                {
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                }
            }

            void FallOutPoints(int node, double lambda, int parentCluster)
            {
                if (node < n)
                {
                    condensed.Add(new CondensedTreeEdge(parentCluster, node, lambda, 1));
                }
                else
                {
                    var (left, right, _, _) = nodeInfo[node];
                    FallOutPoints(left, lambda, parentCluster);
                    FallOutPoints(right, lambda, parentCluster);
                }
            }

            ProcessNode(root, root);
            return condensed;
        }

        private static (int[] Labels, double[] Probabilities, int ClusterCount) ExtractClusters(
            List<CondensedTreeEdge> condensedTree, int n, ClusterSelectionMethod method, bool allowSingleCluster)
        {
            var labels = new int[n];
            var probabilities = new double[n];
            for (var i = 0; i < n; i++) labels[i] = -1;

            if (condensedTree.Count == 0)
            {
                return (labels, probabilities, 0);
            }

            var clusters = condensedTree
                .Select(e => e.SourceCluster)
                .Distinct()
                .ToHashSet();

            if (clusters.Count == 1 && !allowSingleCluster)
            {
                return (labels, probabilities, 0);
            }

            HashSet<int> selectedClusters;
            if (method == ClusterSelectionMethod.Leaf)
            {
                var clustersWithChildren = condensedTree
                    .Where(e => e.Target >= n)
                    .Select(e => e.SourceCluster)
                    .ToHashSet();
                selectedClusters = clusters.Where(c => !clustersWithChildren.Contains(c) || c == n && clusters.Count == 1).ToHashSet();
            }
            else
            {
                selectedClusters = SelectClustersEom(condensedTree, clusters, n, allowSingleCluster);
            }

            if (selectedClusters.Count == 0)
            {
                return (labels, probabilities, 0);
            }

            var clusterMaxLambda = new Dictionary<int, double>();
            foreach (var cluster in selectedClusters)
            {
                var maxLambda = condensedTree
                    .Where(e => e.SourceCluster == cluster && e.Target < n)
                    .Select(e => e.Lambda)
                    .DefaultIfEmpty(0)
                    .Max();
                clusterMaxLambda[cluster] = maxLambda;
            }

            var clusterToLabel = selectedClusters.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

            foreach (var edge in condensedTree.Where(e => e.Target < n))
            {
                var point = edge.Target;
                var cluster = edge.SourceCluster;

                while (!selectedClusters.Contains(cluster))
                {
                    var parentEdge = condensedTree.FirstOrDefault(e => e.Target == cluster);
                    if (parentEdge.SourceCluster == 0 && parentEdge.Target == 0) break;
                    cluster = parentEdge.SourceCluster;
                }

                if (selectedClusters.Contains(cluster))
                {
                    labels[point] = clusterToLabel[cluster];
                    var maxLambda = clusterMaxLambda[cluster];
                    probabilities[point] = maxLambda > 0 ? Math.Min(1.0, edge.Lambda / maxLambda) : 1.0;
                }
            }

            return (labels, probabilities, selectedClusters.Count);
        }

        private static HashSet<int> SelectClustersEom(List<CondensedTreeEdge> condensedTree, HashSet<int> clusters,
            int n, bool allowSingleCluster)
        {
            var stability = new Dictionary<int, double>();
            foreach (var cluster in clusters)
            {
                var birthLambda = condensedTree
                    .Where(e => e.Target == cluster)
                    .Select(e => e.Lambda)
                    .DefaultIfEmpty(0)
                    .First();

                var stab = 0.0;
                foreach (var edge in condensedTree.Where(e => e.SourceCluster == cluster))
                {
                    stab += (edge.Lambda - birthLambda) * edge.Size;
                }

                stability[cluster] = Math.Max(0, stab);
            }

            var childClusters = new Dictionary<int, List<int>>();
            foreach (var cluster in clusters)
            {
                childClusters[cluster] = condensedTree
                    .Where(e => e.SourceCluster == cluster && e.Target >= n)
                    .Select(e => e.Target)
                    .ToList();
            }

            var selected = new HashSet<int>();
            var processed = new HashSet<int>();

            var leafClusters = clusters.Where(c => childClusters[c].Count == 0).ToList();
            foreach (var leaf in leafClusters)
            {
                selected.Add(leaf);
                processed.Add(leaf);
            }

            void RemoveDescendants(int cluster)
            {
                selected.Remove(cluster);
                if (childClusters.ContainsKey(cluster))
                {
                    foreach (var child in childClusters[cluster])
                    {
                        RemoveDescendants(child);
                    }
                }
            }

            var queue = new Queue<int>(clusters.Where(c => !processed.Contains(c)));
            while (queue.Count > 0)
            {
                var cluster = queue.Dequeue();
                var children = childClusters[cluster];

                if (children.All(processed.Contains))
                {
                    var childStability = children.Sum(c => stability[c]);
                    if (stability[cluster] > childStability)
                    {
                        foreach (var child in children)
                        {
                            RemoveDescendants(child);
                        }
                        selected.Add(cluster);
                    }
                    else
                    {
                        stability[cluster] = childStability;
                    }
                    processed.Add(cluster);
                }
                else
                {
                    queue.Enqueue(cluster);
                }
            }

            if (selected.Contains(n))
            {
                if (selected.Count == 1)
                {
                    if (!allowSingleCluster)
                    {
                        selected.Clear();
                    }
                }
                else
                {
                    selected.Remove(n);
                }
            }

            return selected;
        }

        private static double[] ComputeOutlierScores(List<CondensedTreeEdge> condensedTree, int[] labels, int n)
        {
            var scores = new double[n];

            var maxLambda = condensedTree.Count > 0
                ? condensedTree.Max(e => e.Lambda)
                : 1.0;

            for (var i = 0; i < n; i++)
            {
                if (labels[i] == -1)
                {
                    scores[i] = 1.0;
                }
                else
                {
                    var pointEdge = condensedTree.FirstOrDefault(e => e.Target == i);
                    if (pointEdge.Lambda > 0)
                    {
                        scores[i] = 1.0 - (pointEdge.Lambda / maxLambda);
                    }
                }
            }

            return scores;
        }



        // ================================================================================
        // Internal data structures
        //
        // The HDBSCAN algorithm uses several intermediate tree representations:
        //
        // 1. MstEdge: Edges in the minimum spanning tree of the mutual reachability graph.
        //
        // 2. SingleLinkageNode: Nodes in the hierarchical clustering dendrogram.
        //    Each node represents the merge of two sub-clusters at a given distance.
        //    Node IDs: points are 0..n-1, internal nodes are n, n+1, n+2, ...
        //
        // 3. CondensedTreeEdge: The condensed tree stored as a list of directed edges.
        //    Each edge goes from a SourceCluster to a Target, which is either:
        //    - Another cluster (Target >= n): a cluster split
        //    - A point (Target < n): a point "falling out" at some density level
        //
        // 4. UnionFind: Disjoint set data structure for building the dendrogram.
        // ================================================================================

        private readonly struct MstEdge(int a, int b, double distance)
        {
            public readonly int A = a;
            public readonly int B = b;
            public readonly double Distance = distance;
        }

        private readonly struct SingleLinkageNode(int left, int right, double distance, int size)
        {
            public readonly int Left = left;
            public readonly int Right = right;
            public readonly double Distance = distance;
            public readonly int Size = size;
        }

        private readonly struct CondensedTreeEdge(int sourceCluster, int target, double lambda, int size)
        {
            public readonly int SourceCluster = sourceCluster;
            public readonly int Target = target;
            public readonly double Lambda = lambda;
            public readonly int Size = size;
        }

        private sealed class UnionFind
        {
            private readonly int[] parent;
            private readonly int[] size;
            private readonly int[] label;

            public UnionFind(int n)
            {
                parent = new int[2 * n];
                size = new int[2 * n];
                label = new int[2 * n];

                for (var i = 0; i < 2 * n; i++)
                {
                    parent[i] = i;
                    size[i] = i < n ? 1 : 0;
                    label[i] = i;
                }
            }

            public int Find(int x)
            {
                if (parent[x] != x)
                {
                    parent[x] = Find(parent[x]);
                }
                return parent[x];
            }

            public void Union(int x, int y, int newLabel)
            {
                var rootX = Find(x);
                var rootY = Find(y);

                if (rootX != rootY)
                {
                    var newSize = size[rootX] + size[rootY];
                    parent[rootX] = newLabel;
                    parent[rootY] = newLabel;
                    parent[newLabel] = newLabel;
                    size[newLabel] = newSize;
                    label[newLabel] = newLabel;
                }
            }

            public int GetSize(int x) => size[Find(x)];
            public int GetLabel(int x) => label[Find(x)];
        }
    }
}
