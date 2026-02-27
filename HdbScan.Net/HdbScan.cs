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

            // Algorithm pipeline (Paper, Section 3):
            //
            //   1. Compute core distances           (Definition 3)
            //   2. Build mutual reachability graph   (Definition 4)
            //   3. Compute MST of that graph         (Theorem 1 — equivalent to MST_k)
            //   4. Build hierarchical clustering     (single-linkage dendrogram)
            //   5. Condense the dendrogram           (Section 4 — extract condensed tree)
            //   6. Extract flat clustering           (FOSC framework, Section 5)
            //   7. Compute outlier scores            (GLOSH — Campello et al. 2015)

            var coreDistances = ComputeCoreDistances(points, distanceMetric, minSamples);

            var mst = BuildMst(points, distanceMetric, coreDistances);

            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var singleLinkageTree = BuildSingleLinkageTree(mst, n);

            var condensedTree = CondenseTree(singleLinkageTree, minClusterSize);

            var (clusterLabels, clusterProbs, numClusters) = ExtractClusters(
                condensedTree, n, options.ClusterSelectionMethod, options.AllowSingleCluster);

            var scores = ComputeOutlierScores(condensedTree, n);

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
        /// Paper, Definition 3: core_k(x) = distance to the k-th nearest neighbor (including x).
        /// After sorting all n distances (where distances[0] = 0 for self), distances[k-1] is
        /// the k-th nearest including self. Matches sklearn: tree.query(X, k=min_samples)[0][:, -1].
        /// </summary>
        private static double[] ComputeCoreDistances(IReadOnlyList<T> points, Func<T, T, double> dm, int k)
        {
            var n = points.Count;
            var coreDistances = new double[n];
            k = Math.Min(k, n);

            var distances = new double[n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    distances[j] = i == j ? 0 : dm(points[i], points[j]);
                }
                Array.Sort(distances);
                coreDistances[i] = distances[k - 1];
            }

            return coreDistances;
        }

        /// <summary>
        /// Paper, Theorem 1: the MST of the mutual reachability graph equals MST_k, the
        /// "extended minimum spanning tree" that encodes all density-based hierarchical
        /// clusterings. Built via Prim's algorithm in O(n^2), matching sklearn's generic path.
        /// </summary>
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

        /// <summary>
        /// Paper, Definition 4: d_mreach(a, b) = max(core(a), core(b), d(a, b)).
        /// Effectively "pushes apart" points in sparse regions while preserving distances
        /// in dense regions.
        /// </summary>
        private static double MutualReachabilityDistance(IReadOnlyList<T> points, Func<T, T, double> dm,
            double[] coreDistances, int i, int j)
        {
            var dist = dm(points[i], points[j]);
            return Math.Max(Math.Max(coreDistances[i], coreDistances[j]), dist);
        }

        /// <summary>
        /// Converts the sorted MST into a single-linkage dendrogram using Union-Find.
        /// Produces the same structure as scipy.cluster.hierarchy.linkage: each row is
        /// (left, right, distance, size). Points are 0..n-1; internal nodes are n, n+1, ...
        /// </summary>
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

        /// <summary>
        /// Paper, Section 4: walks the dendrogram top-down, pruning splits where a child has
        /// fewer than minClusterSize points. Uses lambda = 1/distance as the density level.
        ///
        /// At each split:
        ///   - Both children large enough  -> genuine split, two new child clusters
        ///   - One child too small          -> small side's points "fall out"; large side inherits
        ///   - Both too small               -> all points fall out of the parent cluster
        ///
        /// The output is a list of directed edges (parent -> child_cluster or parent -> point).
        /// </summary>
        private static List<CondensedTreeEdge> CondenseTree(SingleLinkageNode[] singleLinkageTree, int minClusterSize)
        {
            const double epsilon = 1e-10;

            var n = singleLinkageTree.Length + 1;
            var condensed = new List<CondensedTreeEdge>();

            // Map each dendrogram node to (left, right, lambda, size).
            // Lambda = 1/(distance + epsilon) converts distance to density level.
            var nodeInfo = new Dictionary<int, (int Left, int Right, double Lambda, int Size)>();
            var nodeLabel = n;
            foreach (var node in singleLinkageTree)
            {
                nodeInfo[nodeLabel] = (node.Left, node.Right, 1.0 / (node.Distance + epsilon), node.Size);
                nodeLabel++;
            }

            // Relabel clusters in the condensed tree starting from n (the root).
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
                    // Genuine split: both children become new clusters.
                    relabel[left] = nextCondensedLabel++;
                    relabel[right] = nextCondensedLabel++;
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[left], lambdaVal, leftSize));
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[right], lambdaVal, rightSize));
                    ProcessNode(left, left);
                    ProcessNode(right, right);
                }
                else if (leftSize >= minClusterSize)
                {
                    // Left survives; right's points fall out of the current cluster.
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
                    // Neither child is large enough — both fall out.
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                }
            }

            // Records all leaf points under a too-small subtree as departing at the split's lambda.
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

        private static (Dictionary<int, List<CondensedTreeEdge>> EdgesBySource, Dictionary<int, CondensedTreeEdge> EdgeByTarget)
            BuildCondensedTreeIndex(List<CondensedTreeEdge> condensedTree)
        {
            var edgesBySource = new Dictionary<int, List<CondensedTreeEdge>>();
            var edgeByTarget = new Dictionary<int, CondensedTreeEdge>();

            foreach (var edge in condensedTree)
            {
                if (!edgesBySource.TryGetValue(edge.SourceCluster, out var list))
                {
                    list = new List<CondensedTreeEdge>();
                    edgesBySource[edge.SourceCluster] = list;
                }
                list.Add(edge);
                edgeByTarget[edge.Target] = edge;
            }

            return (edgesBySource, edgeByTarget);
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

            var (edgesBySource, edgeByTarget) = BuildCondensedTreeIndex(condensedTree);

            var clusters = edgesBySource.Keys.ToHashSet();

            if (clusters.Count == 1 && !allowSingleCluster)
            {
                return (labels, probabilities, 0);
            }

            HashSet<int> selectedClusters;
            if (method == ClusterSelectionMethod.Leaf)
            {
                // Leaf method: select clusters that have no child clusters (only point edges).
                var clustersWithChildren = new HashSet<int>();
                foreach (var (source, edges) in edgesBySource)
                {
                    foreach (var edge in edges)
                    {
                        if (edge.Target >= n)
                        {
                            clustersWithChildren.Add(source);
                            break;
                        }
                    }
                }
                selectedClusters = clusters.Where(c => !clustersWithChildren.Contains(c) || c == n && clusters.Count == 1).ToHashSet();
            }
            else
            {
                selectedClusters = SelectClustersEom(edgesBySource, edgeByTarget, clusters, n, allowSingleCluster);
            }

            if (selectedClusters.Count == 0)
            {
                return (labels, probabilities, 0);
            }

            // Max lambda from the selected cluster's direct point edges, used to normalize
            // membership probabilities. Points from descendant sub-clusters (which may have
            // higher lambdas) get clamped to 1.0 — they are the most central cluster members.
            // Matches sklearn's _get_probabilities behavior.
            var clusterMaxLambda = new Dictionary<int, double>();
            foreach (var cluster in selectedClusters)
            {
                var maxLambda = 0.0;
                if (edgesBySource.TryGetValue(cluster, out var edges))
                {
                    foreach (var edge in edges)
                    {
                        if (edge.Target < n && edge.Lambda > maxLambda)
                        {
                            maxLambda = edge.Lambda;
                        }
                    }
                }
                clusterMaxLambda[cluster] = maxLambda;
            }

            var clusterToLabel = selectedClusters.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

            // Assign each point to the nearest selected ancestor in the condensed tree.
            // Probability = lambda_point / max_lambda_cluster, clamped to [0, 1].
            foreach (var edge in condensedTree.Where(e => e.Target < n))
            {
                var point = edge.Target;
                var cluster = edge.SourceCluster;

                // Walk up to the selected cluster (may be the source itself or an ancestor).
                while (!selectedClusters.Contains(cluster))
                {
                    if (!edgeByTarget.TryGetValue(cluster, out var parentEdge)) break;
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

        /// <summary>
        /// Paper, Section 5 / FOSC framework: selects a flat clustering by maximizing
        /// total cluster stability subject to the constraint that selected clusters are
        /// non-overlapping (at most one per root-to-leaf path in the condensed tree).
        ///
        /// Stability of cluster C (Equation 3):
        ///   S(C) = sum over all edges from C of: (lambda_edge - lambda_birth(C)) * edge.Size
        ///
        /// Bottom-up traversal: if a parent's stability exceeds the sum of its children's
        /// stabilities, select the parent and deselect all descendants. Otherwise, propagate
        /// the children's combined stability upward. This greedy approach is provably optimal.
        /// </summary>
        private static HashSet<int> SelectClustersEom(
            Dictionary<int, List<CondensedTreeEdge>> edgesBySource,
            Dictionary<int, CondensedTreeEdge> edgeByTarget,
            HashSet<int> clusters, int n, bool allowSingleCluster)
        {
            // Compute stability for each cluster.
            var stability = new Dictionary<int, double>();
            foreach (var cluster in clusters)
            {
                // Birth lambda = the lambda at which this cluster first appeared.
                // For the root cluster, birth lambda = 0.
                var birthLambda = edgeByTarget.TryGetValue(cluster, out var parentEdge) ? parentEdge.Lambda : 0.0;

                var stab = 0.0;
                if (edgesBySource.TryGetValue(cluster, out var edges))
                {
                    foreach (var edge in edges)
                    {
                        stab += (edge.Lambda - birthLambda) * edge.Size;
                    }
                }

                stability[cluster] = Math.Max(0, stab);
            }

            var childClusters = new Dictionary<int, List<int>>();
            foreach (var cluster in clusters)
            {
                var children = new List<int>();
                if (edgesBySource.TryGetValue(cluster, out var clusterEdges))
                {
                    foreach (var edge in clusterEdges)
                    {
                        if (edge.Target >= n)
                        {
                            children.Add(edge.Target);
                        }
                    }
                }
                childClusters[cluster] = children;
            }

            // Start with all leaf clusters selected, then propagate upward.
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

            // Process non-leaf clusters bottom-up. A cluster is ready once all
            // its children have been processed; re-enqueue if not yet ready.
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
                        // Parent wins: select it, deselect all descendants.
                        foreach (var child in children)
                        {
                            RemoveDescendants(child);
                        }
                        selected.Add(cluster);
                    }
                    else
                    {
                        // Children win: propagate their combined stability upward.
                        stability[cluster] = childStability;
                    }
                    processed.Add(cluster);
                }
                else
                {
                    queue.Enqueue(cluster);
                }
            }

            // Handle the root cluster (id = n). If it's the only selection,
            // respect the allowSingleCluster setting.
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

        /// <summary>
        /// GLOSH (Global-Local Outlier Score from Hierarchies).
        /// Campello et al. (2015), "A framework for optimal selection of clusters."
        ///
        /// For each point: score = 1 - lambda_p / lambda_max(cluster).
        /// lambda_max is propagated upward so each cluster's "death" reflects the peak
        /// density anywhere in its subtree. This makes the score local — a point in a
        /// sparse cluster is compared against that cluster's peak, not a distant dense one.
        ///
        /// Score near 0 = deep inside its cluster. Score near 1 = strong outlier.
        /// Points absent from the condensed tree default to score 1.0.
        /// </summary>
        private static double[] ComputeOutlierScores(List<CondensedTreeEdge> condensedTree, int n)
        {
            var scores = new double[n];
            for (var i = 0; i < n; i++) scores[i] = 1.0;

            if (condensedTree.Count == 0) return scores;

            var deaths = new Dictionary<int, double>();
            var parentOf = new Dictionary<int, int>();
            var pointEdge = new Dictionary<int, CondensedTreeEdge>();

            foreach (var edge in condensedTree)
            {
                if (deaths.TryGetValue(edge.SourceCluster, out var current))
                {
                    if (edge.Lambda > current) deaths[edge.SourceCluster] = edge.Lambda;
                }
                else
                {
                    deaths[edge.SourceCluster] = edge.Lambda;
                }

                if (edge.Target >= n)
                {
                    parentOf[edge.Target] = edge.SourceCluster;
                }
                else
                {
                    pointEdge[edge.Target] = edge;
                }
            }

            // Propagate max lambda from children to parents. Children have higher IDs
            // than parents, so reverse-sorted order guarantees correct propagation.
            var clusterIds = new List<int>(deaths.Keys);
            clusterIds.Sort();
            for (var i = clusterIds.Count - 1; i >= 0; i--)
            {
                var cluster = clusterIds[i];
                if (parentOf.TryGetValue(cluster, out var parent))
                {
                    if (deaths.TryGetValue(parent, out var parentDeath))
                    {
                        if (deaths[cluster] > parentDeath) deaths[parent] = deaths[cluster];
                    }
                    else
                    {
                        deaths[parent] = deaths[cluster];
                    }
                }
            }

            for (var i = 0; i < n; i++)
            {
                if (pointEdge.TryGetValue(i, out var edge))
                {
                    var clusterDeath = deaths.TryGetValue(edge.SourceCluster, out var d) ? d : 1.0;
                    if (clusterDeath > 0)
                    {
                        scores[i] = Math.Max(0, Math.Min(1.0, 1.0 - edge.Lambda / clusterDeath));
                    }
                }
            }

            return scores;
        }



        // ================================================================================
        // Internal data structures
        //
        // The algorithm transforms data through a sequence of representations:
        //
        //   MstEdge[]  ->  SingleLinkageNode[]  ->  List<CondensedTreeEdge>
        //   (MST)          (dendrogram)              (condensed tree)
        //
        // Node ID convention throughout: points are 0..n-1, internal/cluster nodes are n+.
        // This matches scipy's linkage matrix format and sklearn's condensed tree layout.
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

        /// <summary>
        /// Condensed tree edge: SourceCluster -> Target at a given density level (Lambda).
        /// Target is either a child cluster (>= n) or a point (&lt; n) that "fell out".
        /// Size is the number of points: 1 for point edges, child cluster size for splits.
        /// </summary>
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
                // 2*n slots: n for leaf points + up to n-1 for internal merge nodes.
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
                    parent[x] = Find(parent[x]); // Path compression.
                }
                return parent[x];
            }

            /// <summary>
            /// Merges two components under a new label (the next internal node ID).
            /// Both old roots become children of newLabel, preserving dendrogram structure.
            /// </summary>
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
