using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;

namespace HdbScan.Net
{
    /// <summary>
    /// T-independent HDBSCAN pipeline steps. All methods operate on pairwise distances,
    /// integer indices, and double distances — no knowledge of the point type T.
    /// </summary>
    internal static class HdbScanAlgorithm
    {
        /// <summary>
        /// Section 5.1: density level λ = 1/ε for a distance ε. Distance 0 (duplicate points)
        /// is the highest possible density, λ = ∞ — the same convention as sklearn.
        /// </summary>
        internal static double Lambda(double distance)
        {
            return distance > 0 ? 1.0 / distance : double.PositiveInfinity;
        }

        /// <summary>
        /// Definition 3.1: dcore(xp) = distance from xp to its mpts-nearest neighbor (incl. xp).
        /// Uses a bounded max-heap (PriorityQueue with reversed comparison, size k) to find
        /// the k-th smallest distance in O(n log k) per point instead of O(n log n) via full sort.
        /// Matches sklearn: tree.query(X, k=min_samples)[0][:, -1].
        /// </summary>
        internal static double[] ComputeCoreDistances(Distances distances, int k, int parallelism = 1)
        {
            var n = distances.Count;
            var coreDistances = new double[n];
            k = Math.Min(k, n);

            // Max-heap of size k: root is always the largest among the k smallest distances seen so far.
            // The points are independent of each other, so they can be handled in parallel;
            // every thread then works with a heap and a row buffer of its own.
            var maxFirst = Comparer<double>.Create((a, b) => b.CompareTo(a));
            For(n, parallelism, () => (Heap: new PriorityQueue<byte, double>(k, maxFirst), Buffer: new double[n]), (i, own) =>
            {
                var heap = own.Heap;
                heap.Clear();
                var kthSmallest = 0.0;
                var row = distances.Row(i, own.Buffer);
                for (var j = 0; j < n; j++)
                {
                    var d = i == j ? 0.0 : row[j];
                    if (heap.Count < k)
                    {
                        heap.Enqueue(0, d);
                    }
                    else if (d < kthSmallest)
                    {
                        heap.EnqueueDequeue(0, d);
                    }
                    else
                    {
                        // Not among the k smallest: the common case, kept out of the heap for speed.
                        continue;
                    }
                    heap.TryPeek(out _, out kthSmallest);
                }
                coreDistances[i] = kthSmallest;
            });

            return coreDistances;
        }

        /// <summary>
        /// Definition 3.2: d_mreach(a, b) = max{dcore(a), dcore(b), d(a, b)}.
        /// Effectively "pushes apart" points in sparse regions while preserving distances
        /// in dense regions.
        /// </summary>
        internal static double MutualReachabilityDistance(double[] coreDistances, int i, int j, double dist)
        {
            return Math.Max(Math.Max(coreDistances[i], coreDistances[j]), dist);
        }

        /// <summary>
        /// Algorithm 1 Step 2 / Proposition 3.4: the MST of the mutual reachability
        /// graph Gmpts encodes the density-based clustering hierarchy. Single-linkage on
        /// mutual reachability distances produces all DBSCAN* partitions hierarchically.
        /// Built via Prim's algorithm in O(n^2). Each step needs one row of distances: from the
        /// point just added to the points that are not in the tree yet.
        /// All distances must be finite (checked where they are computed), so every
        /// step finds a next point and the result is always a spanning tree of n - 1 edges.
        /// </summary>
        internal static MstEdge[] BuildMst(Distances distances, double[] coreDistances, int parallelism = 1)
        {
            var n = distances.Count;
            var buffer = new double[n];
            var inMst = new bool[n];
            var minDist = new double[n];
            var minEdge = new int[n];
            var edges = new MstEdge[n - 1];

            // minDist[j] = distance from point j to the nearest point in the tree, minEdge[j] = that point.
            Array.Fill(minDist, double.PositiveInfinity);
            Array.Fill(minEdge, -1);

            // The tree starts as point 0 and grows by one point per step.
            var current = 0;
            inMst[0] = true;

            for (var step = 0; step < n - 1; step++)
            {
                // One pass over the points outside the tree: the point just added may be nearer
                // to them than the rest of the tree is; then pick the one nearest to the tree.
                var row = distances.Row(current, buffer, inMst, parallelism);
                var minVal = double.PositiveInfinity;
                var minIdx = -1;
                for (var j = 0; j < n; j++)
                {
                    if (inMst[j])
                    {
                        continue;
                    }

                    var d = MutualReachabilityDistance(coreDistances, current, j, row[j]);
                    if (d < minDist[j])
                    {
                        minDist[j] = d;
                        minEdge[j] = current;
                    }

                    if (minDist[j] < minVal)
                    {
                        minVal = minDist[j];
                        minIdx = j;
                    }
                }

                inMst[minIdx] = true;
                edges[step] = new MstEdge(minEdge[minIdx], minIdx, minVal);
                current = minIdx;
            }

            return edges;
        }

        /// <summary>
        /// Converts the sorted MST into a single-linkage dendrogram using Union-Find.
        /// Produces the same structure as scipy.cluster.hierarchy.linkage: each row is
        /// (left, right, distance, size). Points are 0..n-1; internal nodes are n, n+1, ...
        /// </summary>
        internal static SingleLinkageNode[] BuildSingleLinkageTree(MstEdge[] sortedMst, int n)
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
        /// Section 3.3 / Algorithm 2: walks the dendrogram top-down, pruning splits
        /// where a child has fewer than minClusterSize points. Uses lambda = 1/distance as
        /// the density level.
        ///
        /// At each split:
        ///   - Both children large enough  -> genuine split, two new child clusters
        ///   - One child too small          -> small side's points "fall out"; large side inherits
        ///   - Both too small               -> all points fall out of the parent cluster
        ///
        /// The output is a list of directed edges (parent -> child_cluster or parent -> point).
        /// </summary>
        internal static List<CondensedTreeEdge> CondenseTree(SingleLinkageNode[] singleLinkageTree, int minClusterSize)
        {
            var n = singleLinkageTree.Length + 1;
            var condensed = new List<CondensedTreeEdge>();

            // Map each dendrogram node to (left, right, lambda, size).
            // Lambda = 1/distance converts distance to density level (Section 5.1: λ = 1/ε).
            var nodeInfo = new Dictionary<int, (int Left, int Right, double Lambda, int Size)>();
            var nodeLabel = n;
            foreach (var node in singleLinkageTree)
            {
                nodeInfo[nodeLabel] = (node.Left, node.Right, Lambda(node.Distance), node.Size);
                nodeLabel++;
            }

            // The last merge in the dendrogram joins the final two components into a
            // single tree, so its label (n + (n-1) - 1 = 2n-2) is always the root.
            // nodeLabel was incremented once per merge, so nodeLabel - 1 points to it.
            var root = nodeLabel - 1;
            var relabel = new Dictionary<int, int> { [root] = n };
            var nextCondensedLabel = n + 1;

            // Records all leaf points under a too-small subtree as departing at the split's lambda.
            void FallOutPoints(int startNode, double lambda, int parentCluster)
            {
                var pending = new Stack<int>();
                pending.Push(startNode);
                while (pending.Count > 0)
                {
                    var current = pending.Pop();
                    // Leaf point (index < n) → emit as a fallen-out point edge.
                    // Internal node (index >= n) → descend into its children.
                    if (current < n)
                    {
                        condensed.Add(new CondensedTreeEdge(parentCluster, current, lambda, 1));
                    }
                    else
                    {
                        var (left, right, _, _) = nodeInfo[current];
                        pending.Push(right);
                        pending.Push(left);
                    }
                }
            }

            // Parent tracks the current cluster identity: (child, child) = new cluster,
            // (child, parent) = continuation of the parent cluster (no genuine split).
            // The walk is breadth-first, like sklearn's, so clusters get the same ids —
            // and in the end the same labels — as in sklearn.
            var queue = new Queue<(int Node, int Parent)>();
            queue.Enqueue((root, root));
            while (queue.Count > 0)
            {
                var (node, parent) = queue.Dequeue();

                // Only internal nodes (index >= n) are ever queued: a node is queued when it
                // has at least minClusterSize >= 2 points. Inspect its children to decide
                // whether this is a genuine split or a continuation.
                var (left, right, lambdaVal, size) = nodeInfo[node];
                // A leaf child (< n) has size 1; an internal child's size is stored in nodeInfo.
                var leftSize = left < n ? 1 : nodeInfo[left].Size;
                var rightSize = right < n ? 1 : nodeInfo[right].Size;

                if (leftSize >= minClusterSize && rightSize >= minClusterSize)
                {
                    // Genuine split: both children become new clusters.
                    relabel[left] = nextCondensedLabel++;
                    relabel[right] = nextCondensedLabel++;
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[left], lambdaVal, leftSize));
                    condensed.Add(new CondensedTreeEdge(relabel[parent], relabel[right], lambdaVal, rightSize));
                    queue.Enqueue((left, left));
                    queue.Enqueue((right, right));
                }
                else if (leftSize >= minClusterSize)
                {
                    // Left survives; right's points fall out of the current cluster.
                    relabel[left] = relabel[parent];
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                    queue.Enqueue((left, parent));
                }
                else if (rightSize >= minClusterSize)
                {
                    // Right survives; left's points fall out of the current cluster.
                    relabel[right] = relabel[parent];
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    queue.Enqueue((right, parent));
                }
                else
                {
                    // Neither child is large enough — both fall out.
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                }
            }

            return condensed;
        }

        /// <summary>
        /// Indexes condensed tree edges by source cluster and by target for O(1) lookups.
        /// </summary>
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

        /// <summary>
        /// Section 5.2: extracts the flat clustering from the condensed tree. Selects the
        /// clusters (EOM or Leaf), numbers them 0, 1, 2, ... in order of their condensed tree id
        /// (as sklearn does), and labels every point with <see cref="ClusterSelection.Assign"/>.
        /// </summary>
        internal static (int[] Labels, double[] Probabilities, ClusterSelection Selection) ExtractClusters(
            List<CondensedTreeEdge> condensedTree, int n, ClusterSelectionMethod method, bool allowSingleCluster)
        {
            var labels = new int[n];
            var probabilities = new double[n];
            Array.Fill(labels, -1);

            var (edgesBySource, edgeByTarget) = BuildCondensedTreeIndex(condensedTree);

            // All internal nodes that appear as parents = the set of clusters in the condensed tree.
            var clusters = edgesBySource.Keys.ToHashSet();

            HashSet<int> selectedClusters;
            if (clusters.Count == 1 && !allowSingleCluster)
            {
                // The root is the only cluster and may not be selected: everything is noise.
                selectedClusters = new HashSet<int>();
            }
            else if (method == ClusterSelectionMethod.Leaf)
            {
                // Leaf method: select clusters that have no child clusters (only point edges).
                // If the root has no child clusters it is the single cluster. (sklearn selects
                // the root in that case too, but then overwrites the selection and returns
                // only noise.)
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

            // Max lambda across all edges (point + child cluster splits) for each selected
            // cluster, used to normalize membership probabilities. This matches sklearn's
            // max_lambdas() in _tree.pyx which considers all edges, not just point edges.
            var clusterMaxLambda = new Dictionary<int, double>();
            foreach (var cluster in selectedClusters)
            {
                var maxLambda = 0.0;
                foreach (var edge in edgesBySource[cluster])
                {
                    if (edge.Lambda > maxLambda)
                    {
                        maxLambda = edge.Lambda;
                    }
                }
                clusterMaxLambda[cluster] = maxLambda;
            }

            // Map internal cluster IDs to 0-based output labels, in order of cluster ID.
            var clusterToLabel = selectedClusters.OrderBy(c => c).Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

            var selection = new ClusterSelection(edgeByTarget, clusterToLabel, clusterMaxLambda, n);

            // Each point edge says "this point left SourceCluster at density Lambda".
            foreach (var edge in condensedTree.Where(e => e.Target < n))
            {
                (labels[edge.Target], probabilities[edge.Target]) = selection.Assign(edge.SourceCluster, edge.Lambda);
            }

            return (labels, probabilities, selection);
        }

        /// <summary>
        /// The selected clusters, plus the rule that turns "a point left cluster C at density λ"
        /// into a label and a membership probability. The same rule labels the training points
        /// (<see cref="ExtractClusters"/>) and new points (<see cref="Predict"/>).
        /// </summary>
        /// <param name="edgeByTarget">For every point and every non-root cluster: the condensed tree edge from its parent cluster.</param>
        /// <param name="clusterToLabel">Output label of every selected cluster.</param>
        /// <param name="clusterMaxLambda">Highest density at which each selected cluster still exists.</param>
        /// <param name="root">Id of the root cluster (= n, the number of points).</param>
        internal sealed class ClusterSelection(
            Dictionary<int, CondensedTreeEdge> edgeByTarget,
            Dictionary<int, int> clusterToLabel,
            Dictionary<int, double> clusterMaxLambda,
            int root)
        {
            public int ClusterCount => clusterToLabel.Count;

            /// <summary>
            /// Label and membership probability of a point that left <paramref name="cluster"/>
            /// at density <paramref name="lambda"/>. Matches sklearn's _do_labelling and
            /// get_probabilities in _tree.pyx.
            /// </summary>
            public (int Label, double Probability) Assign(int cluster, double lambda)
            {
                // The point belongs to the nearest selected cluster on the way up to the root
                // (the cluster itself or an ancestor). If there is none, it is noise.
                while (!clusterToLabel.ContainsKey(cluster))
                {
                    if (!edgeByTarget.TryGetValue(cluster, out var parentEdge))
                    {
                        return (-1, 0.0);
                    }
                    cluster = parentEdge.SourceCluster;
                }

                var maxLambda = clusterMaxLambda[cluster];

                // The root contains every point, so when it is the selected cluster (only
                // possible with allowSingleCluster, and then it is the only one) membership
                // needs a cut-off: sklearn keeps the points that stay until the root's highest
                // density and treats the points that left earlier as noise.
                if (cluster == root && lambda < maxLambda)
                {
                    return (-1, 0.0);
                }

                // Probability = lambda_point / max_lambda_cluster. A point that left a
                // sub-cluster has lambda > max_lambda and gets 1, as does a duplicate (λ = ∞).
                var probability = double.IsPositiveInfinity(lambda) ? 1.0 : Math.Min(lambda, maxLambda) / maxLambda;
                return (clusterToLabel[cluster], probability);
            }

            /// <summary>
            /// Predicts the cluster of a new point x from its distances to the training points,
            /// without changing the clustering: finds where x would attach to the condensed tree
            /// and labels it like a training point in that position. Follows approximate_predict
            /// of the scikit-learn-contrib/hdbscan library (sklearn itself has no prediction).
            /// </summary>
            /// <param name="distances">Distance from x to each training point.</param>
            /// <param name="coreDistances">Core distance of each training point.</param>
            /// <param name="k">The k (min samples) used for the core distances.</param>
            public (int Label, double Probability) Predict(double[] distances, double[] coreDistances, int k)
            {
                var n = distances.Length;
                k = Math.Min(k, n);

                // Definition 3.1, measured against the training data: the core distance of x is
                // the radius around x that holds k training points. For a training point this is
                // exactly its own core distance (it counts itself, at distance 0).
                var sorted = (double[])distances.Clone();
                Array.Sort(sorted);
                var coreDistance = sorted[k - 1];

                // The training point nearest to x in mutual reachability distance (Definition 3.2)
                // is where x attaches to the hierarchy, at density lambda. Ties go to the point
                // that is nearest in plain distance.
                var neighbor = -1;
                var neighborDistance = double.PositiveInfinity;
                for (var i = 0; i < n; i++)
                {
                    var d = Math.Max(Math.Max(coreDistance, coreDistances[i]), distances[i]);
                    if (d < neighborDistance || d == neighborDistance && distances[i] < distances[neighbor])
                    {
                        neighborDistance = d;
                        neighbor = i;
                    }
                }
                var lambda = Lambda(neighborDistance);

                var neighborEdge = edgeByTarget[neighbor];
                var cluster = neighborEdge.SourceCluster;
                if (neighborEdge.Lambda <= lambda)
                {
                    // x holds on to its neighbor at least as long as the neighbor holds on to
                    // its cluster, so x leaves that cluster together with the neighbor.
                    lambda = neighborEdge.Lambda;
                }
                else
                {
                    // x lets go of its neighbor earlier, at density lambda. It can only be a
                    // member of clusters that already existed then: climb past every cluster
                    // that was born (split off from its parent) at density lambda or higher.
                    while (edgeByTarget.TryGetValue(cluster, out var birthEdge) && birthEdge.Lambda >= lambda)
                    {
                        cluster = birthEdge.SourceCluster;
                    }
                }

                return Assign(cluster, lambda);
            }
        }

        /// <summary>
        /// Section 5.2 / Algorithm 3: selects a flat clustering by maximizing
        /// total cluster stability subject to the constraint that selected clusters are
        /// non-overlapping (at most one per root-to-leaf path in the condensed tree).
        ///
        /// Stability of cluster C (Equation 3):
        ///   S(C) = sum_{xj in C} (lambda_max(xj, C) - lambda_min(C))
        ///
        /// Bottom-up traversal (Equation 5): if a parent's own stability exceeds the sum
        /// of its children's propagated stabilities, select the parent and deselect all
        /// descendants. Otherwise, propagate the children's combined stability upward.
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
                        // Equation (3): S(Ci) = Σ (λ_max(xj,Ci) - λ_min(Ci)).
                        // edge.Size groups points that share the same departure lambda,
                        // so × size is equivalent to summing over individual points.
                        // Edges at the birth lambda itself add nothing; skipping them also
                        // avoids ∞ - ∞ for a cluster born at distance 0 (duplicate points).
                        if (edge.Lambda > birthLambda)
                        {
                            stab += (edge.Lambda - birthLambda) * edge.Size;
                        }
                    }
                }

                stability[cluster] = Math.Max(0, stab);
            }

            // Build parent → child-cluster mapping (ignoring point edges).
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

            // Deselects a cluster and everything below it. Iterative: the condensed tree
            // can be deep enough to overflow the call stack with recursion.
            void RemoveDescendants(int cluster)
            {
                var pending = new Stack<int>();
                pending.Push(cluster);
                while (pending.Count > 0)
                {
                    var current = pending.Pop();
                    selected.Remove(current);
                    foreach (var child in childClusters[current])
                    {
                        pending.Push(child);
                    }
                }
            }

            // Process non-leaf clusters bottom-up (Equation 5).
            // When allowSingleCluster is false, exclude the root (id = n) from the
            // comparison so it cannot deselect all descendants. This matches sklearn:
            // node_list = sorted(stability.keys(), reverse=True)[:-1]  # exclude root
            var queue = new Queue<int>(clusters.Where(c => !processed.Contains(c)));
            while (queue.Count > 0)
            {
                var cluster = queue.Dequeue();
                var children = childClusters[cluster];

                // Bottom-up order is not enforced by sorting; instead, a cluster is
                // re-enqueued (below) if its children haven't been processed yet.
                // This guarantees children are always handled before their parent.
                if (children.All(processed.Contains))
                {
                    if (cluster == n && !allowSingleCluster)
                    {
                        // Root excluded from comparison: just propagate children upward.
                        processed.Add(cluster);
                    }
                    else
                    {
                        var childStability = children.Sum(c => stability[c]);
                        if (childStability > stability[cluster])
                        {
                            // Children win: propagate their combined stability upward.
                            stability[cluster] = childStability;
                        }
                        else
                        {
                            // Parent wins (or tie): select it, deselect all descendants.
                            foreach (var child in children)
                            {
                                RemoveDescendants(child);
                            }
                            selected.Add(cluster);
                        }
                        processed.Add(cluster);
                    }
                }
                else
                {
                    queue.Enqueue(cluster);
                }
            }

            // Remove the root cluster from the selection — it is either excluded
            // (allowSingleCluster=false) or kept only if it is the sole selection.
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
        /// Campello et al. (2015), "Hierarchical Density Estimates for Data Clustering,
        /// Visualization, and Outlier Detection." ACM Trans. Knowl. Discov. Data.
        ///
        /// For each point: score = 1 - lambda_p / lambda_max(cluster).
        /// lambda_max is propagated upward so each cluster's "death" reflects the peak
        /// density anywhere in its subtree. This makes the score local — a point in a
        /// sparse cluster is compared against that cluster's peak, not a distant dense one.
        ///
        /// Score near 0 = deep inside its cluster. Score near 1 = strong outlier.
        /// Points absent from the condensed tree default to score 1.0.
        /// </summary>
        internal static double[] ComputeOutlierScores(List<CondensedTreeEdge> condensedTree, int n)
        {
            var scores = new double[n];
            Array.Fill(scores, 1.0);

            if (condensedTree.Count == 0) return scores;

            // deaths[cluster] = max lambda seen among its child edges (the cluster's "death" density).
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

            // Algorithm 4 Step 1: propagate ε_max from children to parents.
            // Children have higher IDs than parents, so reverse-sorted order
            // guarantees correct propagation.
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
                    // Equation (8): GLOSH(xp) = 1 - ε_max(xp) / ε(xp) = 1 - λ(xp) / λ_max(Ci).
                    // A point with λ = ∞ sits exactly on its cluster's densest spot (duplicate
                    // points): score 0, the same convention as the hdbscan library.
                    scores[i] = double.IsPositiveInfinity(edge.Lambda)
                        ? 0.0
                        : Math.Max(0, Math.Min(1.0, 1.0 - edge.Lambda / deaths[edge.SourceCluster]));
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
        // Node ID convention (used throughout the entire pipeline):
        //
        //   index < n   →  leaf point     (one of the n original data points)
        //   index >= n  →  internal node  (a merge/cluster node in the dendrogram)
        //
        // The dendrogram's i-th merge creates internal node (n + i). For example,
        // with 5 data points (n=5): points are 0-4, and the first merge is node 5,
        // the second is node 6, etc. The root is always (2n - 2).
        //
        // The "node < n" guard appears wherever the code must distinguish between
        // descending further into the tree (internal node) vs. acting on an actual
        // data point (leaf). This convention matches scipy's linkage matrix format
        // and sklearn's condensed tree layout.
        // ================================================================================

        /// <summary>
        /// The pairwise distances between the n points, handed out one row at a time.
        /// Core distances and the MST only ever need whole rows, so the distances can either
        /// be stored (<see cref="DistanceMatrix"/>) or be computed at the moment a row is asked for.
        /// </summary>
        internal abstract class Distances(int count)
        {
            /// <summary>The number of points n.</summary>
            public int Count { get; } = count;

            /// <summary>
            /// The distances from point <paramref name="i"/> to all n points.
            /// </summary>
            /// <param name="i">The point.</param>
            /// <param name="buffer">Scratch space of length n that may be used to hold the result.</param>
            /// <param name="skip">Entries j with skip[j] = true are not needed and may hold any value.</param>
            /// <param name="parallelism">Number of threads that may be used to compute the row.</param>
            public abstract ReadOnlySpan<double> Row(int i, double[] buffer, bool[]? skip = null, int parallelism = 1);
        }

        /// <summary>
        /// All n×n distances stored as a flat array in row-major order.
        /// </summary>
        internal sealed class DistanceMatrix(double[] matrix, int n) : Distances(n)
        {
            public override ReadOnlySpan<double> Row(int i, double[] buffer, bool[]? skip = null, int parallelism = 1)
            {
                return matrix.AsSpan(i * Count, Count);
            }
        }

        /// <summary>
        /// Runs body(0) ... body(count - 1): in order when parallelism is 1, otherwise in
        /// parallel on at most that many threads (-1 = no limit).
        /// </summary>
        internal static void For(int count, int parallelism, Action<int> body)
        {
            For<object?>(count, parallelism, () => null, (i, _) => body(i));
        }

        /// <summary>
        /// The same, for a body that needs scratch space: <paramref name="newScratch"/> makes one
        /// per thread, and it is passed to every body call on that thread.
        /// </summary>
        internal static void For<TScratch>(int count, int parallelism, Func<TScratch> newScratch, Action<int, TScratch> body)
        {
            if (parallelism == 1)
            {
                var scratch = newScratch();
                for (var i = 0; i < count; i++)
                {
                    body(i, scratch);
                }

                return;
            }

            try
            {
                Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, newScratch,
                    (i, _, scratch) => { body(i, scratch); return scratch; }, _ => { });
            }
            catch (AggregateException e)
            {
                // Parallel.For wraps exceptions; rethrow the original one (for example the
                // ArgumentException for an invalid distance), as in the non-parallel case.
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
        }

        internal readonly struct MstEdge(int a, int b, double distance)
        {
            public readonly int A = a;
            public readonly int B = b;
            public readonly double Distance = distance;
        }

        internal readonly struct SingleLinkageNode(int left, int right, double distance, int size)
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
        internal readonly struct CondensedTreeEdge(int sourceCluster, int target, double lambda, int size)
        {
            public readonly int SourceCluster = sourceCluster;
            public readonly int Target = target;
            public readonly double Lambda = lambda;
            public readonly int Size = size;
        }

        internal sealed class UnionFind
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
                var root = x;
                while (parent[root] != root)
                    root = parent[root];
                while (parent[x] != root)
                {
                    var next = parent[x];
                    parent[x] = root;
                    x = next;
                }
                return root;
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
