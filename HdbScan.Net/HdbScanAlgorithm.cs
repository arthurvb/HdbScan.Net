using System;
using System.Collections.Generic;
using System.Linq;

namespace HdbScan.Net
{
    /// <summary>
    /// T-independent HDBSCAN pipeline steps. All methods operate on distance matrices,
    /// integer indices, and double distances — no knowledge of the point type T.
    /// </summary>
    internal static class HdbScanAlgorithm
    {
        /// <summary>
        /// Small epsilon added to distances before inverting to lambda (= 1/(distance + ε))
        /// to avoid division by zero when distance is exactly 0.
        /// </summary>
        internal const double CondenseEpsilon = 1e-10;
        /// <summary>
        /// Definition 3.1: dcore(xp) = distance from xp to its mpts-nearest neighbor (incl. xp).
        /// Uses a bounded max-heap (PriorityQueue with reversed comparison, size k) to find
        /// the k-th smallest distance in O(n log k) per point instead of O(n log n) via full sort.
        /// Matches sklearn: tree.query(X, k=min_samples)[0][:, -1].
        /// </summary>
        internal static double[] ComputeCoreDistances(double[] distMatrix, int n, int k)
        {
            var coreDistances = new double[n];
            k = Math.Min(k, n);

            // Max-heap of size k: root is always the largest among the k smallest distances seen so far.
            var heap = new PriorityQueue<byte, double>(k, Comparer<double>.Create((a, b) => b.CompareTo(a)));

            for (var i = 0; i < n; i++)
            {
                heap.Clear();
                var rowOffset = i * n;
                for (var j = 0; j < n; j++)
                {
                    var d = i == j ? 0.0 : distMatrix[rowOffset + j];
                    if (heap.Count < k)
                    {
                        heap.Enqueue(0, d);
                    }
                    else
                    {
                        heap.EnqueueDequeue(0, d);
                    }
                }
                heap.TryPeek(out _, out var kthSmallest);
                coreDistances[i] = kthSmallest;
            }

            return coreDistances;
        }

        /// <summary>
        /// Definition 3.2: d_mreach(a, b) = max{dcore(a), dcore(b), d(a, b)}.
        /// Effectively "pushes apart" points in sparse regions while preserving distances
        /// in dense regions.
        /// </summary>
        internal static double MutualReachabilityDistance(double[] distMatrix, int n,
            double[] coreDistances, int i, int j)
        {
            var dist = distMatrix[i * n + j];
            return Math.Max(Math.Max(coreDistances[i], coreDistances[j]), dist);
        }

        /// <summary>
        /// Algorithm 1 Step 2 / Proposition 3.4: the MST of the mutual reachability
        /// graph Gmpts encodes the density-based clustering hierarchy. Single-linkage on
        /// mutual reachability distances produces all DBSCAN* partitions hierarchically.
        /// Built via Prim's algorithm in O(n^2).
        /// </summary>
        internal static MstEdge[] BuildMst(double[] distMatrix, int n, double[] coreDistances)
        {
            var inMst = new bool[n];
            var minDist = new double[n];
            var minEdge = new int[n];
            var edges = new MstEdge[n - 1];

            Array.Fill(minDist, double.MaxValue);
            Array.Fill(minEdge, -1);

            inMst[0] = true;
            for (var j = 1; j < n; j++)
            {
                var d = MutualReachabilityDistance(distMatrix, n, coreDistances, 0, j);
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
                edges[step] = new MstEdge(minEdge[minIdx], minIdx, minVal);

                for (var j = 0; j < n; j++)
                {
                    if (!inMst[j])
                    {
                        var d = MutualReachabilityDistance(distMatrix, n, coreDistances, minIdx, j);
                        if (d < minDist[j])
                        {
                            minDist[j] = d;
                            minEdge[j] = minIdx;
                        }
                    }
                }
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
            // Lambda = 1/(distance + ε) converts distance to density level (Section 5.1: λ = 1/ε).
            var nodeInfo = new Dictionary<int, (int Left, int Right, double Lambda, int Size)>();
            var nodeLabel = n;
            foreach (var node in singleLinkageTree)
            {
                nodeInfo[nodeLabel] = (node.Left, node.Right, 1.0 / (node.Distance + CondenseEpsilon), node.Size);
                nodeLabel++;
            }

            // Relabel clusters in the condensed tree starting from n (the root).
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

            var stack = new Stack<(int Node, int Parent)>();
            stack.Push((root, root));
            while (stack.Count > 0)
            {
                var (node, parent) = stack.Pop();

                if (node < n)
                {
                    var lambda = nodeInfo.ContainsKey(parent) ? nodeInfo[parent].Lambda : 0;
                    condensed.Add(new CondensedTreeEdge(relabel[parent], node, lambda, 1));
                    continue;
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
                    stack.Push((right, right));
                    stack.Push((left, left));
                }
                else if (leftSize >= minClusterSize)
                {
                    // Left survives; right's points fall out of the current cluster.
                    relabel[left] = relabel[parent];
                    FallOutPoints(right, lambdaVal, relabel[parent]);
                    stack.Push((left, parent));
                }
                else if (rightSize >= minClusterSize)
                {
                    relabel[right] = relabel[parent];
                    FallOutPoints(left, lambdaVal, relabel[parent]);
                    stack.Push((right, parent));
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

        internal static (int[] Labels, double[] Probabilities, int ClusterCount) ExtractClusters(
            List<CondensedTreeEdge> condensedTree, int n, ClusterSelectionMethod method, bool allowSingleCluster)
        {
            var labels = new int[n];
            var probabilities = new double[n];
            Array.Fill(labels, -1);

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
                        // Equation (3): S(Ci) = Σ (λ_max(xj,Ci) - λ_min(Ci)) · |points|
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

            // Process non-leaf clusters bottom-up (Equation 5). A cluster is ready
            // once all its children have been processed; re-enqueue if not yet ready.
            // When allowSingleCluster is false, exclude the root (id = n) from the
            // comparison so it cannot deselect all descendants. This matches sklearn:
            // node_list = sorted(stability.keys(), reverse=True)[:-1]  # exclude root
            var queue = new Queue<int>(clusters.Where(c => !processed.Contains(c)));
            while (queue.Count > 0)
            {
                var cluster = queue.Dequeue();
                var children = childClusters[cluster];

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
                    var clusterDeath = deaths.TryGetValue(edge.SourceCluster, out var d) ? d : 1.0;
                    if (clusterDeath > 0)
                    {
                        // Equation (8): GLOSH(xp) = 1 - ε_max(xp) / ε_max(Ci)
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
