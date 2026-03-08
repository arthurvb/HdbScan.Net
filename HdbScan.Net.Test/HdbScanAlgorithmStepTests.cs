using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using HdbScan.Net;

namespace HdbScan.Net.Test
{
    /// <summary>
    /// Step-by-step tests for each stage of the HDBSCAN pipeline.
    /// Each test uses a small hand-traceable example with comments explaining
    /// the expected values.
    /// </summary>
    public class HdbScanAlgorithmStepTests
    {
        // -----------------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------------

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
        /// Computes the distance matrix for the given points. This is the only
        /// T-dependent step (uses HdbScan&lt;double[]&gt; indirectly via inline logic).
        /// </summary>
        private static double[] ComputeDistMatrix(double[][] points)
        {
            var n = points.Length;
            var matrix = new double[n * n];

            for (var i = 0; i < n; i++)
            {
                for (var j = i + 1; j < n; j++)
                {
                    var d = EuclideanDistance(points[i], points[j]);
                    matrix[i * n + j] = d;
                    matrix[j * n + i] = d;
                }
            }

            return matrix;
        }

        /// <summary>
        /// Runs the full MST -> dendrogram pipeline on given points so we can
        /// verify UnionFind behavior through its observable output.
        /// </summary>
        private static HdbScanAlgorithm.SingleLinkageNode[] BuildDendrogram(double[][] pts, int k)
        {
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, pts.Length, k);
            var mst = HdbScanAlgorithm.BuildMst(dm, pts.Length, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            return HdbScanAlgorithm.BuildSingleLinkageTree(mst, pts.Length);
        }

        // -----------------------------------------------------------------------
        // Tests
        // -----------------------------------------------------------------------

        /// <summary>
        /// Test 1: UnionFind data structure (tested through BuildSingleLinkageTree).
        ///
        /// 5 points on a 1D line: P0=(0), P1=(1), P2=(3), P3=(6), P4=(10)
        /// With k=3, core distances = [3, 2, 3, 4, 7].
        ///
        /// MRD weights (selected): mrd(0,1)=3, mrd(0,2)=3, mrd(2,3)=4, mrd(2,4)=7
        /// MST (Prim's): (0,1,3), (0,2,3), (2,3,4), (2,4,7)
        ///
        /// Dendrogram (4 merge nodes, internal IDs 5..8):
        ///   Edge (0,1,3) -> node 5: P0+P1, size=2
        ///   Edge (0,2,3) -> node 6: node5+P2, size=3
        ///   Edge (2,3,4) -> node 7: node6+P3, size=4
        ///   Edge (2,4,7) -> node 8: node7+P4, size=5
        ///
        /// The last merge exercises deep Find chains: Find(2) must traverse
        /// 2->6->7 (two hops through earlier union roots) to locate the current
        /// component root. GetLabel must then return 7. With 4 sequential unions,
        /// parent chains reach depth 2+, exercising path compression. Any bug in
        /// Find, Union, GetSize, or GetLabel would corrupt the values below.
        /// </summary>
        [Test]
        public void UnionFind_MergesAndPathCompression()
        {
            // 5 points force 4 merge nodes, creating deeper Union chains than test 5
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 }, new[] { 10.0 } };
            var slt = BuildDendrogram(pts, 3);

            Assert.That(slt.Length, Is.EqualTo(4), "5 points -> 4 dendrogram nodes");

            // Node 0 (internal 5): P0 + P1 at distance 3, size 2
            Assert.That(new[] { slt[0].Left, slt[0].Right }.OrderBy(x => x), Is.EqualTo(new[] { 0, 1 }));
            Assert.That(slt[0].Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(slt[0].Size, Is.EqualTo(2));

            // Node 1 (internal 6): {P0,P1}=node5 + P2 at distance 3, size 3
            Assert.That(new[] { slt[1].Left, slt[1].Right }.OrderBy(x => x), Is.EqualTo(new[] { 2, 5 }));
            Assert.That(slt[1].Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(slt[1].Size, Is.EqualTo(3));

            // Node 2 (internal 7): {P0,P1,P2}=node6 + P3 at distance 4, size 4
            Assert.That(new[] { slt[2].Left, slt[2].Right }.OrderBy(x => x), Is.EqualTo(new[] { 3, 6 }));
            Assert.That(slt[2].Distance, Is.EqualTo(4.0).Within(1e-10));
            Assert.That(slt[2].Size, Is.EqualTo(4));

            // Node 3 (internal 8): {P0..P3}=node7 + P4 at distance 7, size 5
            // Key UnionFind test: Find(2) traverses the parent chain 2->6->7
            // (two hops through earlier union roots) to reach the current root,
            // then GetLabel returns 7.
            Assert.That(new[] { slt[3].Left, slt[3].Right }.OrderBy(x => x), Is.EqualTo(new[] { 4, 7 }));
            Assert.That(slt[3].Distance, Is.EqualTo(7.0).Within(1e-10));
            Assert.That(slt[3].Size, Is.EqualTo(5));

            // Distances must be non-decreasing (structural invariant)
            for (var i = 1; i < slt.Length; i++)
            {
                Assert.That(slt[i].Distance, Is.GreaterThanOrEqualTo(slt[i - 1].Distance),
                    $"Dendrogram distances must be non-decreasing (node {i - 1} -> {i})");
            }
        }

        /// <summary>
        /// Test 2: Core distances (Definition 3.1).
        ///
        /// Points on a 1D line: P0=(0), P1=(1), P2=(3), P3=(6)
        /// With k=3, core distance = 3rd smallest distance (including self=0).
        ///
        ///   P0: sorted dists [0, 1, 3, 6] -> 3rd = 3
        ///   P1: sorted dists [0, 1, 2, 5] -> 3rd = 2   (central, smallest core)
        ///   P2: sorted dists [0, 2, 3, 3] -> 3rd = 3
        ///   P3: sorted dists [0, 3, 5, 6] -> 3rd = 5   (isolated, largest core)
        /// </summary>
        [Test]
        public void CoreDistances_KthNearestNeighbor()
        {
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 } };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 4, 3);

            Assert.That(core[0], Is.EqualTo(3.0).Within(1e-10), "P0: 3rd neighbor at distance 3");
            Assert.That(core[1], Is.EqualTo(2.0).Within(1e-10), "P1: 3rd neighbor at distance 2");
            Assert.That(core[2], Is.EqualTo(3.0).Within(1e-10), "P2: 3rd neighbor at distance 3");
            Assert.That(core[3], Is.EqualTo(5.0).Within(1e-10), "P3: 3rd neighbor at distance 5");
        }

        /// <summary>
        /// Test 3: Mutual reachability distance (Definition 3.2).
        ///
        /// mrd(a,b) = max(core_a, core_b, d(a,b))
        ///
        /// Same 4-point line, core distances [3, 2, 3, 5]:
        ///   mrd(0,1) = max(3, 2, 1) = 3   <- core_a dominates
        ///   mrd(1,2) = max(2, 3, 2) = 3   <- core_b dominates
        ///   mrd(0,3) = max(3, 5, 6) = 6   <- raw distance dominates
        /// </summary>
        [Test]
        public void MutualReachabilityDistance_MaxOfThree()
        {
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 } };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 4, 3);

            Assert.That(HdbScanAlgorithm.MutualReachabilityDistance(dm, 4, core, 0, 1), Is.EqualTo(3.0).Within(1e-10), "core_a wins");
            Assert.That(HdbScanAlgorithm.MutualReachabilityDistance(dm, 4, core, 1, 2), Is.EqualTo(3.0).Within(1e-10), "core_b wins");
            Assert.That(HdbScanAlgorithm.MutualReachabilityDistance(dm, 4, core, 0, 3), Is.EqualTo(6.0).Within(1e-10), "d(a,b) wins");
        }

        /// <summary>
        /// Test 4: Minimum spanning tree (Proposition 3.4, Prim's algorithm).
        ///
        /// 4-point line, k=3. MRD weights:
        ///   mrd(0,1)=3, mrd(0,2)=3, mrd(0,3)=6
        ///   mrd(1,2)=3, mrd(1,3)=5, mrd(2,3)=5
        ///
        /// MST has 3 edges with total weight 3 + 3 + 5 = 11.
        /// The heaviest edge (weight 5) connects to P3 (isolated point).
        /// </summary>
        [Test]
        public void BuildMst_MinimumSpanningTree()
        {
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 } };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 4, 3);

            var mst = HdbScanAlgorithm.BuildMst(dm, 4, core);
            Assert.That(mst.Length, Is.EqualTo(3), "MST of 4 points has 3 edges");

            var totalWeight = mst.Sum(e => e.Distance);
            Assert.That(totalWeight, Is.EqualTo(11.0).Within(1e-10), "Total weight = 3 + 3 + 5");

            // Two edges at weight 3, one at weight 5
            var sorted = mst.OrderBy(e => e.Distance).ToArray();
            Assert.That(sorted[0].Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(sorted[1].Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(sorted[2].Distance, Is.EqualTo(5.0).Within(1e-10));

            // Heaviest edge connects to P3
            Assert.That(sorted[2].A == 3 || sorted[2].B == 3, Is.True,
                "Heaviest edge connects to isolated point P3");
        }

        /// <summary>
        /// Test 5: Single-linkage dendrogram (Algorithm 1).
        ///
        /// Sorted MST: (0,1,3), (0,2,3), (?,3,5)
        /// Internal nodes start at n=4.
        ///
        ///   Edge (0,1,3) -> node 4: merge P0+P1, size=2, dist=3
        ///   Edge (0,2,3) -> node 5: merge node4+P2, size=3, dist=3
        ///   Edge (?,3,5) -> node 6: merge node5+P3, size=4, dist=5
        /// </summary>
        [Test]
        public void SingleLinkageTree_Dendrogram()
        {
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 } };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 4, 3);
            var mst = HdbScanAlgorithm.BuildMst(dm, 4, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var slt = HdbScanAlgorithm.BuildSingleLinkageTree(mst, 4);

            Assert.That(slt.Length, Is.EqualTo(3), "4 points -> 3 dendrogram nodes");

            var n0 = slt[0];
            var n1 = slt[1];
            var n2 = slt[2];

            // Node 0: P0 + P1 at distance 3, size 2
            Assert.That(new[] { n0.Left, n0.Right }.OrderBy(x => x), Is.EqualTo(new[] { 0, 1 }));
            Assert.That(n0.Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(n0.Size, Is.EqualTo(2));

            // Node 1: {P0,P1}=node4 + P2 at distance 3, size 3
            Assert.That(new[] { n1.Left, n1.Right }.OrderBy(x => x), Is.EqualTo(new[] { 2, 4 }));
            Assert.That(n1.Distance, Is.EqualTo(3.0).Within(1e-10));
            Assert.That(n1.Size, Is.EqualTo(3));

            // Node 2: {P0,P1,P2}=node5 + P3 at distance 5, size 4
            Assert.That(new[] { n2.Left, n2.Right }.OrderBy(x => x), Is.EqualTo(new[] { 3, 5 }));
            Assert.That(n2.Distance, Is.EqualTo(5.0).Within(1e-10));
            Assert.That(n2.Size, Is.EqualTo(4));
        }

        /// <summary>
        /// Test 6: Condensed tree with no genuine split.
        ///
        /// 4-point line, minClusterSize=2. Every merge adds just 1 point,
        /// so no split produces two children >= 2. Only the root cluster exists.
        ///
        /// Points fall out at decreasing density (increasing distance):
        ///   P3 at lambda=1/(5+eps)  -- most isolated, departs first
        ///   P0, P1, P2 at lambda=1/(3+eps) -- denser region, depart later
        /// </summary>
        [Test]
        public void CondenseTree_PointsFallOut()
        {
            var eps = HdbScanAlgorithm.CondenseEpsilon;
            double[][] pts = { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 }, new[] { 6.0 } };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 4, 3);
            var mst = HdbScanAlgorithm.BuildMst(dm, 4, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var slt = HdbScanAlgorithm.BuildSingleLinkageTree(mst, 4);
            var condensed = HdbScanAlgorithm.CondenseTree(slt, 2);

            // Only root cluster (no genuine split)
            var clusters = condensed.Select(e => e.SourceCluster).Distinct().ToList();
            Assert.That(clusters, Has.Count.EqualTo(1), "Only root cluster exists");

            // All 4 points fall out
            var pts4 = condensed.Where(e => e.Target < 4).ToList();
            Assert.That(pts4.Select(e => e.Target).OrderBy(x => x), Is.EqualTo(new[] { 0, 1, 2, 3 }));

            // P3 falls out at lambda = 1/(5+eps) -- lowest density
            var p3 = pts4.First(e => e.Target == 3);
            Assert.That(p3.Lambda, Is.EqualTo(1.0 / (5.0 + eps)).Within(1e-6),
                "P3 (isolated) departs at lambda=1/(5+eps)");

            // P0, P1, P2 fall out at lambda = 1/(3+eps) -- higher density
            var lam3 = 1.0 / (3.0 + eps);
            foreach (var pt in new[] { 0, 1, 2 })
            {
                var e = pts4.First(x => x.Target == pt);
                Assert.That(e.Lambda, Is.EqualTo(lam3).Within(1e-6), $"P{pt} departs at lambda=1/(3+eps)");
            }
        }

        /// <summary>
        /// Test 7: Condensed tree with a genuine split.
        ///
        /// Two clusters of 3 points each, well separated:
        ///   A: P0=(0,0) P1=(1,0) P2=(0,1)    B: P3=(10,0) P4=(11,0) P5=(10,1)
        ///
        /// With minClusterSize=3, the root splits into two child clusters
        /// (each has 3 points >= minClusterSize). Within each child, subsequent
        /// splits produce children of size &lt; 3, so all points fall out.
        ///
        /// MRD = Mutual Reachability Distance (Definition 3.2):
        ///   MRD(a, b) = max(core(a), core(b), d(a, b))
        ///
        /// The inter-cluster MST edge connects P1=(1,0) to P3=(10,0) with d=9.
        /// Both core distances are smaller (intra-cluster), so the raw distance
        /// dominates: MRD(P1, P3) = max(core(P1), core(P3), 9) = 9.
        /// Child clusters are born at lambda=1/(9+eps). Points within each cluster
        /// fall out at lambda=1/(sqrt(2)+eps) (the intra-cluster MRD).
        /// </summary>
        [Test]
        public void CondenseTree_GenuineSplit()
        {
            double[][] pts =
            {
                new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
                new[] { 10.0, 0.0 }, new[] { 11.0, 0.0 }, new[] { 10.0, 1.0 }
            };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 6, 3);
            var mst = HdbScanAlgorithm.BuildMst(dm, 6, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var slt = HdbScanAlgorithm.BuildSingleLinkageTree(mst, 6);
            var condensed = HdbScanAlgorithm.CondenseTree(slt, 3);

            // Root (id=6) splits into two child clusters (targets >= 6)
            var clusterEdges = condensed.Where(e => e.Target >= 6).ToList();
            Assert.That(clusterEdges, Has.Count.EqualTo(2), "Root splits into 2 children");
            Assert.That(clusterEdges.All(e => e.SourceCluster == 6), Is.True, "Both children come from root");
            Assert.That(clusterEdges[0].Size, Is.EqualTo(3));
            Assert.That(clusterEdges[1].Size, Is.EqualTo(3));

            // Both children born at the same lambda
            Assert.That(clusterEdges[0].Lambda, Is.EqualTo(clusterEdges[1].Lambda).Within(1e-10));

            // Each child has exactly 3 point-edges
            var c1 = clusterEdges[0].Target;
            var c2 = clusterEdges[1].Target;
            var c1pts = condensed.Where(e => e.SourceCluster == c1 && e.Target < 6).ToList();
            var c2pts = condensed.Where(e => e.SourceCluster == c2 && e.Target < 6).ToList();
            Assert.That(c1pts, Has.Count.EqualTo(3));
            Assert.That(c2pts, Has.Count.EqualTo(3));

            // Points fall out at higher lambda than the birth lambda
            var birthLam = clusterEdges[0].Lambda;
            foreach (var e in c1pts.Concat(c2pts))
                Assert.That(e.Lambda, Is.GreaterThan(birthLam), $"Point {e.Target} falls out above birth lambda");
        }

        /// <summary>
        /// Test 8: EOM cluster selection (Algorithm 3).
        ///
        /// Using the 6-point two-cluster condensed tree from test 7.
        ///
        /// Stability S(C) = sum (lambda_point - lambda_birth) for each edge in C.
        ///   Root:   birth=0,       3 points each side at lambda_split -> S approx 6/(9+eps) approx 0.67
        ///   Child1: birth=lambda_split, 3 points at lambda_intra     -> S approx 3*(1/sqrt(2) - 1/9) approx 1.79
        ///   Child2: same by symmetry                                 -> S approx 1.79
        ///
        /// Combined children stability (approx 3.58) > root (approx 0.67) -> EOM selects children.
        /// </summary>
        [Test]
        public void EomStability_ChildrenBeatRoot()
        {
            double[][] pts =
            {
                new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
                new[] { 10.0, 0.0 }, new[] { 11.0, 0.0 }, new[] { 10.0, 1.0 }
            };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 6, 3);
            var mst = HdbScanAlgorithm.BuildMst(dm, 6, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var slt = HdbScanAlgorithm.BuildSingleLinkageTree(mst, 6);
            var condensed = HdbScanAlgorithm.CondenseTree(slt, 3);

            var (labels, _, count) = HdbScanAlgorithm.ExtractClusters(
                condensed, 6, ClusterSelectionMethod.ExcessOfMass, false);

            Assert.That(count, Is.EqualTo(2), "EOM selects 2 clusters");

            // Cluster A: P0, P1, P2 share one label
            Assert.That(labels[0], Is.EqualTo(labels[1]));
            Assert.That(labels[0], Is.EqualTo(labels[2]));

            // Cluster B: P3, P4, P5 share another label
            Assert.That(labels[3], Is.EqualTo(labels[4]));
            Assert.That(labels[3], Is.EqualTo(labels[5]));

            // Different clusters
            Assert.That(labels[0], Is.Not.EqualTo(labels[3]));

            // No noise
            Assert.That(labels.All(l => l >= 0), Is.True, "No noise points");
        }

        /// <summary>
        /// Test 9: GLOSH outlier scores (Algorithm 4).
        ///
        /// 7 points on a 1D line with a straggler:
        ///   Cluster A core:      P0=(0), P1=(1), P2=(2)   (tight, spacing 1)
        ///   Cluster A straggler: P3=(8)                    (gap of 6 from core)
        ///   Cluster B:           P4=(30), P5=(31), P6=(32) (tight, spacing 1)
        ///
        /// With k=3, core distances = [2, 1, 2, 7, 2, 1, 2].
        /// The MST connects A to B through P3-P4 at MRD=22, the heaviest edge.
        ///
        /// Condensed tree (minClusterSize=3):
        ///   Root (7) splits into cluster A (8, size 4) and cluster B (9, size 3)
        ///     at λ_birth = 1/(22+ε).
        ///   Cluster A: P3 falls out at λ=1/(7+ε)   (straggler, low density)
        ///              P0,P1,P2 fall out at λ=1/(2+ε) (core, high density)
        ///   Cluster B: P4,P5,P6 all fall out at λ=1/(2+ε) (uniform density)
        ///
        /// GLOSH: score(p) = 1 - λ_p / λ_max(cluster)
        ///
        ///   λ_max(A) = 1/(2+ε)  (peak density from core points)
        ///   λ_max(B) = 1/(2+ε)  (all points at same density)
        ///
        ///   P3: score = 1 - (1/(7+ε)) / (1/(2+ε)) = 1 - (2+ε)/(7+ε) = 5/(7+ε) ≈ 5/7
        ///       Straggler departs far below peak density -> strong outlier signal.
        ///   P0,P1,P2: score = 1 - 1 = 0   (at peak density of cluster A)
        ///   P4,P5,P6: score = 1 - 1 = 0   (at peak density of cluster B)
        ///
        /// This verifies:
        ///   - Non-trivial outlier scores: P3 gets ~0.71 while core points get 0
        ///   - Per-cluster normalization: each cluster uses its own λ_max
        ///   - λ_max propagation: root's death inherits max from its children
        /// </summary>
        [Test]
        public void GloshOutlierScores_StragglerDetection()
        {
            double[][] pts =
            {
                new[] { 0.0 }, new[] { 1.0 }, new[] { 2.0 }, new[] { 8.0 },
                new[] { 30.0 }, new[] { 31.0 }, new[] { 32.0 }
            };
            var dm = ComputeDistMatrix(pts);
            var core = HdbScanAlgorithm.ComputeCoreDistances(dm, 7, 3);
            var mst = HdbScanAlgorithm.BuildMst(dm, 7, core);
            Array.Sort(mst, (a, b) => a.Distance.CompareTo(b.Distance));
            var slt = HdbScanAlgorithm.BuildSingleLinkageTree(mst, 7);
            var condensed = HdbScanAlgorithm.CondenseTree(slt, 3);

            var scores = HdbScanAlgorithm.ComputeOutlierScores(condensed, 7);
            Assert.That(scores.Length, Is.EqualTo(7));

            // All scores in [0, 1]
            foreach (var s in scores)
            {
                Assert.That(s, Is.GreaterThanOrEqualTo(0.0));
                Assert.That(s, Is.LessThanOrEqualTo(1.0));
            }

            // P3 (straggler): score = 5/(7+ε) ≈ 5/7 ≈ 0.714
            var eps = HdbScanAlgorithm.CondenseEpsilon;
            Assert.That(scores[3], Is.EqualTo(5.0 / (7.0 + eps)).Within(1e-6),
                "P3 (straggler) score = 1 - (2+ε)/(7+ε) = 5/(7+ε) ≈ 5/7");

            // Core points of cluster A: at peak density -> score 0
            Assert.That(scores[0], Is.EqualTo(0.0).Within(1e-6), "P0 at peak density of cluster A");
            Assert.That(scores[1], Is.EqualTo(0.0).Within(1e-6), "P1 at peak density of cluster A");
            Assert.That(scores[2], Is.EqualTo(0.0).Within(1e-6), "P2 at peak density of cluster A");

            // Cluster B: all at uniform density -> score 0
            Assert.That(scores[4], Is.EqualTo(0.0).Within(1e-6), "P4 at peak density of cluster B");
            Assert.That(scores[5], Is.EqualTo(0.0).Within(1e-6), "P5 at peak density of cluster B");
            Assert.That(scores[6], Is.EqualTo(0.0).Within(1e-6), "P6 at peak density of cluster B");

            // P3 is clearly the strongest outlier, far above any core point
            var maxCoreScore = new[] { scores[0], scores[1], scores[2], scores[4], scores[5], scores[6] }.Max();
            Assert.That(scores[3], Is.GreaterThan(maxCoreScore + 0.5),
                "Straggler's outlier score should be much higher than any core point's");
        }
    }
}
