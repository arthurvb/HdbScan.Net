using System;

namespace HdbScan.Net
{
    /// <summary>
    /// Specifies the method used to select clusters from the condensed tree.
    /// </summary>
    public enum ClusterSelectionMethod
    {
        /// <summary>
        /// Excess of Mass (EOM) selects clusters based on stability.
        /// This tends to produce larger, more stable clusters.
        /// </summary>
        ExcessOfMass,

        /// <summary>
        /// Leaf selects the leaf nodes of the condensed tree as clusters.
        /// This produces more fine-grained, homogeneous clusters.
        /// </summary>
        Leaf
    }

    /// <summary>
    /// Specifies options for HDBSCAN.
    /// </summary>
    public sealed class HdbScanOptions
    {
        private int minClusterSize;
        private int? minSamples;
        private ClusterSelectionMethod clusterSelectionMethod;
        private bool allowSingleCluster;
        private bool storeDistanceMatrix;
        private int maxDegreeOfParallelism;

        /// <summary>
        /// Creates an instance of <see cref="HdbScanOptions"/> with default parameters.
        /// </summary>
        public HdbScanOptions()
        {
            minClusterSize = 5;
            minSamples = null;
            clusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass;
            allowSingleCluster = false;
            storeDistanceMatrix = true;
            maxDegreeOfParallelism = 1;
        }

        /// <summary>
        /// The minimum number of points required to form a cluster.
        /// </summary>
        public int MinClusterSize
        {
            get => minClusterSize;

            set
            {
                if (value < 2)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The minimum cluster size must be at least 2.");
                }

                minClusterSize = value;
            }
        }

        /// <summary>
        /// The number of samples in a neighborhood for a point to be considered a core point,
        /// including the point itself. Must be at least 1; with 1 every point is a core point
        /// and the mutual reachability distance is the plain distance.
        /// If not set, defaults to <see cref="MinClusterSize"/>.
        /// </summary>
        /// <remarks>
        /// This follows the sklearn.cluster.HDBSCAN convention where min_samples includes the
        /// point itself. For equivalent results with the scikit-learn-contrib/hdbscan library
        /// (which excludes self), add 1 to your value.
        /// </remarks>
        public int MinSamples
        {
            get => minSamples ?? minClusterSize;

            set
            {
                if (value < 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The minimum samples must be at least 1.");
                }

                minSamples = value;
            }
        }

        /// <summary>
        /// The method used to select clusters from the condensed tree.
        /// </summary>
        public ClusterSelectionMethod ClusterSelectionMethod
        {
            get => clusterSelectionMethod;
            set => clusterSelectionMethod = value;
        }

        /// <summary>
        /// Whether to allow a single cluster when the data forms one dense region.
        /// </summary>
        public bool AllowSingleCluster
        {
            get => allowSingleCluster;
            set => allowSingleCluster = value;
        }

        /// <summary>
        /// Whether to compute every pairwise distance once and keep them all in memory.
        /// This is the fastest way, but takes 8·n² bytes for n points (800 MB for 10,000 points,
        /// 17 GB for 46,000). The default is true.
        /// </summary>
        /// <remarks>
        /// If false, or if there are more than 46,340 points (the matrix then no longer fits in
        /// one array), distances are computed at the moment they are needed and not kept.
        /// Memory use is then proportional to n, the distance metric is called three times as
        /// often, and the result is exactly the same.
        /// </remarks>
        public bool StoreDistanceMatrix
        {
            get => storeDistanceMatrix;
            set => storeDistanceMatrix = value;
        }

        /// <summary>
        /// The maximum number of threads that compute distances at the same time, or -1 for
        /// no limit (all processors). The default is 1: no parallelism.
        /// </summary>
        /// <remarks>
        /// With a value other than 1 the distance metric is called from several threads at
        /// once, so it must be thread-safe. The result does not depend on this setting.
        /// </remarks>
        public int MaxDegreeOfParallelism
        {
            get => maxDegreeOfParallelism;

            set
            {
                if (value < 1 && value != -1)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The maximum degree of parallelism must be at least 1, or -1 for no limit.");
                }

                maxDegreeOfParallelism = value;
            }
        }
    }
}
