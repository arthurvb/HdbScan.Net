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

        /// <summary>
        /// Creates an instance of <see cref="HdbScanOptions"/> with default parameters.
        /// </summary>
        public HdbScanOptions()
        {
            minClusterSize = 5;
            minSamples = null;
            clusterSelectionMethod = ClusterSelectionMethod.ExcessOfMass;
            allowSingleCluster = false;
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
        /// including the point itself. Must be at least 2.
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
                if (value < 2)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The minimum samples must be at least 2.");
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
    }
}
