# HdbScan.Net

[![NuGet](https://img.shields.io/nuget/v/HdbScan.Net.svg)](https://www.nuget.org/packages/HdbScan.Net)

A .NET implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

HDBSCAN extends DBSCAN by building a hierarchy of clusterings at all density levels and extracting a flat clustering based on cluster stability. Unlike k-means or GMM, it does not require specifying the number of clusters and can identify noise points.

## Installation

```
dotnet add package HdbScan.Net
```

## Usage

```csharp
using HdbScan.Net;

// Define your distance metric
Func<double[], double[], double> euclidean = (a, b) =>
{
    var sum = 0.0;
    for (var i = 0; i < a.Length; i++)
    {
        var d = a[i] - b[i];
        sum += d * d;
    }
    return Math.Sqrt(sum);
};

// Cluster your data
var options = new HdbScanOptions { MinClusterSize = 5 };
var model = new HdbScan<double[]>(points, euclidean, options);

// Results
Console.WriteLine($"Clusters found: {model.ClusterCount}");
for (var i = 0; i < model.Labels.Count; i++)
{
    Console.WriteLine($"Point {i}: cluster {model.Labels[i]}, probability {model.Probabilities[i]:F3}");
}
```

### Custom types

HDBSCAN works with any type as long as you provide a distance function:

```csharp
Func<string, string, double> hammingDistance = (a, b) =>
{
    var dist = 0;
    var len = Math.Min(a.Length, b.Length);
    for (var i = 0; i < len; i++)
        if (a[i] != b[i]) dist++;
    return dist + Math.Abs(a.Length - b.Length);
};

var model = new HdbScan<string>(words, hammingDistance);
```

### Prediction

Store prediction data to classify new points after fitting:

```csharp
var model = new HdbScan<double[]>(points, euclidean, options, predictionData: true);

var (label, probability) = model.PredictWithProbability(newPoint);
```

The clustering itself does not change. The new point is attached to the cluster hierarchy through its nearest training point (in mutual reachability distance) and labelled like a training point in that position, so a point that is not dense enough to belong to any cluster gets label `-1` (noise) and probability 0. Predicting a training point returns its fitted label and probability. This is the `approximate_predict` method of the scikit-learn-contrib/hdbscan library.

### Outlier detection

Each point receives a GLOSH outlier score between 0 and 1. Higher values indicate stronger outliers:

```csharp
for (var i = 0; i < model.OutlierScores.Count; i++)
{
    if (model.OutlierScores[i] > 0.9)
        Console.WriteLine($"Point {i} is a strong outlier (score {model.OutlierScores[i]:F3})");
}
```

### Options

| Property | Default | Description |
|---|---|---|
| `MinClusterSize` | 5 | Minimum number of points to form a cluster (>= 2) |
| `MinSamples` | `MinClusterSize` | Number of neighbors for core point definition, including the point itself (>= 1). See [sklearn compatibility](#sklearn-compatibility). |
| `ClusterSelectionMethod` | `ExcessOfMass` | `ExcessOfMass` for stable clusters, `Leaf` for fine-grained clusters |
| `AllowSingleCluster` | `false` | Whether the data may form one single cluster (points in its sparse outskirts are still noise) |
| `StoreDistanceMatrix` | `true` | Keep all pairwise distances in memory (fastest). See [large data sets](#large-data-sets). |
| `MaxDegreeOfParallelism` | 1 | Number of threads that compute distances (-1 = all processors). The distance function must then be thread-safe. |

### Distance function

The distance function must be symmetric and return finite, non-negative values; anything else throws an `ArgumentException`.

### Large data sets

The running time grows with the square of the number of points n, whatever the settings. Memory depends on how the distances are handled:

- **Stored** (the default): every pairwise distance is computed once and kept in an n×n matrix. This is the fastest, but takes 8·n² bytes: 800 MB for 10,000 points, 17 GB for 46,000.
- **On demand**: distances are computed when they are needed and not kept. Memory use is proportional to n (under 200 MB for 200,000 points), and the distance function is called three times as often. This mode is used automatically above 46,340 points, where the matrix no longer fits in a .NET array, and can be chosen for smaller data with `StoreDistanceMatrix = false`.

Both modes give exactly the same result, with or without parallelism.

```csharp
var options = new HdbScanOptions
{
    StoreDistanceMatrix = false,   // only needed below 46,340 points, to save memory
    MaxDegreeOfParallelism = -1    // use all processors; the distance function must be thread-safe
};
```

As an indication: 200,000 three-dimensional points with Euclidean distance take a little over two minutes on a 12-core laptop with `MaxDegreeOfParallelism = -1`.

## sklearn compatibility

This implementation follows the **sklearn.cluster.HDBSCAN** convention where `MinSamples` includes the point itself. Results are validated against scikit-learn's output on multiple datasets: labels (including their numbering) and probabilities are the same.

Two known differences:

- When several mutual reachability distances are exactly equal, the hierarchy is not unique and the result depends on the order in which ties are processed. scikit-learn's own `algorithm="brute"` and `algorithm="kd_tree"` can differ from each other in that case, and so can this library.
- With `ClusterSelectionMethod.Leaf` and `AllowSingleCluster = true`, data without any sub-cluster forms a single cluster here; scikit-learn returns only noise.

If you are migrating from the **scikit-learn-contrib/hdbscan** library (which excludes self from the count), add 1 to your `min_samples` value:

```csharp
// scikit-learn-contrib/hdbscan: min_samples=4
// sklearn.cluster.HDBSCAN / HdbScan.Net: MinSamples = 5
var options = new HdbScanOptions { MinSamples = 5 };
```

## A note on AI assistance

This library was developed with the help of AI (Claude). A human was in the loop for design decisions and review, and correctness is not taken on faith: the implementation follows the original HDBSCAN* paper and its results are validated against scikit-learn's `HDBSCAN` output on multiple datasets (see the test suite). If you spot anything odd, please open an issue — bug reports are very welcome.

## Reference

Campello, R.J.G.B., Moulavi, D., Zimek, A., Sander, J. (2015). "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection." ACM Trans. Knowl. Discov. Data 10, 1, Article 5 (July 2015). https://doi.org/10.1145/2733381

## License

MIT
