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
| `MinSamples` | `MinClusterSize` | Number of neighbors for core point definition, including the point itself (>= 2). See [sklearn compatibility](#sklearn-compatibility). |
| `ClusterSelectionMethod` | `ExcessOfMass` | `ExcessOfMass` for stable clusters, `Leaf` for fine-grained clusters |
| `AllowSingleCluster` | `false` | Whether to allow all points in a single cluster |

## sklearn compatibility

This implementation follows the **sklearn.cluster.HDBSCAN** convention where `MinSamples` includes the point itself. Results are validated against scikit-learn's output on multiple datasets.

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
