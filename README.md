# HdbScan.Net

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

### Options

| Property | Default | Description |
|---|---|---|
| `MinClusterSize` | 5 | Minimum number of points to form a cluster (>= 2) |
| `MinSamples` | 0 (= MinClusterSize) | Number of neighbors for core point definition |
| `ClusterSelectionMethod` | `ExcessOfMass` | `ExcessOfMass` for stable clusters, `Leaf` for fine-grained clusters |
| `AllowSingleCluster` | `false` | Whether to allow all points in a single cluster |

## Reference

Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." PAKDD 2013. Lecture Notes in Computer Science, vol 7819. Springer.

## License

MIT
