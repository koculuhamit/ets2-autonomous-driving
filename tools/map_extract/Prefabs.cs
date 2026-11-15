using System.Numerics;
using TruckLib;
using TruckLib.HashFs;
using TruckLib.Models.Ppd;
using TruckLib.ScsMap;
using TruckLib.Sii;

namespace Ets2Ad.MapExtract;

/// <summary>
/// Resolves a prefab on the map into the drivable geometry inside it.
/// </summary>
/// <remarks>
/// A junction, roundabout, service area or company yard is a prefab, and the
/// map file holds only its position and the name of its model. What is inside
/// -- the lanes through it, which ones lead to which, which traffic light
/// governs them -- is in a .ppd descriptor in base.scs, found through
/// /def/world/prefab.sii. Without this, the road network has a hole at every
/// junction, and a route that cannot cross a roundabout is not a route.
/// </remarks>
internal sealed class PrefabResolver : IDisposable
{
    private readonly LayeredFileSystem models;
    private readonly Dictionary<string, string> descriptorPaths = [];
    private readonly Dictionary<string, PrefabDescriptor?> descriptors = [];

    public PrefabResolver(string gameDirectory, IFileSystem definitions, IEnumerable<string> dlcs)
    {
        // A DLC's junctions are described in its own archive.
        models = new LayeredFileSystem(
            ModelArchives.Concat(dlcs).Select(name => Path.Combine(gameDirectory, name)));

        var units = Definitions.Files(definitions, "/def/world", "prefab")
            .SelectMany(file => SiiFile.Open(file, definitions).Units);
        foreach (var unit in units)
        {
            if (!unit.Attributes.TryGetValue("prefab_desc", out var path))
            {
                continue;
            }
            // Unit names are "prefab.<token>"; the map stores the token.
            var token = unit.Name.StartsWith("prefab.", StringComparison.Ordinal)
                ? unit.Name["prefab.".Length..]
                : unit.Name;
            descriptorPaths[token] = (string)path;
        }
    }

    public int KnownModels => descriptorPaths.Count;

    public PrefabDescriptor? Describe(string model)
    {
        if (descriptors.TryGetValue(model, out var cached))
        {
            return cached;
        }

        PrefabDescriptor? descriptor = null;
        if (descriptorPaths.TryGetValue(model, out var path))
        {
            try
            {
                descriptor = PrefabDescriptor.Open(path, models);
            }
            catch (Exception exception) when (exception is IOException or InvalidDataException)
            {
                descriptor = null;
            }
        }
        descriptors[model] = descriptor;
        return descriptor;
    }

    /// <summary>
    /// Where this prefab's descriptor coordinates land in the world.
    /// </summary>
    public static Placement Place(Prefab prefab, PrefabDescriptor descriptor)
    {
        // The map item's node list is rotated so that its first entry is the
        // descriptor's origin node.
        var count = Math.Min(prefab.Nodes.Count, descriptor.Nodes.Count);
        var local = new List<Vector3>(count);
        var world = new List<Vector3>(count);
        for (var i = 0; i < count; i++)
        {
            var index = ((i - prefab.Origin) % count + count) % count;
            local.Add(descriptor.Nodes[i].Position);
            world.Add(prefab.Nodes[index].Position);
        }

        if (count == 1)
        {
            var rotation = prefab.Nodes[0].Rotation;
            var forward = Vector3.Transform(new Vector3(0f, 0f, 1f), rotation);
            return Placement.FromSingleNode(
                local[0], descriptor.Nodes[0].Direction, world[0],
                (float)Math.Atan2(forward.X, forward.Z));
        }
        return Placement.Fit(local, world);
    }

    public void Dispose() => models.Dispose();

    /// <summary>Where prefab descriptors live, in priority order.</summary>
    private static readonly string[] ModelArchives = ["base.scs", "base_share.scs"];
}
