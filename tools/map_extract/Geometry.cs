using System.Numerics;

namespace Ets2Ad.MapExtract;

/// <summary>
/// A rigid placement of prefab-local coordinates into the world: a yaw and a
/// translation.
/// </summary>
/// <remarks>
/// Prefabs are stored twice over. The map says where one is and which model it
/// is; the model's descriptor says what is inside it, in its own coordinates.
/// Joining the two needs the transform between them, and the map does not
/// state it -- what it states is where each of the prefab's control nodes
/// ended up. Those are the same nodes the descriptor lists, so the transform
/// is whatever maps one set onto the other, and fitting it is both simpler and
/// safer than reproducing the game's placement rules: the residual says
/// whether the answer is right.
/// </remarks>
internal readonly record struct Placement(float Yaw, Vector3 Offset, float Residual)
{
    public Vector3 Apply(Vector3 local)
    {
        var (sin, cos) = ((float)Math.Sin(Yaw), (float)Math.Cos(Yaw));
        return new Vector3(
            Offset.X + local.X * cos + local.Z * sin,
            Offset.Y + local.Y,
            Offset.Z - local.X * sin + local.Z * cos);
    }

    public Vector3 ApplyDirection(Vector3 local)
    {
        var (sin, cos) = ((float)Math.Sin(Yaw), (float)Math.Cos(Yaw));
        return new Vector3(
            local.X * cos + local.Z * sin,
            local.Y,
            -local.X * sin + local.Z * cos);
    }

    /// <summary>
    /// Fits the transform taking <paramref name="local"/> onto
    /// <paramref name="world"/>, in the ground plane, with the height taken as
    /// a mean offset.
    /// </summary>
    public static Placement Fit(IReadOnlyList<Vector3> local, IReadOnlyList<Vector3> world)
    {
        if (local.Count != world.Count || local.Count == 0)
        {
            throw new ArgumentException("need the same number of points on both sides");
        }

        var localCentre = Centroid(local);
        var worldCentre = Centroid(world);

        // The yaw that best aligns the two point sets, by the usual
        // cross/dot sum. With a single node this is zero over zero, and the
        // caller has to supply the rotation from the node instead.
        double cross = 0.0, dot = 0.0;
        for (var i = 0; i < local.Count; i++)
        {
            var a = local[i] - localCentre;
            var b = world[i] - worldCentre;
            cross += a.X * b.Z - a.Z * b.X;
            dot += a.X * b.X + a.Z * b.Z;
        }
        var yaw = (float)Math.Atan2(-cross, dot);

        var provisional = new Placement(yaw, Vector3.Zero, 0f);
        var offset = worldCentre - provisional.Apply(localCentre);

        var placement = new Placement(yaw, offset, 0f);
        var worst = 0f;
        for (var i = 0; i < local.Count; i++)
        {
            var mapped = placement.Apply(local[i]);
            worst = Math.Max(worst, Vector2.Distance(
                new Vector2(mapped.X, mapped.Z), new Vector2(world[i].X, world[i].Z)));
        }
        return placement with { Residual = worst };
    }

    /// <summary>The transform implied by one node's position and heading.</summary>
    public static Placement FromSingleNode(Vector3 local, Vector3 localDirection, Vector3 world, float worldYaw)
    {
        var localYaw = (float)Math.Atan2(localDirection.X, localDirection.Z);
        var yaw = worldYaw - localYaw;
        var provisional = new Placement(yaw, Vector3.Zero, 0f);
        return new Placement(yaw, world - provisional.Apply(local), 0f);
    }

    private static Vector3 Centroid(IReadOnlyList<Vector3> points)
    {
        var sum = Vector3.Zero;
        foreach (var point in points)
        {
            sum += point;
        }
        return sum / points.Count;
    }
}

internal static class Curves
{
    /// <summary>
    /// The direction a nav curve leaves or enters at, from its rotation.
    /// </summary>
    /// <remarks>
    /// Measured rather than assumed: on the straight curves of a prefab, where
    /// the chord is the tangent, rotating (0, 0, -1) agrees with the chord to
    /// a mean dot product of 0.975 and every other axis disagrees.
    /// </remarks>
    public static Vector3 Direction(Quaternion rotation) =>
        Vector3.Normalize(Vector3.Transform(new Vector3(0f, 0f, -1f), rotation));

    /// <summary>
    /// Samples the cubic Hermite between two oriented points, at roughly
    /// <paramref name="spacing"/> metres.
    /// </summary>
    public static List<Vector3> Sample(
        Vector3 start, Vector3 startDirection,
        Vector3 end, Vector3 endDirection,
        float length, float spacing)
    {
        // Tangent magnitude is the curve's own length: shorter and the curve
        // pinches at its ends, longer and it bulges past them.
        var startTangent = startDirection * length;
        var endTangent = endDirection * length;

        var steps = Math.Max(1, (int)Math.Ceiling(length / spacing));
        var points = new List<Vector3>(steps + 1);
        for (var step = 0; step <= steps; step++)
        {
            var t = (float)step / steps;
            var t2 = t * t;
            var t3 = t2 * t;
            points.Add(
                (2 * t3 - 3 * t2 + 1) * start
                + (t3 - 2 * t2 + t) * startTangent
                + (-2 * t3 + 3 * t2) * end
                + (t3 - t2) * endTangent);
        }
        return points;
    }
}
