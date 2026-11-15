using System.Numerics;
using TruckLib;
using TruckLib.HashFs;
using TruckLib.Sii;

namespace Ets2Ad.MapExtract;

/// <summary>
/// How many lanes a road look has, and how wide the road is.
/// </summary>
/// <remarks>
/// A road on the map is one line and a token: `template46` and a curve. How
/// many lanes that is, and where their centres are, is in the road look, and
/// the look defines the lanes only by counting them -- the width is baked into
/// a model file. Which is why the widths here are measured instead; see
/// <see cref="LaneOffsetSurvey"/>.
/// </remarks>
internal sealed record RoadLook(
    int LanesLeft,
    int LanesRight,
    float RoadOffset,
    IReadOnlyList<string> LaneTypesLeft,
    IReadOnlyList<string> LaneTypesRight)
{
    public bool OneWay => LanesLeft == 0 || LanesRight == 0;
}

internal sealed class RoadLookDefinitions
{
    private readonly Dictionary<string, RoadLook> looks = [];

    public RoadLookDefinitions(IFileSystem definitions)
    {
        foreach (var path in Definitions.Files(definitions, "/def/world", "road_look"))
        {
            SiiFile file;
            try
            {
                file = SiiFile.Open(path, definitions);
            }
            catch (FileNotFoundException)
            {
                continue;
            }

            foreach (var unit in file.Units)
            {
                if (unit.Class != "road_look")
                {
                    continue;
                }
                // Unit names are "road.<token>"; the map stores the token.
                var token = unit.Name.StartsWith("road.", StringComparison.Ordinal)
                    ? unit.Name["road.".Length..]
                    : unit.Name;
                looks[token] = new RoadLook(
                    CountLanes(unit.Attributes, "lanes_left"),
                    CountLanes(unit.Attributes, "lanes_right"),
                    Number(unit.Attributes, "road_offset"),
                    LaneTypes(unit.Attributes, "lanes_left"),
                    LaneTypes(unit.Attributes, "lanes_right"));
            }
        }
    }

    public int Known => looks.Count;

    public RoadLook? Find(string token) => looks.GetValueOrDefault(token);

    private static int CountLanes(Dictionary<string, dynamic> attributes, string key)
    {
        if (!attributes.TryGetValue(key, out var value))
        {
            return 0;
        }
        // One lane is a bare pointer; several are a list of them.
        return value is System.Collections.IEnumerable and not string
            ? ((System.Collections.IEnumerable)value).Cast<object>().Count()
            : 1;
    }

    /// <summary>The traffic lane type of each lane: <c>traffic_lane.road.motorway</c>.</summary>
    private static IReadOnlyList<string> LaneTypes(Dictionary<string, dynamic> attributes, string key)
    {
        if (!attributes.TryGetValue(key, out var value) || value is null)
        {
            return [];
        }
        return value is System.Collections.IEnumerable and not string
            ? ((System.Collections.IEnumerable)value).Cast<object>().Select(Navigation.Text).ToList()
            : [Navigation.Text(value)];
    }

    private static float Number(Dictionary<string, dynamic> attributes, string key) =>
        attributes.TryGetValue(key, out var value) && value is not null
            ? Convert.ToSingle(value)
            : 0f;

}

/// <summary>
/// Where a road look's lane centres are, measured from the junctions it meets.
/// </summary>
/// <remarks>
/// The game does not state a lane width anywhere readable; it is in the road
/// template's model. But every road ends at a prefab, and a prefab's lanes are
/// exact geometry -- so the lateral offsets of the lanes arriving at a road's
/// end node *are* that road's lane offsets, stated by the game rather than
/// assumed by us.
///
/// They come out clean: template46 is a lane each way at ±2.25 m, template60
/// is two each way at ±3.25 and ±7.75, and the implied lane width is 4.5 m
/// throughout, which is the number a guess would have used -- but now it is a
/// measurement, and the narrow service roads that are 2.5 m wide would have
/// been wrong.
/// </remarks>
internal sealed class LaneOffsetSurvey
{
    private readonly Dictionary<string, List<(float Offset, bool Forward)>> samples = [];

    /// <summary>
    /// Records the lane offsets, and which way their traffic runs, where a
    /// road meets a junction.
    /// </summary>
    /// <param name="atRoadEnd">
    /// Whether this is the road's far node. It decides how to read a junction
    /// lane: one that *starts* at the road's far end is taking traffic off the
    /// road, so that lane runs with the road's direction; at the near end it
    /// is the other way round.
    /// </param>
    public void Observe(string look, Vector3 node, Vector3 direction,
        IEnumerable<LaneEnd> laneEnds, bool atRoadEnd)
    {
        var heading = Vector2.Normalize(new Vector2(direction.X, direction.Z));
        if (!float.IsFinite(heading.X))
        {
            return;
        }

        var list = samples.TryGetValue(look, out var existing) ? existing : samples[look] = [];
        foreach (var end in laneEnds)
        {
            var offset = new Vector2(end.Position.X - node.X, end.Position.Z - node.Z);
            var along = offset.X * heading.X + offset.Y * heading.Y;
            if (Math.Abs(along) > AlongTolerance)
            {
                continue;
            }
            // Left of a heading (sin, cos) is (cos, -sin).
            var across = offset.X * heading.Y - offset.Y * heading.X;
            list.Add((across, atRoadEnd == end.IsStart));
        }
    }

    /// <summary>
    /// The lane centre offsets for a look, positive to the left of the road's
    /// direction, or null if too few junctions were seen to be sure.
    /// </summary>
    /// <param name="expected">
    /// How many lanes the definition says there are. A junction contributes
    /// its own turning and slip lanes at the same node, so measurement alone
    /// sometimes finds one cluster too many -- the count is the definition's
    /// to state, and only the placement is measured. Zero means take whatever
    /// was found.
    /// </param>
    public IReadOnlyList<MeasuredLane>? Lanes(string look, int expected)
    {
        if (!samples.TryGetValue(look, out var values) || values.Count < MinimumSamples)
        {
            return null;
        }

        var sorted = values.OrderBy(v => v.Offset).ToList();
        var clusters = new List<List<(float Offset, bool Forward)>>();
        var current = new List<(float Offset, bool Forward)> { sorted[0] };
        foreach (var value in sorted.Skip(1))
        {
            if (value.Offset - current[^1].Offset < ClusterGap)
            {
                current.Add(value);
            }
            else
            {
                clusters.Add(current);
                current = [value];
            }
        }
        clusters.Add(current);

        var kept = clusters.Where(c => c.Count >= MinimumPerLane).ToList();
        if (expected > 0 && kept.Count > expected)
        {
            kept = kept.OrderByDescending(c => c.Count).Take(expected).ToList();
        }

        var lanes = kept
            .Select(cluster =>
            {
                var forward = cluster.Count(v => v.Forward);
                return new MeasuredLane(
                    cluster.Average(v => v.Offset),
                    forward * 2 >= cluster.Count,
                    // 1.0 when every junction agreed, which is what happens:
                    // no lane in the surveyed sector came back split.
                    Math.Max(forward, cluster.Count - forward) / (float)cluster.Count,
                    cluster.Count);
            })
            .OrderBy(lane => lane.Offset)
            .ToList();
        return lanes.Count > 0 ? lanes : null;
    }

    /// <summary>Metres from the node along the road: a lane end should be at it.</summary>
    private const float AlongTolerance = 0.5f;

    /// <summary>Offsets closer together than this belong to the same lane.</summary>
    private const float ClusterGap = 0.8f;

    private const int MinimumSamples = 6;
    private const int MinimumPerLane = 3;
}

/// <summary>
/// The lane endpoints inside junctions, in a grid so a road can ask what is at
/// its own end without walking all of them.
/// </summary>
internal readonly record struct LaneEnd(Vector3 Position, bool IsStart);

/// <summary>A lane offset measured at a junction, and which way its traffic runs.</summary>
internal readonly record struct MeasuredLane(float Offset, bool Forward, float Agreement, int Samples);

internal sealed class LaneEndIndex
{
    private readonly Dictionary<(int, int), List<LaneEnd>> cells = [];

    public void Add(Vector3 point, bool isStart)
    {
        var key = Cell(point);
        if (!cells.TryGetValue(key, out var list))
        {
            cells[key] = list = [];
        }
        list.Add(new LaneEnd(point, isStart));
    }

    public IEnumerable<LaneEnd> Near(Vector3 point)
    {
        var (cx, cz) = Cell(point);
        for (var dx = -1; dx <= 1; dx++)
        {
            for (var dz = -1; dz <= 1; dz++)
            {
                if (cells.TryGetValue((cx + dx, cz + dz), out var list))
                {
                    foreach (var candidate in list)
                    {
                        if (Vector2.Distance(
                                new Vector2(candidate.Position.X, candidate.Position.Z),
                                new Vector2(point.X, point.Z)) < SearchRadius)
                        {
                            yield return candidate;
                        }
                    }
                }
            }
        }
    }

    private static (int, int) Cell(Vector3 point) =>
        ((int)Math.Floor(point.X / CellSize), (int)Math.Floor(point.Z / CellSize));

    /// <summary>Wider than any carriageway, narrower than the gap to the next junction.</summary>
    private const float SearchRadius = 14f;

    private const float CellSize = 16f;
}
