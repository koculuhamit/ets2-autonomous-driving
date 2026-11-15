using Ets2Ad.MapExtract;
using System.Globalization;
using System.Text.Json;
using TruckLib.HashFs;
using TruckLib.ScsMap;

// Offline map extraction: read the game's own road network out of its archives
// so the stack can plan on it, instead of following a drive somebody recorded
// by hand.
//
// Runs once, writes a cache, and is never part of a driving loop. See ADR-0006
// for why this is C# and why it stays on this side of the fence.

return args switch
{
    ["list", var archive, var directory] => List(archive, directory),
    ["cat", var archive, var file] => Cat(archive, file),
    ["roads", var game, var output, .. var rest] => Roads(game, output, rest),
    _ => Usage(),
};

static int Usage()
{
    Console.Error.WriteLine("""
        usage:
          map_extract list  <archive.scs> <directory>
          map_extract cat   <archive.scs> <file>
          map_extract roads <game directory> <output.json> all
          map_extract roads <game directory> <output.json> <sectorX,sectorZ> [...]

        Reads base_map.scs and every installed DLC archive that has map
        sectors, the way the game mounts them.

        A sector is 4000 m square; the one under a position is
        (floor(x / 4000), floor(z / 4000)).
        """);
    return 2;
}

static int List(string archive, string directory)
{
    using var reader = HashFsReader.Open(archive);
    Console.WriteLine($"{Path.GetFileName(archive)}: {reader.Entries.Count} entries");
    var listing = reader.GetDirectoryListing(directory);
    foreach (var subdirectory in listing.Subdirectories.Order())
    {
        Console.WriteLine($"  dir  {subdirectory}");
    }
    foreach (var file in listing.Files.Order())
    {
        Console.WriteLine($"  file {file}");
    }
    return 0;
}

static int Cat(string archive, string file)
{
    using var reader = HashFsReader.Open(archive);
    using var output = Console.OpenStandardOutput();
    output.Write(reader.Extract(file)[0]);
    return 0;
}

static int Roads(string gameDirectory, string output, IReadOnlyList<string> sectorArguments)
{
    // A map DLC is not only a licence: it ships sectors of its own. Scandinavia
    // is 68 sectors in dlc_north.scs that base_map.scs does not have -- 245 000
    // items -- plus the definitions and prefab descriptors they use. Reading
    // base_map.scs alone had left Sweden and Norway as islands.
    var dlcs = GameArchives.MapDlcs(gameDirectory);
    var mapArchives = new[] { "base_map.scs" }.Concat(dlcs).ToList();
    using var reader = new LayeredFileSystem(mapArchives.Select(a => Path.Combine(gameDirectory, a)));

    var sectors = new List<SectorCoordinate>();
    if (sectorArguments is ["all"])
    {
        sectors = reader.GetFiles("/map/europe")
            .Where(f => f.EndsWith(".base", StringComparison.Ordinal))
            .Select(Sector.SectorCoordsFromSectorFilePath)
            .Distinct()
            .ToList();
    }
    else
    {
        foreach (var argument in sectorArguments)
        {
            var parts = argument.Split(',');
            if (parts.Length != 2
                || !int.TryParse(parts[0], CultureInfo.InvariantCulture, out var x)
                || !int.TryParse(parts[1], CultureInfo.InvariantCulture, out var z))
            {
                Console.Error.WriteLine($"not a sector coordinate: {argument}");
                return 2;
            }
            sectors.Add(new SectorCoordinate(x, z));
        }
    }
    if (sectors.Count == 0)
    {
        Console.Error.WriteLine("no sectors given: name some, or 'all'");
        return 2;
    }
    Console.WriteLine($"map archives: {string.Join(", ", mapArchives)}; {sectors.Count} sectors");

    var map = Map.Open("/map/europe.mbd", reader, sectors);
    using var definitions = new LayeredFileSystem(
        new[] { "def.scs" }.Concat(dlcs).Select(a => Path.Combine(gameDirectory, a)));
    using var resolver = new PrefabResolver(gameDirectory, definitions, dlcs);
    var lookDefinitions = new RoadLookDefinitions(definitions);
    var countries = Countries(definitions);
    var speedClasses = Navigation.SpeedClasses(definitions);
    var truckSpeedLimits = Navigation.TruckSpeedLimits(definitions);
    var semaphoreProfiles = SemaphoreProfiles(definitions);

    // Prefabs first: the junctions are what the roads are measured against.
    var prefabs = new List<object>();
    var laneEnds = new LaneEndIndex();
    foreach (var prefab in map.MapItems.Values.OfType<Prefab>())
    {
        prefabs.Add(Describe(prefab, resolver, laneEnds));
    }

    // Where deliveries go: every company depot and every city, by the same ids
    // the telemetry's job carries, so a job's destination can be found on the
    // map. The depot's parking spots come with it for when the truck has to
    // back a trailer into one.
    var companies = map.MapItems.Values.OfType<Company>()
        .Select(c => new
        {
            company = c.CompanyName.String,
            city = c.CityName.String,
            position = new[] { c.Node.Position.X, c.Node.Position.Y, c.Node.Position.Z },
            prefab = c.Prefab is MapItem owner ? owner.Uid.ToString("X16") : null,
            spots = c.SpawnPoints.Select(s =>
            {
                var direction = Curves.Direction(s.Node.Rotation);
                return new
                {
                    type = s.DepotType.ToString(),
                    difficulty = s.Difficulty.ToString(),
                    position = new[] { s.Node.Position.X, s.Node.Position.Y, s.Node.Position.Z },
                    // Which way the spot faces, as the node is turned: the
                    // same axis as a nav curve's (Curves.Direction).
                    direction = new[] { direction.X, direction.Z },
                };
            }),
        })
        .ToList();
    var cities = map.MapItems.Values.OfType<CityArea>()
        .Select(c => new
        {
            city = c.Name.String,
            position = new[] { c.Node.Position.X, c.Node.Position.Y, c.Node.Position.Z },
            width = c.Width,
            height = c.Height,
            shown = c.ShowInUi,
        })
        .ToList();

    // Where the sea is crossed: a ferry item stands in a port's prefab, where
    // the truck drives on; the game puts it ashore at the other port's item,
    // moved by that item's unload offset. The Channel Tunnel is one too, a
    // train rather than a ship. Which ports are joined, and how far, is in
    // the definitions (FerryConnections).
    var ferries = map.MapItems.Values.OfType<Ferry>()
        .Select(f => new
        {
            uid = f.Uid.ToString("X16"),
            port = f.Port.String,
            train = f.TrainTransport,
            position = new[] { f.Node.Position.X, f.Node.Position.Y, f.Node.Position.Z },
            unload_offset = new[] { f.UnloadOffset.X, f.UnloadOffset.Y, f.UnloadOffset.Z },
            prefab = f.Prefab is MapItem owner ? owner.Uid.ToString("X16") : null,
        })
        .ToList();
    var ferryConnections = FerryConnections(definitions);

    var survey = new LaneOffsetSurvey();
    var roads = new List<object>();
    var looksSeen = new HashSet<string>();
    foreach (var item in map.MapItems.Values)
    {
        if (item is not Road road)
        {
            continue;
        }

        // The road item is a Hermite curve between two nodes, and its own
        // Length is the arc length. Sample it at a fixed spacing so the Python
        // side gets geometry rather than a spline it would have to reimplement.
        var points = new List<float[]>();
        var steps = Math.Max(1, (int)Math.Ceiling(road.Length / SampleSpacing));
        for (var step = 0; step <= steps; step++)
        {
            var point = road.InterpolateCurve((float)step / steps);
            points.Add([point.Position.X, point.Position.Y, point.Position.Z]);
        }

        // Where this road meets a junction, the lanes arriving there are at
        // this road look's lane offsets -- stated by the game rather than
        // assumed by us. See LaneOffsetSurvey.
        var look = road.RoadType.String;
        looksSeen.Add(look);
        if (points.Count >= 2)
        {
            var first = ToVector(points[0]);
            var second = ToVector(points[1]);
            var last = ToVector(points[^1]);
            var penultimate = ToVector(points[^2]);
            // Surveyed separately per side of the road the traffic keeps to.
            // The same look is used in Britain and on the continent, and
            // pooling the two made the motorway looks a coin toss: look1 came
            // back 54% one way, and look8 came back backwards.
            var key = SurveyKey(look, road.LeftHandTraffic);
            survey.Observe(key, first, second - first, laneEnds.Near(first), atRoadEnd: false);
            survey.Observe(key, last, last - penultimate, laneEnds.Near(last), atRoadEnd: true);
        }

        roads.Add(new
        {
            uid = road.Uid.ToString("X16"),
            road_look = road.RoadType.String,
            length = road.Length,
            start_node = road.Node.Uid.ToString("X16"),
            end_node = road.ForwardNode.Uid.ToString("X16"),
            ai_vehicles = road.AiVehicles,
            left_hand_traffic = road.LeftHandTraffic,
            // Which DLC this piece of road belongs to, 0 for the base game.
            // base_map.scs carries every map DLC's roads on every installation;
            // whether a player can drive one is decided by this, per item.
            dlc_guard = road.DlcGuard,
            country = road.Node.ForwardCountry,
            hidden = !road.ShowInUiMap,
            // What the GPS weighs it by (see Navigation): the urban limits
            // apply on a city road, a side's rule can set a limit of its
            // own, and a road the GPS is told to avoid it goes round.
            // TruckLib 0.5.1 marks this obsolete with nothing in its place;
            // the flag is still read from the map, and the GPS still weighs it.
#pragma warning disable CS0612
            city_road = road.IsCityRoad,
#pragma warning restore CS0612
            gps_avoid = road.GpsAvoid,
            rule_left = road.Left.TrafficRule.String,
            rule_right = road.Right.TrafficRule.String,
            points,
        });
    }

    var document = new
    {
        format = "ets2ad-map-roads",
        format_version = 8,
        // Which archives this is the map of: what the installation had.
        archives = mapArchives,
        sector_size = Map.SectorSize,
        sample_spacing = SampleSpacing,
        sectors = sectors.Select(s => new[] { s.X, s.Z }),
        node_count = map.Nodes.Count,
        item_count = map.MapItems.Count,
        // Country id, as the map's nodes carry it, to ISO 3166 alpha-3.
        countries,
        // How the game counts distance: world metres times these, the city
        // one inside a city's area.
        normal_scale = map.NormalScale,
        city_scale = map.CityScale,
        // What the GPS weighs a road by (see Navigation).
        speed_classes = speedClasses,
        truck_speed_limits = truckSpeedLimits,
        traffic_areas = Navigation.TrafficAreas(map),
        // What the traffic lights and barriers on the map are: a prefab's
        // semaphore names a profile, and the profile says whether it is a
        // light or a barrier, and when a light sleeps -- dark, from 23:30 to
        // 03:00 for many of them.
        semaphore_profiles = semaphoreProfiles,
        road_looks = looksSeen.Order().ToDictionary(look => look, look =>
        {
            var definition = lookDefinitions.Find(look);
            var count = (definition?.LanesLeft ?? 0) + (definition?.LanesRight ?? 0);
            var measured = survey.Lanes(SurveyKey(look, false), count);
            var leftHand = survey.Lanes(SurveyKey(look, true), count);
            return (object)new
            {
                // Counted from the definition file...
                lanes_left = definition?.LanesLeft,
                lanes_right = definition?.LanesRight,
                lane_types_left = definition?.LaneTypesLeft,
                lane_types_right = definition?.LaneTypesRight,
                road_offset = definition?.RoadOffset,
                // ...but placed, and pointed, by measurement. Offsets are
                // positive to the left; forward means with the road's own
                // direction. A one-way road has every lane forward, which no
                // rule about which side of the centreline a lane sits on would
                // have got right.
                lane_offsets = measured?.Select(lane => lane.Offset),
                lane_forward = measured?.Select(lane => lane.Forward),
                lane_agreement = measured?.Select(lane => lane.Agreement),
                lane_samples = measured?.Select(lane => lane.Samples),
                // The same, measured on roads where traffic keeps left. Null
                // where the look is not used there, or too rarely to tell.
                left_hand_traffic = leftHand is null ? null : new
                {
                    lane_offsets = leftHand.Select(lane => lane.Offset),
                    lane_forward = leftHand.Select(lane => lane.Forward),
                    lane_agreement = leftHand.Select(lane => lane.Agreement),
                    lane_samples = leftHand.Select(lane => lane.Samples),
                },
            };
        }),
        roads,
        prefabs,
        companies,
        cities,
        ferries,
        ferry_connections = ferryConnections,
    };

    using (var stream = File.Create(output))
    {
        JsonSerializer.Serialize(stream, document, JsonOptions);
    }

    Console.WriteLine(
        $"{map.MapItems.Count} items, {map.Nodes.Count} nodes, "
        + $"{roads.Count} roads, {prefabs.Count} prefabs -> {output}");
    return 0;
}

static object Describe(Prefab prefab, PrefabResolver resolver, LaneEndIndex laneEnds)
{
    var nodes = prefab.Nodes.Select(n => new
    {
        // The same map node the road attached here also carries, so the two
        // halves of the network join by identity rather than by proximity.
        uid = n.Uid.ToString("X16"),
        position = new[] { n.Position.X, n.Position.Y, n.Position.Z },
    });
    var descriptor = resolver.Describe(prefab.Model.String);
    if (descriptor is null)
    {
        return new
        {
            uid = prefab.Uid.ToString("X16"),
            model = prefab.Model.String,
            dlc_guard = prefab.DlcGuard,
            country = prefab.Nodes.Count > 0 ? prefab.Nodes[0].ForwardCountry : (byte)0,
            nodes,
        };
    }

    var placement = PrefabResolver.Place(prefab, descriptor);
    var lanes = new List<object>();
    foreach (var curve in descriptor.NavCurves)
    {
        var points = Curves.Sample(
            curve.StartPosition, Curves.Direction(curve.StartRotation),
            curve.EndPosition, Curves.Direction(curve.EndRotation),
            curve.Length, SampleSpacing);

        var world = points.Select(placement.Apply).ToList();
        laneEnds.Add(world[0], isStart: true);
        laneEnds.Add(world[^1], isStart: false);

        // Where this lane joins a road, if it does. The descriptor has four
        // bytes that look like they say so, but what they mean is not
        // documented and the values do not survive a sanity check, so the
        // attachment is worked out from geometry instead: an endpoint lying in
        // a control node's plane is attached to that node, and its distance to
        // either side of the node is the lane's offset there.
        var entry = Attachment(descriptor, placement, world[0]);
        var exit = Attachment(descriptor, placement, world[^1]);

        lanes.Add(new
        {
            length = curve.Length,
            // The map node's own uid, not the descriptor's node index: the map
            // item's node list is rotated to start at the prefab's origin, so
            // an index is only meaningful to someone who knows the rotation.
            // Handing Python the index put 1038 junction joins on the wrong
            // node across Europe.
            from_node = entry is { } e ? MapNode(prefab, e.Node) : null,
            from_offset = entry?.Offset,
            to_node = exit is { } x ? MapNode(prefab, x.Node) : null,
            to_offset = exit?.Offset,
            // Indices into this prefab's own curve list: the graph inside the
            // junction, which is what a route across it is made of.
            next = curve.NextLines.ToArray(),
            previous = curve.PreviousLines.ToArray(),
            // -1 when no light governs this lane.
            semaphore = curve.SemaphoreId,
            rule = curve.TrafficRule.String,
            blinker = curve.Blinker.ToString(),
            points = world.Select(p => new[] { p.X, p.Y, p.Z }),
        });
    }

    return new
    {
        uid = prefab.Uid.ToString("X16"),
        model = prefab.Model.String,
        dlc_guard = prefab.DlcGuard,
        country = prefab.Nodes.Count > 0 ? prefab.Nodes[0].ForwardCountry : (byte)0,
        nodes,
        // How far the fitted placement misses the prefab's own nodes by. It is
        // the only check that the descriptor was put where the map wanted it.
        placement_residual = placement.Residual,
        // The light timing the map gives this junction. A universal
        // junction's descriptor has its lights whether or not the map lights
        // them; without a profile here they are not there in the game.
        semaphore_profile = prefab.SemaphoreProfile.String,
        lanes,
        semaphores = descriptor.Semaphores.Select(s =>
        {
            var world = placement.Apply(s.Position);
            return new
            {
                id = s.SemaphoreId,
                type = s.Type.ToString(),
                profile = s.Profile.String,
                position = new[] { world.X, world.Y, world.Z },
            };
        }),
        // The prefab's map drawing -- roads and the yard's polygons -- for the
        // depots a truck parks in and the platforms it drives off (only where
        // the prefab has a place to park or unload, to keep the file small):
        // where there is surface to drive on that has no lanes.
        map_points = HasYard(descriptor)
            ? descriptor.MapPoints.Select(m =>
            {
                var world = placement.Apply(m.Position);
                return new
                {
                    position = new[] { world.X, world.Z },
                    neighbors = m.Neighbors.ToArray(),
                    size = m.RoadSize.ToString(),
                    offset = m.RoadOffset.ToString(),
                    color = m.Color.ToString(),
                    road_over = m.RoadOver,
                };
            }).ToList()
            : null,
        // Where a truck stands to fill up: the descriptor's gas station spawn
        // points, one per pump lane.
        gas_pumps = descriptor.SpawnPoints
            .Where(p => p.Type == TruckLib.Models.Ppd.SpawnPointType.GasStation)
            .Select(p =>
            {
                var world = placement.Apply(p.Position);
                return new[] { world.X, world.Y, world.Z };
            }),
    };
}

static System.Numerics.Vector3 ToVector(float[] point) => new(point[0], point[1], point[2]);

/// <summary>Whether a prefab has somewhere to park or unload a trailer.</summary>
static bool HasYard(TruckLib.Models.Ppd.PrefabDescriptor descriptor) =>
    descriptor.SpawnPoints.Any(p => p.Type is
        TruckLib.Models.Ppd.SpawnPointType.UnloadEasy
        or TruckLib.Models.Ppd.SpawnPointType.UnloadMedium
        or TruckLib.Models.Ppd.SpawnPointType.UnloadHard
        or TruckLib.Models.Ppd.SpawnPointType.UnloadRigid
        or TruckLib.Models.Ppd.SpawnPointType.CompanyUnload
        or TruckLib.Models.Ppd.SpawnPointType.Trailer
        or TruckLib.Models.Ppd.SpawnPointType.TrailerSpawn
        or TruckLib.Models.Ppd.SpawnPointType.LongTrailer
        or TruckLib.Models.Ppd.SpawnPointType.Parking);

/// <summary>The control node a lane end joins, and its offset across it.</summary>
static LaneAttachment? Attachment(
    TruckLib.Models.Ppd.PrefabDescriptor descriptor,
    Placement placement,
    System.Numerics.Vector3 end)
{
    for (var index = 0; index < descriptor.Nodes.Count; index++)
    {
        var node = placement.Apply(descriptor.Nodes[index].Position);
        var heading = placement.ApplyDirection(descriptor.Nodes[index].Direction);
        var flat = System.Numerics.Vector2.Normalize(
            new System.Numerics.Vector2(heading.X, heading.Z));
        if (!float.IsFinite(flat.X))
        {
            continue;
        }

        var offset = new System.Numerics.Vector2(end.X - node.X, end.Z - node.Z);
        var along = offset.X * flat.X + offset.Y * flat.Y;
        var across = offset.X * flat.Y - offset.Y * flat.X;
        if (Math.Abs(along) < AttachmentTolerance && Math.Abs(across) < HalfRoadWidth)
        {
            return new LaneAttachment(index, across);
        }
    }
    return null;
}

/// <summary>Every ferry line the definitions state: from which port to which,
/// and its price, time and distance.</summary>
static List<object> FerryConnections(TruckLib.IFileSystem definitions)
{
    const string Prefix = "conn.";
    var connections = new List<object>();
    foreach (var file in definitions.GetFiles("/def/ferry/connection")
                 .Where(f => f.EndsWith(".sii", StringComparison.Ordinal)))
    {
        foreach (var unit in TruckLib.Sii.SiiFile.Open(file, definitions, true).Units)
        {
            if (unit.Class != "ferry_connection" || !unit.Name.StartsWith(Prefix, StringComparison.Ordinal))
            {
                continue;
            }
            var ports = unit.Name[Prefix.Length..].Split('.');
            if (ports.Length != 2)
            {
                continue;
            }
            int? Number(string key) => unit.Attributes.TryGetValue(key, out var value)
                ? Convert.ToInt32(value, CultureInfo.InvariantCulture)
                : null;
            connections.Add(new
            {
                from = ports[0],
                to = ports[1],
                price = Number("price"),
                time = Number("time"),
                distance = Number("distance"),
            });
        }
    }
    return connections;
}

static Dictionary<string, string> Countries(TruckLib.IFileSystem definitions)
{
    var countries = new Dictionary<string, string>();
    foreach (var file in Definitions.Files(definitions, "/def", "country"))
    {
        foreach (var unit in TruckLib.Sii.SiiFile.Open(file, definitions, true).Units)
        {
            if (unit.Class == "country_data"
                && unit.Attributes.TryGetValue("country_id", out var id)
                && unit.Attributes.TryGetValue("iso_country_code", out var iso))
            {
                // An unquoted SII value is a token, not a string.
                object code = iso;
                countries[Convert.ToString(id, CultureInfo.InvariantCulture)!] =
                    code is TruckLib.Token token ? token.String : Convert.ToString(code)!;
            }
        }
    }
    return countries;
}

/// <summary>Every semaphore profile, as the definition files state it.</summary>
/// <remarks>
/// Inheritance is left to the reader: a profile's own attributes, and the
/// name of the one it inherits the rest from.
/// </remarks>
static Dictionary<string, object> SemaphoreProfiles(TruckLib.IFileSystem definitions)
{
    const string Prefix = "tr_sem_prof.";
    var profiles = new Dictionary<string, object>();
    foreach (var file in Definitions.Files(definitions, "/def/world", "semaphore_profile"))
    {
        foreach (var unit in TruckLib.Sii.SiiFile.Open(file, definitions, true).Units)
        {
            if (unit.Class != "tr_semaphore_profile" || !unit.Name.StartsWith(Prefix, StringComparison.Ordinal))
            {
                continue;
            }
            var attributes = unit.Attributes;
            string? Text(object? value) => value switch
            {
                null => null,
                TruckLib.Token token => token.String,
                // A value that names another unit, such as `type[0]: traffic_light`.
                TruckLib.Sii.LinkPointer link => link.Value,
                _ => Convert.ToString(value, CultureInfo.InvariantCulture),
            };
            int? Minutes(string key) =>
                attributes.TryGetValue(key, out var value) ? Convert.ToInt32(value, CultureInfo.InvariantCulture) : null;
            var types = attributes.TryGetValue("type", out var typeValue) && typeValue is System.Collections.IEnumerable list
                    && typeValue is not string
                ? list.Cast<object>().Select(Text).Where(t => !string.IsNullOrEmpty(t)).ToList()
                : new List<string?>();
            var inherited = attributes.TryGetValue("inherited", out var parent) ? Text(parent) : null;
            profiles[unit.Name[Prefix.Length..]] = new
            {
                inherited = inherited is not null && inherited.StartsWith(Prefix, StringComparison.Ordinal)
                    ? inherited[Prefix.Length..]
                    : inherited,
                types,
                sleep_start = Minutes("sleep_time_start"),
                sleep_end = Minutes("sleep_time_end"),
            };
        }
    }
    return profiles;
}

static string SurveyKey(string look, bool leftHandTraffic) =>
    leftHandTraffic ? look + "@lht" : look;

/// <summary>The uid of the map node a descriptor node became.</summary>
static string MapNode(Prefab prefab, int descriptorIndex)
{
    var count = prefab.Nodes.Count;
    var index = ((descriptorIndex - prefab.Origin) % count + count) % count;
    return prefab.Nodes[index].Uid.ToString("X16");
}

internal readonly record struct LaneAttachment(int Node, float Offset);

internal static partial class Program
{
    /// <summary>Metres between sampled points along a road.</summary>
    private const float SampleSpacing = 2f;

    /// <summary>How far off a node's plane an endpoint may be and still join it.</summary>
    private const float AttachmentTolerance = 1.0f;

    /// <summary>Wider than any carriageway in the game.</summary>
    private const float HalfRoadWidth = 20f;

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };
}
