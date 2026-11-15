using System.Globalization;
using TruckLib;
using TruckLib.ScsMap;
using TruckLib.Sii;

namespace Ets2Ad.MapExtract;

/// <summary>
/// What the game's GPS weighs a route by: how fast each road is.
/// </summary>
/// <remarks>
/// The GPS takes the quickest route, not the shortest: from Duisburg to
/// Nynäshamn it drove round by the Danish bridges, 83 km further than the
/// Rostock ferry, because the ferry is five and a half hours. Its time for
/// a road comes from a speed limit, and the limit from three places: the
/// road look's lanes name a traffic lane type, whose speed class
/// (local_road, expressway, motorway) the country's truck limits are given
/// by -- the urban one on city roads -- and a road side or a traffic area
/// can set a limit of its own.
/// </remarks>
internal static class Navigation
{
    /// <summary>Traffic lane type (<c>traffic_lane.road.motorway</c>) to its speed class.</summary>
    public static Dictionary<string, string> SpeedClasses(IFileSystem definitions)
    {
        var classes = new Dictionary<string, string>();
        foreach (var path in Definitions.Files(definitions, "/def/world", "traffic_lane"))
        {
            foreach (var unit in SiiFile.Open(path, definitions, true).Units)
            {
                if (unit.Class == "traffic_lane_data"
                    && unit.Attributes.TryGetValue("speed_class", out var speedClass))
                {
                    classes[unit.Name] = Text(speedClass);
                }
            }
        }
        return classes;
    }

    /// <summary>
    /// Truck speed limits by country id, then speed class: (limit, urban
    /// limit, max limit) in km/h; 0 or absent is no limit of the country's.
    /// </summary>
    /// <remarks>
    /// A country with <c>imperial_units</c> (the United Kingdom) states its
    /// limits in mph: its trucks' 60 on motorways is 96.6 km/h.
    /// </remarks>
    public static Dictionary<string, Dictionary<string, object>> TruckSpeedLimits(IFileSystem definitions)
    {
        var limits = new Dictionary<string, Dictionary<string, object>>();
        foreach (var (directory, id, imperial) in CountryDirectories(definitions))
        {
            var perClass = new Dictionary<string, object>();
            var toKmh = imperial ? KmhPerMph : 1f;
            foreach (var path in Definitions.Files(definitions, $"/def/country/{directory}", "speed_limits"))
            {
                foreach (var unit in SiiFile.Open(path, definitions, true).Units)
                {
                    if (unit.Class != "country_speed_limit"
                        || !unit.Attributes.TryGetValue("vehicle_speed_class", out var vehicle)
                        || Text(vehicle) != "truck")
                    {
                        continue;
                    }
                    var classes = List(unit.Attributes, "lane_speed_class").Select(Text).ToList();
                    var limit = List(unit.Attributes, "limit").Select(v => Number(v) * toKmh).ToList();
                    var urban = List(unit.Attributes, "urban_limit").Select(v => Number(v) * toKmh).ToList();
                    var max = List(unit.Attributes, "max_limit").Select(v => Number(v) * toKmh).ToList();
                    for (var i = 0; i < classes.Count; i++)
                    {
                        perClass[classes[i]] = new
                        {
                            limit = i < limit.Count ? limit[i] : 0f,
                            urban_limit = i < urban.Count ? urban[i] : 0f,
                            max_limit = i < max.Count ? max[i] : 0f,
                        };
                    }
                }
            }
            limits[id] = perClass;
        }
        return limits;
    }

    /// <summary>Areas that set a traffic rule -- a lower limit through a town -- as polygons.</summary>
    public static List<object> TrafficAreas(Map map) =>
        map.MapItems.Values.OfType<TrafficArea>()
            .Select(area => (object)new
            {
                rule = area.Rule.String,
                crossroad = area.CrossroadArea,
                points = area.Nodes.Select(n => new[] { n.Position.X, n.Position.Z }),
            })
            .ToList();

    /// <summary>Country definition directory (<c>germany</c>) to country id.</summary>
    private const float KmhPerMph = 1.609344f;

    private static IEnumerable<(string Directory, string Id, bool Imperial)> CountryDirectories(IFileSystem definitions)
    {
        const string Prefix = "country.data.";
        foreach (var file in Definitions.Files(definitions, "/def", "country"))
        {
            foreach (var unit in SiiFile.Open(file, definitions, true).Units)
            {
                if (unit.Class == "country_data"
                    && unit.Name.StartsWith(Prefix, StringComparison.Ordinal)
                    && unit.Attributes.TryGetValue("country_id", out var id))
                {
                    var imperial = unit.Attributes.TryGetValue("imperial_units", out var units)
                        && units is bool flag && flag;
                    yield return (unit.Name[Prefix.Length..], Convert.ToString(id, CultureInfo.InvariantCulture)!, imperial);
                }
            }
        }
    }

    private static IEnumerable<object> List(Dictionary<string, dynamic> attributes, string key)
    {
        if (!attributes.TryGetValue(key, out var value) || value is null)
        {
            return [];
        }
        return value is System.Collections.IEnumerable and not string
            ? ((System.Collections.IEnumerable)value).Cast<object>()
            : [(object)value];
    }

    /// <summary>An unquoted SII value is a token, not a string.</summary>
    public static string Text(object value) => value switch
    {
        Token token => token.String,
        // A pointer to another unit (lanes_left[]: traffic_lane.road.local).
        LinkPointer link => link.Value,
        _ => Convert.ToString(value, CultureInfo.InvariantCulture)!,
    };

    private static float Number(object value) => Convert.ToSingle(value, CultureInfo.InvariantCulture);
}
