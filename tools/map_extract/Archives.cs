using System.Text;
using TruckLib;
using TruckLib.HashFs;

namespace Ets2Ad.MapExtract;

/// <summary>
/// Several archives read as one, the way the game mounts them.
/// </summary>
/// <remarks>
/// The game does not keep a model in the archive its definition is in, nor
/// all definitions of one kind in one file. Reading only the archive and the
/// file that seemed obvious lost every "universal road" junction on the map,
/// 3340 of 20593 prefabs, and left 2874 roads with no lanes.
/// </remarks>
internal sealed class LayeredFileSystem : IFileSystem, IDisposable
{
    private readonly List<IHashFsReader> layers;

    /// <param name="archives">In priority order: the first that has a path wins.</param>
    public LayeredFileSystem(IEnumerable<string> archives)
    {
        layers = archives.Where(File.Exists).Select(HashFsReader.Open).ToList();
        if (layers.Count == 0)
        {
            throw new FileNotFoundException("none of the archives exist");
        }
    }

    public char DirectorySeparator => '/';

    public bool FileExists(string path) => layers.Any(l => l.EntryExists(path) == EntryType.File);

    public bool DirectoryExists(string path) =>
        layers.Any(l => l.EntryExists(path) == EntryType.Directory);

    /// <summary>Every file under a directory in any layer, once.</summary>
    public IList<string> GetFiles(string path) =>
        layers
            .Where(l => l.EntryExists(path) == EntryType.Directory)
            .SelectMany(l => l.GetDirectoryListing(path).Files)
            .Distinct()
            .ToList();

    public byte[] ReadAllBytes(string path) => Owner(path).Extract(path)[0];

    public string ReadAllText(string path) => Encoding.UTF8.GetString(ReadAllBytes(path));

    public string ReadAllText(string path, Encoding encoding) => encoding.GetString(ReadAllBytes(path));

    public Stream Open(string path) => new MemoryStream(ReadAllBytes(path));

    public string GetParent(string path)
    {
        var index = path.TrimEnd('/').LastIndexOf('/');
        return index <= 0 ? "/" : path[..index];
    }

    public void Dispose()
    {
        foreach (var layer in layers)
        {
            layer.Dispose();
        }
    }

    private IHashFsReader Owner(string path) =>
        layers.FirstOrDefault(l => l.EntryExists(path) == EntryType.File)
        ?? throw new FileNotFoundException(path);
}

internal static class Definitions
{
    /// <summary>
    /// All definition files of one kind: the base file and every
    /// <c>name.suffix.sii</c> beside it, which the game loads as well.
    /// </summary>
    /// <remarks>
    /// <c>/def/world</c> has eleven prefab definition files, not one. The
    /// "universal roads" -- used all over the map -- are defined only in
    /// <c>prefab.mod_universal_roads.sii</c> and
    /// <c>road_look.template.mod_universal_roads.sii</c>.
    /// </remarks>
    public static IEnumerable<string> Files(IFileSystem definitions, string directory, string kind) =>
        definitions
            .GetFiles(directory)
            .Where(file =>
            {
                var name = file[(file.LastIndexOf('/') + 1)..];
                return name.StartsWith(kind + ".", StringComparison.Ordinal)
                    && name.EndsWith(".sii", StringComparison.Ordinal);
            })
            // The base file first, so a suffixed file can override it.
            .OrderBy(file => file.Length)
            .ThenBy(file => file, StringComparer.Ordinal);
}

internal static class GameArchives
{
    /// <summary>
    /// The installed DLC archives that carry map sectors, in a stable order.
    /// </summary>
    /// <remarks>
    /// Which DLCs a player has is which archives Steam has installed; the map
    /// DLCs among them are the ones with a <c>/map/europe</c> of their own.
    /// Their sectors do not overlap base_map.scs's, so reading them together
    /// adds regions rather than replacing any.
    /// </remarks>
    public static List<string> MapDlcs(string gameDirectory) =>
        Directory.GetFiles(gameDirectory, "dlc_*.scs")
            .Where(path =>
            {
                using var archive = HashFsReader.Open(path);
                return archive.EntryExists("/map/europe") == EntryType.Directory;
            })
            .Select(Path.GetFileName)
            .Order(StringComparer.Ordinal)
            .ToList()!;
}
