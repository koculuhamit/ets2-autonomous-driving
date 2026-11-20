# Reading the game's own road network

**Date:** 2026-09-22
**Game:** ETS2 **1.61.1.0**, Linux native (the pack set version in `game.log.txt`)
**Archive:** `base_map.scs`, 324 MB, HashFS v2, 2180 entries
**Tool:** `tools/map_extract` (C#, .NET 10, TruckLib 0.5.1) — see ADR-0006

First step of Phase 4: get the road network out of the archives, and find out
whether what comes out lines up with where the truck actually is.

## What is in there

`base_map.scs` holds `/map/europe.mbd` and `/map/europe/`, which is 1410 files:
five per sector, 282 sectors. A sector is 4000 m square, so the sector under a
position is `(floor(x / 4000), floor(z / 4000))`.

Only one other archive carries map sectors, `dlc_north.scs`. Every other DLC on
this installation is vehicles, paint and cargo.

Sector (−2, +7), the one the taught loop is in:

| | |
|---|---|
| items | 11 109 |
| nodes | 14 602 |
| roads | 804 |
| prefabs | 268 |
| lanes inside prefabs | 4321 |
| semaphores | 300 |
| extraction time | 1.2 s, or 4.8 s with prefab interiors |

Nine sectors together: 35 988 items, 2446 roads, 780 prefabs, still seconds.
Loading all 282 is not needed and was not tried.

## Does it line up with reality

The question that decides whether any of this is usable: the map says a road is
at some coordinate, and the telemetry says the truck is at some coordinate. Are
they the same coordinate?

**They are, with no transform at all.** The 976 m taught loop was recorded at
x ∈ [−6636, −6407], z ∈ [29050, 29327]; the roads extracted around it sit in
exactly that range. No scaling, no offset, no axis swap.

Distance from each recorded position to the nearest extracted road centreline:

| | |
|---|---|
| 10th percentile | 2.43 m |
| median | 6.04 m |
| 90th percentile | 19.2 m |
| worst | 62.8 m |
| within 8 m | 73 % |

A few metres is expected and correct: the truck drives in a lane, the map
stores the centreline of the road, and half a carriageway is 3 to 5 m. The
figure to explain is the tail.

## The tail is prefabs, which is the next piece of work

The worst stretch -- 9 seconds of driving, up to 62.8 m from any road -- is
not a coordinate problem and not a missing sector. Nine sectors were extracted
to rule that out; the answer did not move by a centimetre.

The truck was inside a **prefab**: `dlc_it_124`, two nodes, 138 m across,
centred at (−6534, 29048), which is where the truck was. Prefabs are the
junctions, roundabouts, service areas and company yards, and the map file
stores only their position and which model they are. The geometry inside them
-- including the paths a vehicle is supposed to take through them -- lives in
`.ppd` descriptor files in `base.scs`.

So the road network extracted so far is the network *between* junctions, and
every junction is a hole in it. That is the next step, and it is not optional:
a route that cannot cross a roundabout is not a route.

## The junctions, which were the hole

A prefab's interior is in a `.ppd` descriptor in `base.scs`, reached through
`/def/world/prefab.sii`, which maps the model token the map stores
(`dlc_it_124`) to a path (`/prefab2/gas_station/gas_small_road_01_it.ppd`).
2054 prefab models are defined; every one of the 268 prefabs in this sector
resolved.

Joining the two halves needs the transform from descriptor coordinates to
world coordinates, and the map does not state it. What the map states is where
each of the prefab's control nodes ended up, and those are the same nodes the
descriptor lists, so the transform is whatever maps one set onto the other.
Fitting it -- a yaw and a translation, least squares in the ground plane --
turned out to be both simpler and safer than reproducing the game's placement
rules, because the residual is then a check rather than a hope:

| | |
|---|---|
| median residual | **0.002 m** |
| 90th percentile | 0.002 m |
| worst of 268 | 0.029 m |

Two millimetres. The descriptor lands where the map put it.

### What is inside

4321 lanes and 300 semaphores in this one sector. Each lane carries more than
its shape:

- **Connectivity** -- which lanes it leads to and comes from, as indices. This
  is the routing graph across the junction.
- **Semaphore id** -- which traffic light governs it, or −1. 308 of the 4321
  lanes are governed by one.
- **Traffic rule** -- `cross_end`, `lane_join`, `lane_split`, `limit_30`, and
  others.
- **Blinker** -- 508 lanes say indicate left, 412 say right. The game states
  which turn signal a lane expects, so that is one behaviour that will not need
  to be inferred.

The lane geometry is a cubic Hermite between two oriented points. The tangent
direction was measured rather than assumed: on the straight lanes of a prefab
the chord *is* the tangent, and rotating (0, 0, −1) by the stored rotation
agrees with the chord to a mean dot product of 0.975 while every other axis
disagrees.

### The alignment, again, with the junctions filled in

Same measurement as above, with prefab lanes included:

| | roads only | with prefab lanes |
|---|---|---|
| median | 6.04 m | **1.23 m** |
| 90th percentile | 19.2 m | 6.30 m |
| worst | 62.8 m | **7.84 m** |
| within 8 m | 73 % | **100 %** |

The 62.8 m hole was the gas station. Nothing on the taught loop is now further
than 7.84 m from geometry the game itself uses, and the median of 1.23 m is
what it should be: prefab lanes are lane centrelines, and the truck drives in
a lane.

Eighteen semaphores stand within 40 m of the loop, the nearest 3.2 m from where
the truck drove. It has been driving past traffic lights all along, and now
their positions are known.

## Lanes, measured rather than assumed

A road on the map is one curve and a token. How many lanes that is comes from
the road look, in `/def/world/road_look.template.sii` -- 395 of them -- which
counts the lanes on each side and gives the gap between carriageways. What it
does not give is a width: that is baked into the road template's model file,
and nothing readable states it.

It did not have to be guessed. Every road ends at a junction, and a junction's
lanes are exact geometry, so the lateral offsets of the lanes arriving at a
road's end node *are* that road's lane offsets. Surveyed across the sector:

| Road look | Definition | Measured offsets |
|---|---|---|
| `template46` | 1 + 1 | −2.25, +2.25 |
| `template44` | 4 | −6.75, −2.25, +2.25, +6.75 |
| `template60` | 2 + 2, gap 1 | −7.75, −3.25, +3.25, +7.75 |
| `template41` | 3 + 3, gap 2 | −13.25, −8.75, −4.25, +4.25, +8.75, +13.25 |
| `template45` | 1 + 1, gap 6.5 | −4.25, +4.25 |
| `template48` | 1 + 1 | −1.27, +1.25 |

The implied lane width is 4.5 m almost everywhere, which is the number a guess
would have used -- but `template48`, the narrow service roads, is 2.5 m, and a
guess would have put the truck 1 m out of its lane there.

The count stays the definition's to state and only the placement is measured.
Two looks came back with one cluster too many, both at a junction that
contributes its own turning lane at the same node, 10.75 m out; keeping the
strongest N clusters, where N is the definition's count, resolves both and
leaves no mismatches in the sector.

### Checked against the drive

For every recorded position matched to a road, the distance from the truck to
the nearest lane centre:

| | |
|---|---|
| median | **0.77 m** |
| 90th percentile | 1.57 m |
| worst | 4.30 m |

A hand-driven truck wanders about a metre, so a median of 0.77 m says the lane
centres are where the game thinks they are.

## Which way the traffic runs

Offsets were not enough. The first lane graph pointed every lane by the side
of the centreline it sits on -- right of the road's direction runs forward,
left runs back -- and 266 of 1763 road lanes led nowhere. 130 of those ended
exactly where another lane also *ended*: two arrows pointing into the same
point, which is what a wrongly-pointed lane looks like.

The junctions settle it the same way they settled the offsets. A junction lane
that *starts* at a road's far end is taking traffic off that road, so the road
lane at that offset runs forward; one that *ends* there is feeding the road
from the other side. Surveyed per look and offset:

| Road look | Definition | Measured |
|---|---|---|
| `template46` | 1 + 1 | −2.25 → , +2.25 ← |
| `template41` | 3 + 3 | three →, three ← |
| `template40` | **0 + 3** | −2.25 →, +2.25 →, +6.75 → |
| `template44` | **0 + 4** | all four → |
| `template42` | **0 + 1** | +2.25 → |

Every look with no lanes on its left is **one-way**, with lanes on both sides
of its centreline all running with the road. Six of the fifteen looks surveyed
here are like that, and the side rule got every one of them wrong.

Junctions agree with each other on a lane's direction 98.6% of the time on
average. The exceptions are the narrow service roads, `template48` and
`template55`, at about 90%, and the outer lane of `template44` at 86%. A
majority that clear is used as the direction, and the agreement is kept in the
cache so a planner can distrust the low ones.

## The lane graph

`core/map/graph.py` turns an extraction into directed lanes and joins them:
road lanes by offsetting the centreline and pointing each one as measured,
junction lanes as the descriptor states them, connected within a junction by
its own `next` lists and between items wherever one lane ends at another's
start (0.5 m, which is slack rather than tolerance -- the geometry agrees to
millimetres).

For sector (−2, +7): **6084 lanes, 6627 connections.**

Dead ends among road lanes went from 266 with the side rule to 138 with
measured directions. Of those 138, 32 stop at the sector boundary, where the
road continues in a sector that was not extracted, and 104 are on roads the
game hides from its map -- the service roads inside yards and depots.

On the roads that appear on the map:

| | |
|---|---|
| road lanes | 817 |
| dead ends | 24 |
| of which at the sector boundary | 22 |
| **dead ends inside the sector** | **2** |

In this one sector, then, very few lanes lead nowhere. That turned out to be a
much weaker statement than "the network is connected", which is what it was
first written up as -- see *Europe* below, where it is not.

A* over lane lengths routes across it. From the lane under the start of the
taught loop to the lane under its end: 45 lanes, 915 m -- a different way round
than the one driven, which is what a shortest path is for.

## Europe

Everything above was one sector. All 282 extract in a single run: 63 s,
3.4 GB at peak, 240 MB of JSON -- 1 309 236 items, 74 694 roads, 20 593
prefabs. Doing the whole map at once is what made the lane survey global, and
it found four things that one sector could not.

### Definitions live in more than one file

`/def/world` has eleven prefab definition files, not one, and the game loads
every `name.suffix.sii` beside the base file. The "universal roads", used all
over the map, are defined only in `prefab.mod_universal_roads.sii` and
`road_look.template.mod_universal_roads.sii`. Reading the obvious file alone
lost **3340 of the 20 593 prefabs** and left **2874 visible roads** with no
lanes. With every file read: no prefab unresolved, and 186 of 38 892 visible
roads without lanes -- looks no junction measured, left as holes.

### A node index is only an index to someone who knows the rotation

The extractor worked out which control node a junction lane attaches to, and
handed Python the descriptor's node index. Python used it as an index into the
map item's node list -- which is rotated so that it starts at the prefab's
origin. Wherever the origin was not node 0, the lane was attached to the wrong
node. The extractor now resolves the map node and exports its uid, and the
extraction format went to version 3 so that nothing reads an old file the old
way. That guard caught the very next run, which used a stale binary.

### Britain drives on the other side, and the survey had pooled it

Two motorway looks came back with their lanes' directions barely better than a
coin: `look1` at 54 % agreement, `look8` at 77 % and *backwards*. Both are
used in Britain and on the continent, and left-hand traffic runs every lane
the other way round, so the survey had been pooling two opposite answers.
Surveyed separately by the road's `left_hand_traffic` flag, both come back at
100 %, as mirror images of each other. Direction agreement across all looks
went from 96.8 % to 98.4 % on average.

### Where two looks meet, lanes step sideways

Where a road's look changes -- a lane added, a lane dropped, a different gap
between carriageways -- the lanes of one road do not line up with the next.
1562 visible road lanes across Europe ended between 1 and 6 m from where they
continued. The map node the two roads share says they continue, so lanes left
with nowhere to go are joined through it, to the nearest lane leaving the same
node in the same direction.

### What that did to the dead ends

Visible road lanes with nowhere to go, as each fix went in:

| | dead ends |
|---|---|
| definitions from every file | 2209 (2.50 %) |
| + joining through shared nodes | 1478 (1.67 %) |
| + node uids instead of indices | 1181 (1.34 %) |
| + surveying left-hand traffic apart | **793 (0.90 %)** |

## The cache

`LaneGraph` holds one Python object per lane. For Europe it builds in 22 s and
takes 1.9 GB, and it would do that on every start. So it is built once,
offline, by `tools/build_map.py`, and compiled into `RoadNetwork`: flat
arrays in `.npy` files, memory-mapped on open, lanes and successors as
compressed sparse rows, and a 16 m grid over every lane point.

| | object graph | cache |
|---|---|---|
| start-up | 22 s | **2.9 ms** |
| memory | 1.9 GB | 236 MB after 2000 queries all over Europe |
| nearest lane | 0.2 ms | 0.14 ms |
| on disk | 240 MB JSON | 247 MB |

430 157 lanes, 456 972 connections, 10 449 067 points. Built in 31 s.

Checked against the object graph it was compiled from: successor lists
identical for every lane; points within 3.8 mm, which is float32 at thirty
kilometres from the origin; 30 of 30 random routes the same length.

The nearest-lane query was not the same at first. It chose the lane by its
nearest *point* and only then measured to the line, and in 3 of 2000 queries a
lane whose points were nearer was not the lane whose line was. It now measures
to the line of every lane with a point within one spacing of the nearest, and
agrees with a brute-force search in 2000 of 2000.

## The network is not one network

With the cache it was cheap to ask the question that matters for routing: can
every lane be reached from every other? Strongly connected components over all
430 157 lanes, and for each visible road lane, whether it is in the largest one:

| Visible road lanes | Continent | Britain |
|---|---|---|
| in the core: anywhere and back | **50.6 %** | 0 % |
| can be entered from the core, cannot get back | 14.0 % | 0 % |
| can get to the core, cannot be reached from it | 13.5 % | 0 % |
| connected to the core neither way | 21.8 % | 100 % |

Britain is separate because nothing connects it: the tunnel and the ferries
are not roads. It also falls apart into several pieces of its own, the
largest 5686, 5066 and 4936 lanes, which is not explained yet.

The two middle rows are the map rather than the extraction. Of the dead ends
those regions drain into, **2352 are roads that simply end** -- nothing else at
their last node -- and 265 end at a prefab with no lanes in it, a barrier or a
closed road. The only way out is the oncoming lane of the same road, which is
a U-turn, and a lane graph that offered it would be routing a truck into
turning round on roads that stop. So a route has to stay in the core, and a
destination outside it is a real problem the planner will meet.

The last row, a fifth of the continent's visible roads connected to nothing
either way, was not explained when this was first written. Most of it now is;
see below.

### Which DLCs this is the map of

Asked afterwards, and not something the extraction had considered at all. The
answer turned out to be: all of them.

`base_map.scs` holds part of every map DLC's roads on every installation,
whoever owns what: Italy, the Baltic, the Black Sea, the Balkans and Greece
all have roads in it. This one has three DLC archives, Scandinavia, Krone and
Feldbinder.

**This section first said base_map.scs held *all* of every DLC's roads, and
that `dlc_north.scs` had a single map sector. Both were wrong.** The second
came from reading the first five lines of a directory listing. `dlc_north.scs`
has 68 sectors, none of them in base_map.scs -- the Scandinavian peninsula --
with 245 147 items and 322 496 nodes, plus its own definitions and prefab
descriptors. A map DLC ships map. See *What does* below for what reading it
changed.

What a player can drive is decided per item by a *DLC guard*: 0 for the base
game, otherwise a number for one DLC or for two together. 64 580 roads and
16 681 prefabs carry guard 0; about 9 000 roads and 3 700 prefabs carry one of
24 others.

The game states nowhere what the numbers mean. The TruckLib author's map-docs
wiki has a table, and it was checked against the map rather than taken on
trust: for every guard, the countries its roads are in and the prefab families
it uses.

| Guard | map-docs says | Its roads are in | Its prefabs are |
|---|---|---|---|
| 4 | Italy | Italy 866 | `dlc_it` 388 |
| 6 | Baltic | Russia 1411, Finland 122, Lithuania 82 | `dlc_balt` 753 |
| 11 | Iberia | Spain 162 | Iberian families |
| 16 | West Balkans | Slovenia, Serbia, Croatia, N. Macedonia | `dlc_blkw` 500 |
| 21 | Greece and East Balkans | Greece 159, Turkey 129, Bulgaria 68 | Balkan, Greek |
| 23 | Nordic | Finland 190, Sweden 157, Norway 60 | universal |

and the same for the other eighteen: all 24 guards in the map agree. Two of
them are content nobody can own. **Guard 13 is Heart of Russia**, shelved
before release, and 14 is Russia and the Baltic together; 853 roads that are
in the data and on no player's map.

The first guess about guards was that where the game swaps one piece of road
for another by ownership, the extraction had both. **It does not**: of 600
guarded roads sampled, none lies over another road, and neither does any of
400 base roads as a control. A guard adds a road; not owning the DLC removes
it, and the game closes the gap with a barrier model -- `base.scs` has
`/dlc/dead_end_dlc.pmd` and `direction_blocker_dlc.pmd` for it.

So the cache is now built for someone. `core.map.dlc` holds the table, and
`tools/build_map.py --dlc` chooses:

| Profile | For | Lanes | Continental visible road lanes in the core |
|---|---|---|---|
| `all` | a cache someone else will use: every released DLC | 423 339 | 51.3 % |
| `installed` | this installation: base, Scandinavia, Krone, Feldbinder | 364 117 | 60.4 % |

Leaving out the shelved Russia content moved the core from 50.6 % to 51.3 %;
it is not what fragments the network.

### What does

Split the island lanes by guard, in the `all` map as it first was:

| Visible road lanes | connected to the core neither way |
|---|---|
| base game, guard 0 (63 892) | 9.0 % |
| DLC, guard > 0 (14 428) | 73.3 % |

Three quarters of the DLC roads were islands with every DLC included, so
something the extraction did not see or did not join. Four things, found in
turn by looking at the boundary between the core and what it could not reach.

**There were no lane changes.** On a motorway in Hungary, island lanes sat
exactly 4.5 m from core lanes of the same road: one lane of a carriageway
connected, the one beside it not. Every lane was its own chain, so a lane that
the junction behind it does not feed has no way in, and joining ends -- by
geometry or through the node -- only ever gives a lane somewhere to go, never
a way in. Lanes alongside each other running the same way can now be changed
between, kept as a separate kind of connection with a 50 m price on the route.
Base-game islands went from **9.0 % to 0.1 %**; the continental core from
51.3 % to 78.3 %, Germany to 91 %, Austria 94 %, Switzerland 95 %.

**A map DLC ships map.** Sweden, Norway and Finland were 100 % islands. The
68 sectors of `dlc_north.scs` were not being read; the extractor now reads
base_map.scs and every installed DLC archive that has a `/map/europe`, with
their definitions and prefab descriptors, the way the game mounts them. All
25 183 prefabs resolve, Sweden is 97 % core and Norway 96.5 %.

**Some links are in archives that are not here.** 1868 node references in
base_map.scs point at items it does not contain. Adding dlc_north.scs
resolved 77. The rest sit in the DLC countries -- Russia, Italy, France,
Hungary, Romania -- and in nodes with no country; they are the pieces of road
that each DLC's own archive supplies, and they are what still separates Serbia,
Greece, Finland and most of Hungary from the core in the `all` profile. They
cannot be fixed on an installation without those DLCs, which is why the cache
manifest now records the archives the extraction read: an `all` cache built
here is base_map.scs plus Scandinavia, and a real one has to be extracted
where the DLCs are.

**Britain keeps left, and some of its looks were measured nowhere.** Where a
look was measured only where traffic keeps right, the left-hand layout is the
mirror image -- not assumed: of ten looks measured both ways, all ten are
exact mirrors, the worst lane 0.4 cm out. That filled 36 British roads. It
barely moved Britain's own connectivity, though, and what does is at the
Scottish border: 6 of the 20 roads crossing it use `look4` and `look2`, which
no junction anywhere measures, so they have no lanes and Scotland is cut off.
Two ways to fill them were tested against looks that are measured, and both
rejected: borrowing the layout of a look with the same definition is right for
75 % of two-way looks, and deriving positions from the definition's lane count
and offset fails outright, because lane widths vary from 1.5 m to 4.5 m and the
definition does not say which. They are left as holes.

### Where it stands

`tools/map_health.py` now does this measurement on any cache. Seeded on the
taught loop in Italy:

| Visible road lanes | core | enter only | leave only | island |
|---|---|---|---|---|
| this installation (`installed`) | **82.7 %** | 3.5 % | 3.4 % | 10.4 % |
| everything read here (`all`) | 74.9 % | 4.3 % | 4.1 % | 16.6 % |

The islands in `installed` are Britain: 9109 lanes, 10.3 % of the whole,
separate because neither the tunnel nor the ferries are roads, and split
inside at the Scottish border. The continent is otherwise joined up; the
enter-only and leave-only lanes are roads that end, and roads into DLC regions
this installation does not have.

## Matching a position to a lane is not the same as knowing where you are

The obvious test of the graph is to lay the taught drive over it: at every few
metres, take the nearest lane pointing the way the truck was going, and check
that each lane in the resulting sequence leads to the next.

The distances are good -- 0.84 m median to the matched lane, 1.54 m at the
90th percentile. The sequence is not: of 89 transitions, **32 are between
lanes the graph does not connect**, and 28 of those are inside junctions.

That is not the graph failing. Inside a junction several lanes run close
together -- through, turning, merging -- and the nearest one changes from
sample to sample. Taking it independently each time jumps between paths that
are 11 to 26 m apart along the route, although never more than a couple of
metres apart in space. A position alone does not say which of those the truck
is on; what it was on a moment ago does.

So the next piece is a real map matcher: one that only considers where the
current lane can lead, and holds a hypothesis through a junction rather than
re-deciding at every sample.

## Notes

- `Road.Length` is the arc length of the Hermite curve between its two nodes,
  and TruckLib will interpolate it, so the Python side gets sampled points at
  a fixed 2 m spacing rather than a spline it would have to reimplement.
- Roads carry flags worth keeping: `ai_vehicles` (whether traffic uses it),
  `left_hand_traffic`, and whether it is hidden from the UI map. The hidden
  ones near the taught loop are service roads.
- The road's lane count is not on the road. It is in the *road look*, a token
  like `template46`, defined in `/def/world/road_look.sii`. Not read yet.
- Reproducing this:

  ```bash
  source scripts/dev-env.sh    # puts dotnet on PATH
  cd tools/map_extract
  dotnet run -- roads "$ETS2/base_map.scs" "$ETS2" out.json -2,7   # one sector
  cd ../..
  python tools/build_map.py out.json data/map/europe               # the cache
  ```
