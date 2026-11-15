# Border crossings: what each needs

The map's customs lanes are not always the way through a border: where they
lead over islands or into booths, a driver's line is taught
(`tools/teach_crossing.py`, `configs/crossings.json`). Where the truck has
gone through on its own, both ways, nothing is needed.

These are the seven border crossings the game's map shows on this machine's
map (base map and Scandinavia); other DLCs bring others. Positions are world
(x, z) near the crossing. Directions are by the way the
truck heads, x east and z south.

| Border | Near | One way | The other way |
|---|---|---|---|
| Chiasso, Italy - Switzerland | (-7500, 26600) | taught: `border-it-ch-01` (north-east) | taught: `border-chiasso-ch-it-1` (south-west) |
| Basel, Switzerland - France | (-11900, 16620) | taught: `border-basel-5` (east) | taught: `border-fr-douane-2` (north-west, into France) |
| Geneva area, first | (-16250, 24360) | taught: `border-both-2` (south-east) | taught: `border-both-3` (north-west) |
| Geneva area, second | (-19050, 24870) | taught: `border-fr-ch-1` (north) | taught: `border-ch-fr-1` (south) |
| North of Zurich, Germany - Switzerland | (-4600, 18560) | nothing needed: driven through 25 Oct 2026 | nothing needed: driven through 25 Oct 2026 |
| North of Basel, Germany - Switzerland | (-7365, 16130) | taught: `border-de-ch-basel-1` (south) | taught: `border-de-ch-basel-4` (north) |
| Svinesund, Norway - Sweden | (-2300, -23880) | taught: `border-svinesund-1` (south-east) | nothing needed: no customs that way, the road goes straight on |
