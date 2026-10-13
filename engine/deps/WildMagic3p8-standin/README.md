# Stand-in Wild Magic tree: the files we changed

faf-re builds against Geometric Tools' Wild Magic 3.8, which is no longer obtainable (upstream vendors a private
`WildMagic3p8.7z`). Our build uses a Wild Magic 3.0 tree with FAF's patches at `C:\faf\_deps\WildMagic3p8`
(not in any repository). Upstream faf-re 8977bb9d (2026-10) needs two 3.8 things the 3.0 tree lacked; the files
here are the changed copies, to be dropped into the same places of that tree (`Foundation\Intersection\`,
`Foundation\`) if it is ever rebuilt:

- `Wm3IntrLine3Box3.{h,cpp}`: `DoClipping` with a `bool bSolid` (3.8 semantics: a hollow box intersects a linear
  component only where it crosses the surface, which is what the 3.0 body already did; solid adds the inside
  case). The old signature delegates with `bSolid=false`.
- `Wm3IntrSegment3Box3.{h,cpp}`: the three-argument constructor `(segment, box, bSolid)`, which
  `CColPrimitive<Box3f>::CollideLine` uses with `false` (the shipped exe's `0x00A463A0`).
- `Foundation.vcxproj`: `Containment\Wm3ContConvexHull2.cpp` and `Wm3ContConvexHull3.cpp` compiled (3.0's
  `ContBox2/3` need them; upstream's `Wm3::InBox` pulls `ContBox3` in).
