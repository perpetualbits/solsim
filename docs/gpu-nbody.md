# Barnes–Hut on the GPU — design & implementation notes

This document explains how the galaxy mode's gravity is being moved onto the GPU,
written so you can follow the *ideas* and the *code* and eventually build one
yourself. It grows one phase at a time, alongside `src/physics/gpu.rs`. Everything
here is checked the same way the CPU code is: run the kernel, copy the result back,
and compare against a plain CPU version (see the tests in `gpu.rs`).

## 1. Why, and the plan

The CPU Barnes–Hut (`physics/octree.rs`) is fast, but every step still costs the
whole force calculation on the CPU, and the particle data is copied to the GPU only
to be drawn. The GPU version flips that around:

- **Keep the particles resident on the GPU.** Positions and velocities live in GPU
  buffers and never come back to the CPU. The renderer reads the position buffer
  directly, so there is *no per-frame copy*.
- **Do the whole step in compute shaders**: build a tree, walk it for the forces,
  integrate. A modern GPU runs thousands of these in parallel.

The tree we build on the GPU is an **LBVH** (Linear Bounding Volume Hierarchy) — a
tree derived from sorting the particles along a space-filling curve. That splits
into clean, separately-testable phases:

| Phase | Kernel | Idea |
|------:|--------|------|
| 0 | compute + readback | prove the plumbing works |
| 2 | bounding box | the cube that holds all particles |
| 3 | Morton codes | a Z-order key per particle |
| 4 | **sort** | order particles along the curve |
| 5a | LBVH structure | a tree over the sorted particles |
| 5b | node mass/COM | aggregate up the tree |
| 6 | traverse + integrate | the actual gravity + leapfrog step |

Phases 0, 3, 4 and 5a are implemented and validated so far.

## 2. A short GPU-compute crash course (wgpu)

The program talks to the GPU through **wgpu** (the same library used for drawing).
The pieces you need for compute:

- **Device + queue.** The `Device` creates GPU objects; the `Queue` runs work.
  Compute needs *no window*, so `headless_device()` grabs one with no surface —
  which is exactly why the tests can run on the GPU with no screen.
- **Buffers.** Blocks of GPU memory. Two kinds matter here:
  - **storage buffers** — big, read/write from a shader (our positions, keys, …);
  - **uniform buffers** — small, read-only, the same for every thread (parameters).
  You cannot read a GPU buffer from the CPU directly; you copy it into a special
  **`MAP_READ`** buffer and "map" that (see `map_u32`). That copy-back is how the
  tests inspect results.
- **A compute shader** (written in **WGSL**) is a function the GPU runs many times
  in parallel. It is launched as a grid of **workgroups**, each a small block of
  threads (`@workgroup_size(64)` here). Each thread learns its global index from
  `@builtin(global_invocation_id)` and typically handles one array element:
  ```wgsl
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= arrayLength(&data)) { return; }  // guard the tail
      data[i] = ...;
  }
  ```
- **Bind group + layout.** The shader names its buffers by `@group(0) @binding(k)`;
  the *layout* declares their types and a *bind group* points them at real buffers.
- **Pipeline + dispatch.** A `ComputePipeline` bundles the shader; inside a
  *compute pass* you `set_pipeline`, `set_bind_group`, and `dispatch_workgroups(g)`
  with `g = ceil(N / 64)` groups. Then `queue.submit(...)` runs it.

The smoke test `compute_dispatch_and_readback_work` is the whole loop in miniature:
upload an array, a shader doubles it, copy back, check.

## 3. Phase 3 — Morton (Z-order) codes

To build a tree we first put the particles in an order where **neighbours in space
are neighbours in the list**. The trick is a **space-filling curve**; we use the
**Z-order (Morton) curve**.

The recipe (`morton_code`, and the WGSL `MORTON_SHADER`):

1. **Normalise** the point into the unit cube using the bounding box:
   `u = (p − lo) · inv`, where `inv = 1/(hi − lo)` per axis.
2. **Quantise** each axis to 10 bits: `xi = u32(u.x · 1024)` (0..1023). Ten bits per
   axis × three axes = a 30-bit code, which fits in a `u32`.
3. **Interleave** the bits: bit 0 of the code is x's bit 0, bit 1 is y's bit 0, bit
   2 is z's bit 0, bit 3 is x's bit 1, and so on. Interleaving is what makes nearby
   points share a long common *prefix*.

The interleave is done branch-free by `expand_bits`, which spreads a value's 10 bits
out so each sits in every third position (`abc… → a··b··c…`); OR-ing the three
shifted spreads gives the code. The magic constants are the standard bit-spread
masks — worth stepping through on paper once.

**Why it is exact on both CPU and GPU:** both do the *same* float math and the same
`u32(...)` truncation, so the codes match bit-for-bit — which is what the test
`gpu_morton_matches_cpu` asserts. (Bit-for-bit matching is only reasonable because
these end in *integers*; for floating-point results we compare within a tolerance.)

## 4. Phase 4 — sorting on the GPU (a bitonic network)

Now sort the particles by their Morton code. Sorting is the classic "but how do you
even parallelise that?" problem. Two routes:

- **Radix sort** — count digits, prefix-sum the counts, scatter. Asymptotically the
  best (`O(N)`), but it needs cross-workgroup prefix sums and a *stable* scatter,
  which are fiddly and easy to get subtly wrong.
- **Bitonic sorting network** — `O(N·log²N)`, so more comparisons, but every step is
  an **independent compare-and-swap** with a fixed, data-independent pattern: no
  atomics, no prefix sums, trivially parallel, and fully deterministic. For our N
  (tens of thousands) it is plenty fast and *much* easier to get right. We use this
  one (`bitonic_sort_gpu`, `BITONIC_SHADER`).

### How a bitonic network works

The network sorts an array of size `M` (a power of two) with a fixed schedule of
compare-exchange **sub-passes**. Two loops:

```
for k = 2, 4, 8, …, M:        # block size being merged
    for j = k/2, k/4, …, 1:   # comparison stride
        for every index i in parallel:
            partner = i XOR j
            if partner > i:                 # each pair handled once, by the lower i
                ascending = (i AND k) == 0  # this block sorts up or down
                compare (keys[i],vals[i]) with (keys[partner],vals[partner]);
                swap them if they are in the wrong direction
```

Intuition: the inner `j` loop takes a *bitonic* sequence (one that goes up then
down) and merges it into a fully sorted run; the outer `k` loop builds ever-larger
bitonic sequences by sorting adjacent blocks in *opposite* directions (that is what
`(i & k)` decides). After the last sub-pass the whole array is sorted. The pattern
never depends on the data, so the same sequence of dispatches sorts anything.

Two details in the code:

- **We sort pairs `(key, val)` lexicographically** — key first, then val. `val` is
  the particle's original index, so equal Morton codes come out ordered by index.
  That gives a single, total, deterministic order, which is exactly what the LBVH
  build (phase 5) needs to break ties consistently.
- **Padding.** Bitonic networks need a power-of-two size, so we pad `N` up to
  `M = next_power_of_two(N)` with sentinel `(0xFFFFFFFF, 0xFFFFFFFF)` entries. Being
  the maximum, they sort to the very end, and we simply read back the first `N`.

### Feeding `(j, k)` to each sub-pass, in one submission

Each sub-pass is one dispatch, and each needs different `(j, k)`. You cannot change
a uniform buffer *between* dispatches that are already encoded, so instead we pack
**every** sub-pass's parameters into one uniform buffer (one 256-byte-aligned block
each) and select the current block with a **dynamic offset** when we
`set_bind_group`. All the sub-passes then go into a single command buffer and one
`submit`.

### Getting the ordering right (synchronisation)

Each sub-pass reads what the previous one wrote (same buffers), so they must run
*in order*, not overlap. We put **each dispatch in its own compute pass**; the pass
boundary makes wgpu insert the memory barrier that guarantees the previous writes
are visible before the next pass reads. (Optimised implementations do several
sub-passes inside one pass using workgroup-shared memory; we keep it simple.)

### Trusting it

`gpu_sort_matches_cpu` sorts 5000 keys drawn from a *small* range — so there are
lots of ties — and asserts the GPU result equals Rust's own `sort()` of the
`(key, index)` pairs, keys **and** order. Because both use the total `(key, index)`
order, the match is exact, ties included.

## 5. Phase 5a — building the tree in parallel (Karras 2012)

Now the beautiful part: build a whole binary tree over the sorted particles **with
no recursion and no locks**, one thread per node, each figuring out its own place.
This is Karras' LBVH construction (`LBVH_SHADER`, `build_lbvh_structure_gpu`).

**Node numbering.** For `N` particles there are `N` leaves and `N-1` internal nodes.
We give them one id space: internal nodes `0 … N-2`, leaves `N-1 … 2N-2` (leaf `k` is
id `N-1+k`). The **root is internal node 0**, and it always covers the whole array.

**The key idea — the δ function.** `δ(i, j)` is the number of leading bits the sorted
codes at positions `i` and `j` share (using `countLeadingZeros(code_i XOR code_j)`).
Because we sorted by `(code, index)`, when two codes are *equal* we fall back to the
indices — `δ = 32 + countLeadingZeros(i XOR j)` — so every pair still has a definite
ordering. A large `δ` means "these two particles are close on the curve." The whole
tree is derived from `δ` alone.

Each internal node `i` does three things (all by binary search on `δ`):

1. **Direction.** Compare `δ(i, i+1)` with `δ(i, i-1)`: the node's range extends
   toward whichever neighbour it shares more bits with (`d = ±1`).
2. **Range.** Its range is `[i, j]` for some `j` in that direction. We find `j` by
   first doubling a step until `δ` drops below `δ(i, i-d)` (an upper bound on the
   length), then binary-searching the exact length.
3. **Split.** Within `[i, j]`, the node splits where the shared-prefix length first
   drops — again a binary search, giving position `γ`. The left child covers
   `[first, γ]`, the right `[γ+1, last]`. A side that is a single element becomes a
   **leaf**; otherwise it is the internal node with that index.

The node writes its two children and — crucially — writes *itself* as each child's
parent (`par[child] = i`). Those parent links are what the bottom-up mass pass
(phase 5b) will climb.

**Why no synchronisation is needed here:** every internal node reads only the
(read-only) codes and writes only its own `lft`/`rgt` entry and its children's
`par` entries. Different nodes never write the same slot, so the `N-1` threads are
independent.

**Trusting it.** `gpu_lbvh_is_a_valid_tree` builds the tree from 3000 codes drawn
from a *small* range (so there are lots of duplicate codes, exercising the δ
tiebreak), reads the structure back, and checks two things on the CPU: every child
points back at its parent, and a **left-to-right walk from the root visits the
leaves `0,1,…,N-1` in order** — which is exactly the property a correct radix tree
over a sorted array must have. If any range or split were wrong, the leaf order
would break.

## 6. Still to come

- **Phase 2 — bounding box** on the GPU (a parallel reduction), so the Morton step
  no longer needs a CPU-computed box.
- **Phase 5b — node aggregates**: a bottom-up pass over the parent links that fills
  each node's total mass, centre of mass and bounding box. Leaves start from their
  particle; each internal node is finished when its *second* child arrives (tracked
  with one atomic counter per node), then it climbs to its own parent. This is the
  one place we lean on GPU atomics for ordering, so it gets careful testing against
  a CPU re-computation of the same tree.
- **Phase 6 — traverse + integrate**: one thread per particle walks the tree
  (lump-or-open using each node's mass/COM/size, just like the CPU version), then a
  leapfrog kick-drift-kick updates the resident position/velocity buffers, which the
  point renderer draws directly.

Each lands as its own kernel with its own CPU-reference test, so the whole GPU
pipeline is trustworthy end to end even though the tree lives entirely on the card.
