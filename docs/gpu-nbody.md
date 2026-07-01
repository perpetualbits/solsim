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
| 5b | node mass/COM/box | aggregate up the tree |
| 6 | traverse (forces) | the actual gravity on every body |
| 7 | resident + integrate | keep it all on the GPU, leapfrog step |

All phases are implemented and validated. The galaxy collision now steps entirely on
the GPU (`GpuNBody`), with only the positions copied back each frame to draw.

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

## 6. Phase 5b — filling the tree with mass and centre of mass

The structure alone is just pointers. Barnes–Hut needs, for every node, its **total
mass** and its **centre of mass** (COM) — so a faraway clump can be treated as one
body. A leaf's are just its particle's; an internal node's are the mass-weighted
combination of its two children:

```
M = m_left + m_right
C = (m_left · c_left + m_right · c_right) / M
```

This must be done **bottom-up**: a node can only be combined once both its children
are known. On a CPU that's a post-order walk. On the GPU we want all nodes in a level
done at once — and here is where a real GPU subtlety bites.

### The trap: the one-pass atomic walk-up

The textbook GPU method (Karras) launches **one thread per leaf**, each climbing the
parent pointers. At each internal node a per-node atomic counter decides who does the
work: the *first* child to arrive increments it and stops (its sibling isn't ready);
the *second* child sees the count is already 1, so it knows both subtrees are done,
combines them, and climbs on. Elegant — one dispatch, no barriers.

We built exactly that first, and the test caught it red-handed: the root's total mass
came out **3679 instead of 4000** — about 8 % of the mass simply vanished. The reason
is a genuine hardware fact, not a coding slip. When the second-child thread reads its
sibling's mass, that sibling was written by a *different workgroup*. WGSL's memory
model does **not** guarantee that one workgroup's ordinary storage writes are visible
to another just because you passed an atomic between them — there are no acquire/
release atomics in WGSL. So the reader sometimes gets a stale zero, and that subtree's
mass is lost. It happens to "work" on some drivers, which makes it a nasty trap.

### The fix: refit one level per pass

Instead we lift the "finished" frontier **one level per dispatch** and let the *pass
boundary* provide the visibility — the same barrier the bitonic sort relies on
(§4). The kernel (`AGGREGATE_SHADER`, `aggregate_nodes_gpu`) is one thread per
internal node:

- Leaves start marked **finished**, internal nodes not.
- Each pass, an internal node whose **both children were finished in an earlier pass**
  computes its `M` and `C` and marks itself finished; others do nothing yet.
- The "finished" flags are **double-buffered** (`done_in` read, `done_out` written,
  swapped every pass). Reading last pass's flags means a node is never built from a
  child that only became ready in the *same* pass — so every child's mass/COM write
  lives in a strictly earlier submit, and is therefore visible. No atomics needed.
- We keep dispatching until the **root reports finished** (we read back its one flag
  after each pass). That takes exactly the tree's *height* passes — tens, not
  thousands.

### Trusting it

`gpu_node_aggregates_match_cpu` builds a real tree from 4000 random points, hands the
GPU random leaf masses, and recomputes every node on the CPU by walking the read-back
structure in **reverse-preorder** (children before parents). It checks all `2n-1`
nodes' mass and COM against that refit, and separately that the **root mass equals the
plain sum of all leaf masses** and its COM is the mass-weighted mean — the property
that must hold if nothing leaked. (This is the test that failed loudly on the atomic
version, which is exactly why we trust the level version.)

## 7. Phase 2 — the bounding box (a parallel reduction)

The Morton step (§3) needs the box `[lo, hi]` that contains every particle, to
normalise positions into the unit cube. Until now the CPU computed it; this phase
does it on the GPU so the whole build can stay on the card.

Finding the min and max over an array is a **reduction** — the same shape as summing
an array, but with `min`/`max` instead of `+`. Both are associative and commutative,
so we may combine elements in *any* order, which is exactly what lets a GPU do it in
parallel (`BBOX_SHADER`, `bounding_box_gpu`).

We use a **single workgroup of 256 threads** in two steps:

1. **Grid-stride fold.** Thread `t` walks the array in steps of 256 (elements `t`,
   `t+256`, `t+512`, …), keeping a private running `(min, max)`. This is what lets
   `n` be far bigger than 256 — each thread just folds more elements. Threads that
   run past the end start from `±f32::MAX`, the identity for min/max, so they
   contribute nothing.
2. **Halving tree in shared memory.** The 256 private results are written to
   workgroup-shared scratch, then combined pairwise: 256→128→64→…→1, halving the
   live lanes each round, until thread 0 holds the box, which it writes out.

Here a plain **`workgroupBarrier()`** between rounds is enough — unlike the tree
refit in §6, everything happens inside *one* workgroup, and workgroup-shared memory
with that barrier *is* covered by the memory model. That contrast is the whole point:
`workgroupBarrier` synchronises threads within a workgroup; crossing workgroups needs
a submit/pass boundary. Choosing the right one for the job is most of what makes GPU
code correct.

For our `n` (tens of thousands to a few million), one workgroup striding the array is
already fast; a huge array would instead use many workgroups writing partial boxes
and a second tiny pass to combine them.

`gpu_bounding_box_matches_cpu` checks the result against a plain CPU min/max over
12 345 random points (not a multiple of 256, so the strided tail is exercised). The
box corners are exact copies of input floats, so the comparison is exact.

## 8. Phase 6 — walking the tree for the forces

Everything so far was scaffolding; this is the gravity. One thread per particle
walks the finished tree and sums the pull, lumping distant clumps and opening near
ones — the same Barnes–Hut idea as the CPU octree (`TRAVERSE_SHADER`,
`accelerations_gpu`).

**One extra thing the nodes needed: a size.** The opening test asks "is this node
small compared with its distance?", so each node needs a spatial size. We already
had a bottom-up refit (§6), so we taught it to also carry each node's **axis-aligned
box** (min/max corner): a leaf's box is its point, an internal node's is the union of
its children's. The node's size is then the box **diagonal**. (We first tried the
longest side; it lumps a hair too eagerly and the error sat just over the bar. The
diagonal is a little more conservative — it opens slightly more — and lands the
accuracy right where the CPU octree is. Choosing the size measure *is* choosing the
accuracy.)

**The walk, with a private stack.** A GPU thread can't recurse, so each carries a
small fixed stack (64 entries — plenty, since a tree over `N` leaves is at most about
`30 + log₂N` deep). Starting from the root:

```
push root
while stack not empty:
    node = pop
    r⃗ = node.com − my_position;   r² = r⃗·r⃗
    if node is a leaf:
        add its softened pull        # the leaf that is ME has r⃗ = 0 → adds nothing
    else if node.size² < θ²·r²:      # far enough: treat the whole node as one body
        add its softened pull
    else:
        push node.left, node.right   # too close: open it
```

Each pull is the same softened law as everywhere else,
`a⃗ = G·m·r⃗ / (|r⃗|² + ε²)^{3/2}`. A neat consequence of softening: we don't special-
case "don't pull on yourself" — the leaf holding this very particle has `r⃗ = 0`, so
`m·r⃗ = 0` and it contributes nothing on its own.

**Order in, order out.** The particles were permuted into Morton order to build the
tree, so the walk produces accelerations in *that* order; `accelerations_gpu` scatters
them back to the caller's original indices at the end.

### Trusting it

`gpu_accelerations_match_direct_sum` runs the *entire* pipeline on 3000 random bodies
and compares every acceleration against the exact O(N²) softened sum — the same
ground truth (and the same tolerances, θ = 0.5) the CPU octree is held to. The GPU
tree is a *binary* LBVH, not the CPU's octree, so the two differ node-for-node; that
they both land within Barnes–Hut error of the direct sum is exactly the point. Mean
relative error comes out well under 1 %.

## 9. Making it resident — the whole step on the GPU

The reference `accelerations_gpu` reads each stage back to the CPU and re-uploads it:
perfect for testing, but every readback stalls the CPU, so it would be *slower* than
the CPU octree in real time. The live path (`GpuNBody`) instead keeps **everything
resident** and does a full leapfrog step as **one command submission**.

**State that never leaves the card.** Positions, velocities, accelerations and masses
live in GPU buffers, as does all the tree scratch (codes, the sort arrays, the child
links, the node mass/COM/box arrays). Sizes depend only on `N`, so they are allocated
once. One `step(dt)` encodes, back to back:

```
kick-1 + drift        (integrate)          v += a·½dt ;  x += v·dt
rebuild the tree      box → codes → sort → LBVH → mass/COM/box
walk it               forces → a
kick-2                (integrate)          v += a·½dt
```

Between the validated shaders sit a few tiny **glue kernels** that just move data on
the card: a Morton kernel that reads the box from a buffer (not a CPU uniform), a
sort-setup that writes the padded `(code, index)` arrays, a **gather** that reorders
particles into Morton order for the leaves, a **seed** that primes the leaves for the
refit, and a **scatter** that sends each force back to its original particle. Because
each stage is its own compute pass, the pass boundaries supply all the ordering — the
same mechanism the sort and the refit already relied on.

**Two small design choices that avoid readbacks mid-step.** The refit's convergence
depends on the tree height, which we don't know without looking; rather than read a
flag back each level, we just run a **fixed 64 passes** — comfortably more than the
`~30 + log₂N` a Morton tree can ever be deep, and extra passes are free no-ops. And
the traversal stack is a fixed 64 entries for the same reason. So the entire step is
GPU-only.

**And nothing comes back at all.** The point renderer draws **straight from the
position buffer**: a vertex shader reads `pos[instance_index]`, colours it by index
(the two galaxies split at `n_a`, bright cores at `0` and `n_a`), and even computes the
framing centre itself from `pos[0]` and `pos[n_a]` — every vertex reads those same two
cached addresses, so it is essentially free. That removes the per-frame position
readback, the CPU loop that rebuilt a point per particle, and the re-upload — which at
a million particles is the difference between a smooth frame and a stalled one.

### Trusting it

Two headless tests guard the live path. `resident_stepper_matches_cpu_leapfrog`
integrates the *same* initial conditions on the GPU and with the CPU `Particles`
leapfrog at θ = 0 (exact forces both sides) and checks the particles still coincide
after 50 steps — so the kick/drift/gather/walk/scatter plumbing is correct, not just
the forces. `resident_stepper_large_stays_finite` runs a 16 384-body cloud (a power of
two, so the sort takes the no-padding path) and checks nothing goes NaN or flies
away — a guard on the sort size, the 64-pass refit, and the traversal stack at real
scale.

## 10. Making it fast — measure, then cut

Working is not the same as fast. At 60 002 bodies the first version stepped in about
70 ms. The rule is **measure before you optimise**, so a per-stage timer
(`profile`, run on the real GPU) printed where the step went:

```
   bitonic-sort   1.6 ms
      refit(64)   1.0 ms
       traverse  68.7 ms   ← 96 % of the step
   everything else < 0.9 ms
```

So the tree *walk* is the whole game; nothing else is worth touching. And 60k bodies ×
~2000 interactions in 68 ms is only ~1.8 billion interactions/second — far below what
the chip can do — which is the signature of a **memory-bound** kernel, not a
compute-bound one. The walk was re-reading a lot per node: three `vec4`s (centre of
mass, and the two box corners) plus a separate mass array, and it recomputed the
node's size from the corners *on every visit* — even though one node is visited by
thousands of particles.

The fix is to **pack the node record for the walk**:
- the mass rides in the unused `w` lane of the centre-of-mass vector (`com.w`), so the
  separate mass array is gone;
- the size is computed **once** after the refit (a tiny one-thread-per-node kernel,
  `SIZE_SHADER`) into a `node_size2` array, instead of from the corners every visit.

Now each node touched costs one `vec4` read (plus one float if it's internal), down
from three `vec4`s and an arithmetic size calculation. Same tree, same forces, same
tests green — just less memory traffic. The result:

```
       traverse  68.7 ms  →  25.2 ms     (2.7× on the hot stage)
   full-forces   71.7 ms  →  28.7 ms
```

A 2.7× win from moving bytes, not flops — exactly what "memory-bound" predicted.

### Warp-cooperative traversal — killing the divergence

That still left the walk memory- *and divergence*-bound. Here is the divergence: a
warp is 32 threads running in lockstep on one instruction stream. When each thread
walks its *own* stack, thread A might open a node that thread B would lump — so the
warp has to execute *both* paths, masking off the threads that don't take each one.
The warp only runs as fast as the union of all 32 threads' paths.

But we sorted the particles along the Morton curve, so the 32 threads of a warp are
**neighbours in space** — they want to open almost exactly the same nodes. So let the
whole warp walk **one shared traversal**: at each node, ask `subgroupAny` whether *any*
lane still needs to open it; if so everyone opens it, otherwise everyone lumps it. That
one vote is subgroup-uniform, so every lane makes identical push/pop moves — which
means their private stacks stay identical with no shared memory, and **the control flow
never diverges**. Each lane still accumulates the pull on its *own* particle. The only
cost is that a lane occasionally descends into a node it would have lumped alone; because
warp-neighbours are spatially coherent, that waste is small and the divergence saving
dwarfs it. (Out-of-range tail lanes stay in the loop voting "don't open", so the vote
stays uniform.) It needs the GPU's subgroup feature; without it we fall back to the
per-lane walk automatically.

```
       traverse  25.2 ms  →  12.8 ms     (another 2×)
   full-forces   28.7 ms  →  18.6 ms
```

**Tally.** The tree walk went 68.7 → 25.2 → 12.8 ms — **5.4× overall** — and a full
force step 71.7 → 18.6 ms, all with the same tree and no loss of accuracy (the θ = 0
resident test still matches the CPU leapfrog exactly).

## 11. Pushing toward a million — and where the wall is

Scaling the same code up (a profiler that takes `SOLSIM_N`/`SOLSIM_THETA` runs the real
pipeline on the GPU) shows the walk staying dominant, and shows *where it stops*:

| bodies | full force step (θ = 0.8) | ≈ fps |
|-------:|--------------------------:|------:|
| 250 000 | 30 ms | ~33 |
| 500 000 | 64 ms | ~16 |
| 1 000 000 | 137 ms | ~7 |

**θ, the accuracy knob.** θ sets when a clump is "far enough" to lump. Bigger θ visits
fewer nodes: at a million bodies the walk is 166 ms at θ = 0.6, 87 ms at 0.8, 52 ms at
1.0. For a *visual* galaxy that is a fine trade, so the default is 0.8 and the `[` / `]`
keys change it live (a single uniform write — nothing rebuilds).

**Then draw-from-GPU (§9) paid off at scale.** Copying a million positions back,
rebuilding a million point sprites on the CPU, and re-uploading them every frame is
tens of milliseconds of work that simply vanishes when the vertex shader reads the
position buffer directly. At a million bodies that is the difference between a moving
picture and a slideshow, independent of the physics.

**What did _not_ work — and why that's the answer.** The walk is memory-bound, so the
tempting next idea was to raise **occupancy**: each lane carries a 64-entry stack (64
registers), which caps how many warps run at once, so we tried keeping one stack per
subgroup in shared memory and `subgroupBroadcast`-ing the current node to the lanes.
It was *slower* (92 vs 87 ms). The lesson is the diagnosis itself: a memory-bound
kernel is not helped by freeing registers — the broadcast and shared-memory traffic
just added work while the bottleneck (bytes pulled through the cache per node) stayed
put. We had already squeezed the bytes (§10), and the masses are too small to survive
`f16`, so the walk is genuinely close to this laptop GPU's bandwidth floor.

**So the honest ceiling on this hardware (an RTX A500 laptop GPU):** ~250–300k bodies
at 60 fps, ~500k around 30, and a million at roughly 10 fps (θ = 1.0) — bandwidth,
not cleverness, is the limit there. The remaining compute win is a radix sort (the
bitonic sort is ~30 ms at a million; an O(N) sort would be a few), which most helps the
250k–500k range where the framerate is already comfortable.

---

Every phase landed as its own kernel with its own CPU-reference test, so the whole GPU
pipeline — box, codes, sort, tree, masses, forces, and now a resident leapfrog — is
trustworthy end to end, and the galaxy collision runs entirely on the card.
