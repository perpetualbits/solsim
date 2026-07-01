//! A Barnes–Hut octree for fast gravity on many bodies (the galaxy mode).
//!
//! The honest way to get the gravity on `N` bodies is to add up the pull of every
//! other body — that is `N × N` sums, hopeless once `N` is large. Barnes–Hut is the
//! famous shortcut: build a tree that splits space into ever-smaller cubes (an
//! **octree** — each cube has up to eight children), and remember each cube's total
//! mass and centre of mass. Then, to pull on one body, a clump of far-away bodies
//! is replaced by a **single lump** at their centre of mass whenever the clump is
//! small compared with its distance (`size / distance < θ`). Nearby clumps are
//! opened up and looked at in detail. That turns the cost from `N²` into about
//! `N·log N`.
//!
//! For speed the tree is **flattened**: after building it as a normal pointer tree,
//! we lay the cells out in one array in depth-first order and give each an index to
//! "skip to" if it is accepted as a lump. Walking gravity is then a tight iterative
//! loop over a compact array — either step to the next cell (open it) or jump past
//! its whole subtree (lump it) — which the CPU cache loves. It is far faster than
//! chasing pointers, and the per-body walks are independent, so they run in
//! parallel (see [`Octree::accelerations`]).
//!
//! Everything is in `f64`. Units are left to the caller (pass your own `G`); the
//! forces are softened (a small `ε`) so two bodies that come very close do not fly
//! apart from a division by nearly zero — the right thing for a *smooth* galaxy
//! made of sample particles. Validated by its tests against the direct O(N²) sum.
#![allow(dead_code)]

use glam::DVec3;

/// Deepest the tree may subdivide before it just buckets bodies together.
///
/// What: a safety cap on tree depth.
/// How/why: two bodies at (almost) the same point would otherwise split forever;
/// past this depth we keep them in one leaf. 64 halvings is far finer than any real
/// separation, so it never bites in practice.
/// Units: a count of levels.
const MAX_DEPTH: u32 = 64;

// ---------------------------------------------------------------------------
// Build-time pointer tree
// ---------------------------------------------------------------------------

/// One cube in the build-time pointer tree.
///
/// What: a cell's geometry, the mass and centre of mass inside it, and links to its
/// up-to-eight children (or the body it holds, if a leaf).
/// How/why: this form is convenient to *build* by inserting bodies one at a time;
/// it is then flattened into the fast [`FlatNode`] array and discarded.
/// Units: caller's own.
struct Node {
    center: DVec3,
    half: f64,
    com: DVec3,
    mass: f64,
    children: [i32; 8],
    body: i32,
    extra: Vec<u32>,
    internal: bool,
}

impl Node {
    fn empty(center: DVec3, half: f64) -> Self {
        Node {
            center,
            half,
            com: DVec3::ZERO,
            mass: 0.0,
            children: [-1; 8],
            body: -1,
            extra: Vec::new(),
            internal: false,
        }
    }
}

/// Builds the pointer tree by inserting bodies, then hands over its cells.
struct Builder {
    nodes: Vec<Node>,
}

impl Builder {
    /// Insert body `body` into cell `node` at tree depth `depth`.
    ///
    /// What: adds one body, splitting the cell if it was already occupied.
    /// How/why: fold the body into the cell's running mass and centre of mass, then
    /// place it — an empty cell stores it, an occupied leaf subdivides and pushes
    /// both bodies down, an internal cell forwards it to the right child. At the
    /// depth cap coincident bodies just share the leaf.
    /// Units: caller's own.
    fn insert(&mut self, node: usize, body: usize, depth: u32, pos: &[DVec3], mass: &[f64]) {
        {
            let n = &mut self.nodes[node];
            let bm = mass[body];
            let total = n.mass + bm;
            n.com = if n.mass == 0.0 {
                pos[body]
            } else {
                (n.com * n.mass + pos[body] * bm) / total
            };
            n.mass = total;
        }

        if self.nodes[node].internal {
            self.insert_into_child(node, body, depth, pos, mass);
            return;
        }
        if self.nodes[node].body < 0 {
            self.nodes[node].body = body as i32;
            return;
        }
        if depth >= MAX_DEPTH {
            self.nodes[node].extra.push(body as u32);
            return;
        }
        let existing = self.nodes[node].body as usize;
        self.nodes[node].body = -1;
        self.nodes[node].internal = true;
        self.insert_into_child(node, existing, depth, pos, mass);
        self.insert_into_child(node, body, depth, pos, mass);
    }

    /// Forward a body into the correct child cube, creating it if new.
    fn insert_into_child(&mut self, parent: usize, body: usize, depth: u32, pos: &[DVec3], mass: &[f64]) {
        let (center, half) = {
            let n = &self.nodes[parent];
            (n.center, n.half)
        };
        let oct = octant(center, pos[body]);
        let mut child = self.nodes[parent].children[oct];
        if child < 0 {
            let (cc, ch) = child_box(center, half, oct);
            child = self.nodes.len() as i32;
            self.nodes.push(Node::empty(cc, ch));
            self.nodes[parent].children[oct] = child;
        }
        self.insert(child as usize, body, depth + 1, pos, mass);
    }
}

// ---------------------------------------------------------------------------
// Flattened traversal tree
// ---------------------------------------------------------------------------

/// One cell in the flattened, traversal-optimised tree.
///
/// What: a cell's centre of mass, total mass, squared size, and the flat index to
/// jump to if the cell is accepted as a single lump; for a leaf, the range of
/// bodies it holds inside [`Octree::leaf_bodies`].
/// How/why: laid out in depth-first order so "open this cell" is just the next
/// index and "lump this cell" is a jump to `skip`. `size2` is pre-squared so the
/// opening test is a bare multiply-compare. `body_count == 0` marks an internal
/// cell.
/// Units: caller's own; `size2` a length².
#[derive(Clone, Copy)]
struct FlatNode {
    com: DVec3,
    mass: f64,
    size2: f64,
    skip: u32,
    body_start: u32,
    body_count: u32,
}

/// A built Barnes–Hut octree, ready to answer gravity queries.
///
/// What: the flattened cells, the bodies grouped by leaf, and the accuracy /
/// softening settings.
/// Units: `theta2 = θ²` dimensionless; `softening2 = ε²` a length².
pub struct Octree {
    flat: Vec<FlatNode>,
    leaf_bodies: Vec<u32>,
    theta2: f64,
    softening2: f64,
}

impl Octree {
    /// Build the octree from body positions and masses.
    ///
    /// What: returns a flattened tree ready for fast queries.
    /// How/why: find the cube bounding all bodies, insert each body into a pointer
    /// tree (accumulating every cell's mass and centre of mass), then flatten it
    /// into the depth-first array walked by [`acceleration`](Self::acceleration).
    /// Units: `pos`/`mass` in the caller's units; `theta` dimensionless (smaller =
    /// more accurate, slower); `softening` a length.
    pub fn build(pos: &[DVec3], mass: &[f64], theta: f64, softening: f64) -> Octree {
        let mut tree = Octree {
            flat: Vec::new(),
            leaf_bodies: Vec::new(),
            theta2: theta * theta,
            softening2: softening * softening,
        };
        if pos.is_empty() {
            return tree;
        }
        // Bounding cube.
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let center = (lo + hi) * 0.5;
        let half = (hi - lo).max_element() * 0.5 * 1.0001 + 1e-12;

        // Build the pointer tree.
        let mut builder = Builder {
            nodes: vec![Node::empty(center, half)],
        };
        for i in 0..pos.len() {
            builder.insert(0, i, 0, pos, mass);
        }

        // Flatten it depth-first into the fast array.
        tree.flat.reserve(builder.nodes.len());
        flatten(&builder.nodes, 0, &mut tree.flat, &mut tree.leaf_bodies);
        tree
    }

    /// Gravitational acceleration on body `i` from all the others.
    ///
    /// What: returns the acceleration vector body `i` feels.
    /// How/why: iterate the flattened cells. For an internal cell, if it is far
    /// enough (`size² < θ²·distance²`) add its lump pull and jump past its subtree
    /// (`skip`); otherwise step into it. Leaves are summed body-by-body (skipping
    /// `i`). Each pull uses the softened law `a = G·m·r⃗/(|r⃗|² + ε²)^{3/2}`.
    /// Principle: Newton's gravity, with distant crowds approximated by their centre
    /// of mass.
    /// Units: `pos`/`mass` as [`build`](Self::build); `g` the gravitational constant;
    /// returns an acceleration in those units.
    pub fn acceleration(&self, i: usize, pos: &[DVec3], mass: &[f64], g: f64) -> DVec3 {
        let pi = pos[i];
        let mut acc = DVec3::ZERO;
        let mut idx = 0usize;
        let count = self.flat.len();
        while idx < count {
            let node = &self.flat[idx];
            if node.body_count == 0 {
                // Internal cell: lump it if far enough, else open it.
                let d = node.com - pi;
                let r2 = d.length_squared();
                if node.size2 < self.theta2 * r2 {
                    let soft = r2 + self.softening2;
                    acc += node.mass * d / (soft * soft.sqrt());
                    idx = node.skip as usize;
                } else {
                    idx += 1;
                }
            } else {
                // Leaf: direct pull from each body it holds (never from `i`).
                let s = node.body_start as usize;
                for &b in &self.leaf_bodies[s..s + node.body_count as usize] {
                    let b = b as usize;
                    if b != i {
                        add_pull(pos[b], mass[b], pi, self.softening2, &mut acc);
                    }
                }
                idx += 1;
            }
        }
        acc * g
    }

    /// Build a tree and return the acceleration on every body, in parallel.
    ///
    /// What: the convenient one-call version — build once, query all bodies across
    /// all CPU cores.
    /// How/why: the flattened tree is read-only, and each body's query only reads
    /// it, so the per-body loop is embarrassingly parallel; `rayon` spreads it over
    /// the cores. Range order is preserved, so the output is identical to (and as
    /// deterministic as) a sequential run — the tests that check it against the
    /// direct sum therefore also check the parallel path.
    /// Units: as [`build`](Self::build) / [`acceleration`](Self::acceleration).
    pub fn accelerations(pos: &[DVec3], mass: &[f64], theta: f64, softening: f64, g: f64) -> Vec<DVec3> {
        use rayon::prelude::*;
        let tree = Octree::build(pos, mass, theta, softening);
        (0..pos.len())
            .into_par_iter()
            .map(|i| tree.acceleration(i, pos, mass, g))
            .collect()
    }
}

/// Depth-first flatten of the pointer tree into the fast array.
///
/// What: appends the cell at `ni` and its subtree to `flat`, recording leaf bodies
/// in `leaf_bodies`, and sets each cell's `skip` to the index just past its subtree.
/// How/why: emitting in depth-first order makes "open" = next index and "lump" =
/// jump to `skip`.
/// Units: caller's own.
fn flatten(nodes: &[Node], ni: usize, flat: &mut Vec<FlatNode>, leaf_bodies: &mut Vec<u32>) {
    let n = &nodes[ni];
    let idx = flat.len();
    let size = 2.0 * n.half;
    flat.push(FlatNode {
        com: n.com,
        mass: n.mass,
        size2: size * size,
        skip: 0,
        body_start: 0,
        body_count: 0,
    });
    if n.internal {
        for &c in &n.children {
            if c >= 0 {
                flatten(nodes, c as usize, flat, leaf_bodies);
            }
        }
    } else {
        let start = leaf_bodies.len() as u32;
        if n.body >= 0 {
            leaf_bodies.push(n.body as u32);
        }
        leaf_bodies.extend_from_slice(&n.extra);
        flat[idx].body_start = start;
        flat[idx].body_count = leaf_bodies.len() as u32 - start;
    }
    let end = flat.len() as u32;
    flat[idx].skip = end;
}

/// Add the softened pull of one body at `src` (mass `m`) on a body at `dst`.
///
/// What: accumulates `m·(src−dst)/(|src−dst|² + ε²)^{3/2}` into `acc`.
/// How/why: the softened inverse-square law; the `ε²` keeps the force finite when
/// two sample particles nearly coincide.
/// Units: caller's own; `soft2 = ε²`.
fn add_pull(src: DVec3, m: f64, dst: DVec3, soft2: f64, acc: &mut DVec3) {
    let d = src - dst;
    let soft = d.length_squared() + soft2;
    *acc += m * d / (soft * soft.sqrt());
}

/// Which of the eight octants of a cube a point falls into (x = bit 0, …).
fn octant(center: DVec3, p: DVec3) -> usize {
    (p.x >= center.x) as usize
        | (((p.y >= center.y) as usize) << 1)
        | (((p.z >= center.z) as usize) << 2)
}

/// The centre and half-width of child octant `oct` of a cube.
fn child_box(center: DVec3, half: f64, oct: usize) -> (DVec3, f64) {
    let h = half * 0.5;
    let dx = if oct & 1 != 0 { h } else { -h };
    let dy = if oct & 2 != 0 { h } else { -h };
    let dz = if oct & 4 != 0 { h } else { -h };
    (center + DVec3::new(dx, dy, dz), h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic RNG (SplitMix64) for reproducible test clouds.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Build a random cloud of `n` bodies in the cube [-1,1]³ with masses 0.5..1.5.
    fn cloud(n: usize) -> (Vec<DVec3>, Vec<f64>) {
        let mut rng = Rng(0xDEAD_BEEF_1234_5678);
        let mut pos = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        for _ in 0..n {
            pos.push(DVec3::new(
                2.0 * rng.next() - 1.0,
                2.0 * rng.next() - 1.0,
                2.0 * rng.next() - 1.0,
            ));
            mass.push(0.5 + rng.next());
        }
        (pos, mass)
    }

    /// The exact O(N²) softened acceleration on body `i`, for checking the tree.
    fn direct(i: usize, pos: &[DVec3], mass: &[f64], soft2: f64) -> DVec3 {
        let mut a = DVec3::ZERO;
        for (j, &pj) in pos.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = pj - pos[i];
            let soft = d.length_squared() + soft2;
            a += mass[j] * d / (soft * soft.sqrt());
        }
        a
    }

    /// With θ = 0 no cell is ever lumped, so Barnes–Hut must equal the direct sum.
    #[test]
    fn theta_zero_matches_direct_sum() {
        let (pos, mass) = cloud(300);
        let soft = 0.05;
        let bh = Octree::accelerations(&pos, &mass, 0.0, soft, 1.0);
        for i in 0..pos.len() {
            let d = direct(i, &pos, &mass, soft * soft);
            assert!(
                (bh[i] - d).length() <= 1e-9 * d.length().max(1e-9),
                "body {i}: tree {:?} vs direct {:?}",
                bh[i],
                d
            );
        }
    }

    /// With a normal θ the tree approximates well: mean relative error stays small.
    #[test]
    fn barnes_hut_is_accurate() {
        let (pos, mass) = cloud(1500);
        let soft = 0.02;
        let bh = Octree::accelerations(&pos, &mass, 0.5, soft, 1.0);
        let mut sum_rel = 0.0;
        let mut worst: f64 = 0.0;
        for i in 0..pos.len() {
            let d = direct(i, &pos, &mass, soft * soft);
            let rel = (bh[i] - d).length() / d.length().max(1e-12);
            sum_rel += rel;
            worst = worst.max(rel);
        }
        let mean = sum_rel / pos.len() as f64;
        assert!(mean < 0.01, "mean relative error {mean} too high");
        assert!(worst < 0.1, "worst relative error {worst} too high");
    }

    /// Smaller θ must not be *worse* than larger θ (more opening = more accuracy).
    #[test]
    fn smaller_theta_is_more_accurate() {
        let (pos, mass) = cloud(800);
        let soft = 0.02;
        let err = |theta: f64| {
            let bh = Octree::accelerations(&pos, &mass, theta, soft, 1.0);
            (0..pos.len())
                .map(|i| {
                    let d = direct(i, &pos, &mass, soft * soft);
                    (bh[i] - d).length() / d.length().max(1e-12)
                })
                .sum::<f64>()
                / pos.len() as f64
        };
        assert!(err(0.2) <= err(0.8), "tighter theta should not be worse");
    }

    /// A lone body feels no force, and coincident bodies stay finite.
    #[test]
    fn degenerate_cases_are_safe() {
        let one = Octree::accelerations(&[DVec3::new(1.0, 2.0, 3.0)], &[1.0], 0.5, 0.1, 1.0);
        assert!(one[0].length() < 1e-12, "a single body feels no force");

        let same = vec![DVec3::ZERO; 5];
        let masses = vec![1.0; 5];
        let acc = Octree::accelerations(&same, &masses, 0.5, 0.1, 1.0);
        for a in acc {
            assert!(a.is_finite(), "coincident bodies must stay finite");
        }
    }

}
