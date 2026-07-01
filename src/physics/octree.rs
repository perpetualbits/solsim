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
//! `N·log N`, which is what makes an interactive galaxy collision possible.
//!
//! Everything is in `f64`. Units are left to the caller (pass your own `G`); the
//! forces are softened (a small `ε`) so two bodies that come very close do not fly
//! apart from a division by nearly zero — the right thing for a *smooth* galaxy
//! made of sample particles.
//!
//! Phase 1 of the galaxy mode: the compute core, validated by its tests against the
//! direct O(N²) sum. Nothing draws it yet, so allow the as-yet-unused public API.
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

/// One cube in the octree.
///
/// What: a cell's geometry, the total mass and centre of mass of everything inside
/// it, and links to its up-to-eight children.
/// How/why: a "leaf" holds a single body in `body` (with any exact duplicates in
/// `extra`); an "internal" cell has `internal = true` and points at children. The
/// aggregate `mass`/`com` let Barnes–Hut treat a whole cell as one lump.
/// Units: `center`/`half`/`com` in the caller's length unit; `mass` in its mass
/// unit.
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
    /// A fresh empty cell covering the cube centred at `center` with half-width
    /// `half`.
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

/// A built Barnes–Hut octree over a set of bodies.
///
/// What: the cells (in one flat array) plus the accuracy/softening settings.
/// How/why: a flat `Vec<Node>` with integer child links is cache-friendlier than
/// heap-pointered nodes and avoids lifetime headaches.
/// Units: `theta2 = θ²` and `softening2 = ε²` are dimensionless / length² in the
/// caller's units.
pub struct Octree {
    nodes: Vec<Node>,
    theta2: f64,
    softening2: f64,
}

impl Octree {
    /// Build the octree from body positions and masses.
    ///
    /// What: returns a tree ready to answer gravity queries.
    /// How/why: find the cube that bounds all bodies, then insert each body,
    /// accumulating every cell's total mass and centre of mass as we go.
    /// Principle: the tree records, at every scale, "how much mass is roughly here",
    /// which is all Barnes–Hut needs to lump distant groups together.
    /// Units: `pos` in a length unit, `mass` in a mass unit; `theta` dimensionless
    /// (smaller = more accurate, slower); `softening` in the same length unit.
    pub fn build(pos: &[DVec3], mass: &[f64], theta: f64, softening: f64) -> Octree {
        let mut tree = Octree {
            nodes: Vec::new(),
            theta2: theta * theta,
            softening2: softening * softening,
        };
        if pos.is_empty() {
            tree.nodes.push(Node::empty(DVec3::ZERO, 1.0));
            return tree;
        }
        // The smallest cube that contains every body.
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let center = (lo + hi) * 0.5;
        // Pad a touch so bodies on the boundary sit strictly inside.
        let half = (hi - lo).max_element() * 0.5 * 1.0001 + 1e-12;
        tree.nodes.push(Node::empty(center, half));
        for i in 0..pos.len() {
            tree.insert(0, i, 0, pos, mass);
        }
        tree
    }

    /// Insert body `body` into the cell `node` at tree depth `depth`.
    ///
    /// What: adds one body to the tree, splitting the cell if it was already
    /// occupied.
    /// How/why: first fold the body into this cell's running mass and centre of
    /// mass. An empty cell simply stores the body; an occupied leaf subdivides and
    /// pushes both bodies down into the right child cubes; an already-internal cell
    /// forwards the body to the child it falls in. At the depth cap, coincident
    /// bodies just share the leaf.
    /// Units: as [`build`].
    fn insert(&mut self, node: usize, body: usize, depth: u32, pos: &[DVec3], mass: &[f64]) {
        // Fold this body into the cell's total mass and centre of mass.
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
        // Empty leaf: just take the body.
        if self.nodes[node].body < 0 {
            self.nodes[node].body = body as i32;
            return;
        }
        // Occupied leaf at the depth cap: bucket coincident bodies together.
        if depth >= MAX_DEPTH {
            self.nodes[node].extra.push(body as u32);
            return;
        }
        // Otherwise split: move the sitting body down, then the new one.
        let existing = self.nodes[node].body as usize;
        self.nodes[node].body = -1;
        self.nodes[node].internal = true;
        self.insert_into_child(node, existing, depth, pos, mass);
        self.insert_into_child(node, body, depth, pos, mass);
    }

    /// Forward a body into the correct child cube of `parent`, creating it if new.
    ///
    /// What: routes a body one level down the tree.
    /// How/why: pick the octant the body lies in, make that child cube if it does
    /// not exist yet, then insert into it one level deeper.
    /// Units: as [`build`].
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

    /// Gravitational acceleration on body `i` from all the others.
    ///
    /// What: returns the acceleration vector body `i` feels.
    /// How/why: walk the tree from the root. A far-enough internal cell (its size is
    /// small next to its distance, `size/distance < θ`) is treated as one lump at
    /// its centre of mass; otherwise we open it and look at its children. Leaves are
    /// summed body-by-body (skipping `i` itself). Each pull uses the softened
    /// inverse-square law `a = G·m·(r⃗)/(|r⃗|² + ε²)^{3/2}`.
    /// Principle: Newton's gravity, with distant crowds approximated by their centre
    /// of mass — accurate because a far clump really does pull almost exactly like a
    /// point at its centre of mass.
    /// Units: `pos`/`mass` as [`build`]; `g` the gravitational constant in the
    /// caller's units; returns an acceleration in those units.
    pub fn acceleration(&self, i: usize, pos: &[DVec3], mass: &[f64], g: f64) -> DVec3 {
        let mut acc = DVec3::ZERO;
        self.accel_node(0, i, pos, mass, &mut acc);
        acc * g
    }

    /// Recursive helper for [`acceleration`] (adds one cell's pull into `acc`).
    ///
    /// What: accumulates the pull of cell `node` on body `i`.
    /// How/why: applies the opening test `(size)² < θ²·distance²` to decide lump vs
    /// recurse; see [`acceleration`].
    /// Units: accumulates the un-scaled sum (the caller multiplies by `g`).
    fn accel_node(&self, node: usize, i: usize, pos: &[DVec3], mass: &[f64], acc: &mut DVec3) {
        let n = &self.nodes[node];
        if n.mass == 0.0 {
            return;
        }
        if n.internal {
            let d = n.com - pos[i];
            let r2 = d.length_squared();
            let size = 2.0 * n.half;
            if size * size < self.theta2 * r2 {
                let soft = r2 + self.softening2;
                *acc += n.mass * d / (soft * soft.sqrt());
            } else {
                for &c in &n.children {
                    if c >= 0 {
                        self.accel_node(c as usize, i, pos, mass, acc);
                    }
                }
            }
        } else {
            if n.body >= 0 && n.body as usize != i {
                add_pull(pos[n.body as usize], mass[n.body as usize], pos[i], self.softening2, acc);
            }
            for &b in &n.extra {
                if b as usize != i {
                    add_pull(pos[b as usize], mass[b as usize], pos[i], self.softening2, acc);
                }
            }
        }
    }

    /// Build a tree and return the acceleration on every body (single-threaded).
    ///
    /// What: the convenient one-call version — build once, query all bodies.
    /// How/why: the per-body queries are independent and read-only, so this loop is
    /// the natural place to spread across cores later (e.g. with `rayon`); for now
    /// it is a plain sequential map so correctness is easy to check.
    /// Units: as [`build`] / [`acceleration`].
    pub fn accelerations(pos: &[DVec3], mass: &[f64], theta: f64, softening: f64, g: f64) -> Vec<DVec3> {
        let tree = Octree::build(pos, mass, theta, softening);
        (0..pos.len())
            .map(|i| tree.acceleration(i, pos, mass, g))
            .collect()
    }
}

/// Add the softened pull of one body at `src` (mass `m`) on a body at `dst`.
///
/// What: accumulates `m·(src−dst)/(|src−dst|² + ε²)^{3/2}` into `acc`.
/// How/why: the softened inverse-square law; the `ε²` keeps the force finite when
/// two sample particles nearly coincide (we model a smooth mass, not point stars).
/// Units: length/mass in the caller's units; `soft2 = ε²`.
fn add_pull(src: DVec3, m: f64, dst: DVec3, soft2: f64, acc: &mut DVec3) {
    let d = src - dst;
    let soft = d.length_squared() + soft2;
    *acc += m * d / (soft * soft.sqrt());
}

/// Which of the eight octants of a cube a point falls into.
///
/// What: returns 0..8, one bit per axis (x = bit 0, y = bit 1, z = bit 2).
/// How/why: comparing the point to the cube centre on each axis gives the child
/// index directly.
/// Units: positions in the caller's length unit.
fn octant(center: DVec3, p: DVec3) -> usize {
    (p.x >= center.x) as usize
        | (((p.y >= center.y) as usize) << 1)
        | (((p.z >= center.z) as usize) << 2)
}

/// The centre and half-width of child octant `oct` of a cube.
///
/// What: returns `(child_center, child_half)`.
/// How/why: each child is half the size, offset by a quarter of the parent width in
/// each axis according to the octant's bits.
/// Units: length in the caller's unit.
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

    /// A lone body feels no force (no self-attraction), and coincident bodies are
    /// handled without blowing up.
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
