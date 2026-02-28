# CLAUDE.md — Artificial Life Evolution Simulator
## 0. Working TODO
* CONSIDER: before running a full simm for evolution, run a viability check, that the organism lives in this condition, IE no competition just food.

## 1. Project Overview

This project is a **2D artificial life evolution simulator** where genetically distinct organisms compete for surface area dominance on a hex grid battlefield. Think of it as simulating a petri dish: organisms are placed on the grid, consume resources, grow, reproduce, fight, and the genome that occupies the most surface area (cell count) at the end of a match wins.

### Core Goals
- **Diverse meta-strategies must be viable**: predators, resource gatherers, plants/growers/amoebas, armored fortresses, fast swarming reproducers, parasites, etc.
- **The genotype encodes everything**: organism shape (body plan), features (cell types), and behavioral parameters.
- **Rich feature set**: ~100 cell types with tunable parameters (spikes, mouths, teeth, armor, skin, eyes, feet/flagella, photosynthetic membranes, chemical sensors, signal emitters, etc.) enabling different strategies to emerge.
- **Organisms are physically embodied on the grid**: individual cells can be destroyed, severed, or regrown. A bite removes a specific cell. Shape matters tactically. Organisms can/should be multi cellular.
- **Evolution happens between matches**, not during. The simulation is a fitness evaluation function. Tournament-based evolutionary algorithm drives the outer loop.
- **Deterministic simulation**: given the same inputs and random seed, a match replays identically. This enables batch evolution runs followed by replay visualization of interesting matches.

### Design Philosophy
- **Modularity is paramount**: genome encoding, brain architecture, evolution strategy, body plan generation, and sensor systems are all swappable modules behind clean interfaces. We are uncertain which approaches will produce the best emergent behavior, so the system must support experimentation.
- **CPU-viable, GPU-optimized**: test/demo runs on CPU. Architecture uses flat arrays and pure-function cell updates so the hot simulation loop can be ported to GPU (Taichi) without restructuring.
- **Build incrementally**: each implementation phase produces a runnable, testable system. Never more than one phase away from something you can observe.

---

## 2. Tech Stack & Environment

### Languages
- **Kotlin (JVM)**: Primary language. Owns the orchestrator layer — evolution, tournament management, genome management, configuration, match coordination, serialization, CLI/entry point.
- **Python**: Simulation computation layer. Owns the GPU-acceleratable simulation kernel with Taichi, CPPN evaluation, and any ML workloads. Called from Kotlin via Jep (Java Embedded Python). 
  - Always use the project's Python venv for Python commands: prefix with `.venv/bin/` (e.g., `.venv/bin/python`, `.venv/bin/pytest`).
  - When python dictionary / set types are ambiguous, like dict[int,int] use TypeAlias
### Coding Standards:
- Always use 4 spaces for indentation.
- Important!: For function names that can not be easily described by their name, add a one to two line comment. Not
  all functions should have this, at most 50%, so be selective. Us this when description from name and actual 
  implementation are abstract enough to justify a 1 or 2 line comment. 

#### **Python**:
 - ALWAYS include type hints and type annotations were possible. (Except in Taichi kernels, where it breaks compilation)

#### **Kotlin**:
 - No * imports, full paths for everything.

### Key Libraries & Frameworks
- **Jep (Java Embedded Python)**: Embeds a Python interpreter inside the JVM process. Kotlin calls Python functions directly with near-zero overhead. NumPy arrays can be shared with minimal copying. This avoids subprocess/serialization overhead of a client-server architecture.
- **Taichi Lang** (Python): GPU-acceleratable simulation kernels. Compiles Python-like code to CUDA/OpenCL/CPU-LLVM. Supports hex grid simulations natively. CPU mode is fast enough for development; GPU mode for production evolution runs.
- **kotlinx.serialization**: Genome and configuration persistence (JSON format).
- **kotlinx.coroutines**: Parallel match execution on CPU cores.
- **Multik or ejml**: Matrix operations for neural network evaluation if/when NN brains are implemented.

### Build & Environment
- **Gradle with Kotlin DSL**: Build system.
- **Docker**: Primary deployment target. Single container with JDK 21+, Python 3.11+, Jep, Taichi, CUDA toolkit.
- **Base image**: `nvidia/cuda:12.2-devel-ubuntu22.04` for GPU support.
- **CPU fallback**: Must work without GPU. Taichi automatically falls back to CPU (LLVM backend). Docker container should function without `--gpus` flag, just slower.
- **Volume mounts**: Evolution state, genome databases, match replays, and checkpoints persisted to mounted host directories. Never lose long evolution runs to container restarts.

### Docker Considerations
- **Linux**: Full GPU support via NVIDIA Container Toolkit (`docker run --gpus all`).
- **Windows**: GPU passthrough via WSL2 + Docker Desktop. Supported with recent NVIDIA drivers.
- **Mac**: No NVIDIA GPU support. CPU-only mode. Taichi CPU backend (LLVM) is still reasonably fast for development and small-scale runs.
- **Runtime detection**: At startup, detect GPU availability and select compute backend accordingly.

### Jep Integration Notes
- Python interpreter runs inside the JVM process — not a separate service.
- Each Jep `SharedInterpreter` is thread-bound (Python GIL constraint). For parallel matches, create multiple interpreters on separate threads.
- The GIL is released during GPU kernel execution (Taichi/CUDA), so GPU compute is not GIL-bottlenecked.
- Taichi kernels stay compiled in GPU memory across matches within the same interpreter — no recompilation overhead between matches.
- Data exchange: match config and genome data passed as Python dicts/numpy arrays. Results returned as Python objects converted to Kotlin types.

---

## 3. Architecture Overview

The system has two layers connected by Jep:

```
┌─────────────────────────────────────────────────┐
│  KOTLIN ORCHESTRATOR (JVM)                      │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐              │
│  │  Evolution  │  │  Tournament  │              │
│  │  Strategy   │  │  Manager     │              │
│  └──────┬──────┘  └──────┬───────┘              │
│         │                │                      │
│  ┌──────┴──────┐  ┌──────┴───────┐              │
│  │  Genome     │  │  Match       │              │
│  │  Manager    │  │  Coordinator │              │
│  └─────────────┘  └──────┬───────┘              │
│                          │ Jep                  │
├──────────────────────────┼──────────────────────┤
│  PYTHON SIMULATION (embedded via Jep)           │
│                          │                      │
│  ┌──────────────┐   ┌────┴──────────┐           │
│  │  Body Plan   │   │  Simulation   │           │
│  │  Providers   │   │  Engine       │           │
│  │  (CPPN etc)  │   │  (Taichi)     │           │
│  └──────────────┘   └──────┬────────┘           │
│                            │                    │
│  ┌──────────────┐  ┌───────┴───────┐            │
│  │  Brain       │  │  Hex Grid     │            │
│  │  Providers   │  │  (core data)  │            │
│  └──────────────┘  └───────────────┘            │
└─────────────────────────────────────────────────┘
```

### Separation of Concerns
- **Kotlin** owns: evolution logic, tournament brackets, genome CRUD, population management, configuration, persistence, entry point, replay management.
- **Python** owns: simulation execution, hex grid state, cell update kernels, CPPN evaluation, sensor aggregation, brain evaluation, combat resolution, energy accounting.
- **The boundary**: Kotlin sends match configuration + serialized genomes to Python. Python runs the match and returns fitness scores + optional replay data. This is a function call via Jep, not a network request.

### Six Swappable Module Interfaces
All behind clean interfaces. Implementations can be swapped via configuration:

1. **Body Plan Provider** — genome → cell layout. Implementations: CPPN, lookup table, L-system, GRN.
2. **Brain Provider** — sensor inputs → action outputs. Implementations: rule-based, fixed NN, NEAT NN, behavior tree, hardcoded script.
3. **Genome Representation** — the heritable data structure. Implementations: flat float vector, NEAT graph genome, composite genome.
4. **Evolution Strategy** — match results → next generation. Implementations: tournament selection, Elo-rated, MAP-Elites, manual selection.
5. **Match Configuration** — data-driven arena setup. Grid size, tick limit, resource distribution, competitor count, placement strategy.

---




## Core Domain Model

### Hex Grid
- **Coordinate system**: Axial coordinates (q, r). Each hex has exactly 6 equidistant neighbors.
- **Toroidal topology**: the grid wraps in both axes — coordinate `width` maps back to `0`, and `-1` maps to `width-1` (same for height). All coordinate functions (`neighbors`, `hex_distance`, `line_of_sight`) account for wrapping and return the shortest toroidal path. There are no edges or corners; every tile has exactly 6 neighbors.
- **Why hex**: eliminates diagonal movement ambiguity of square grids, produces natural organic shapes, rotation is a clean coordinate transform (6 orientations at 60° increments), uniform distance in all directions.
- **Grid contents**: each hex tile has two layers: a **terrain type** (ground, water, rock, fertile soil, toxic, etc.) and **contents** (empty, resource, or organism cell). Terrain is the floor; organisms and resources sit on top. A water tile can contain a swimming organism's cell. A bedrock tile is impassable.

### Organisms
An organism is a **coherent entity** composed of contiguous cells on the hex grid. It is NOT an agent floating above the grid — its cells physically occupy hex tiles.

Each organism has:
- **Unique ID**: assigned at birth.
- **Genome**: heritable data encoding body plan, brain parameters, and metabolic parameters.
- **Cell set**: the set of hex tiles this organism's cells occupy. Cells have types (mouth, spike, armor, photosynthetic, eye, flagella, etc.).
- **Energy pool**: shared across all cells. Contributed to by photosynthetic cells and eating. Drained by cell maintenance, movement, growth, reproduction.
- **Brain state**: the current output of the brain (movement direction, grow/attack/reproduce decisions). Updated each tick from sensor inputs.

### Cells (~100 types, examples)
Cell Attributes:
see cell_types.py

### Energy System
- Every cell costs maintenance energy per tick (varies by cell type), total energy cost for all cells is subtracted from the organism each tick.
- Photosynthetic cells generate energy proportional to light availability.
- Mouth cells transfer energy from consumed food/cells to the organism's pool.
- Movement costs energy proportional to organism mass (cell count). Terrain modifies cost (water may reduce movement cost for aquatic organisms).
- Growth (adding a new cell) costs energy dependent on cell type.
- Reproduction costs: energy equal to the new cells generated + energy transferred to offspring as starting energy + base reproduction cost.
- If energy reaches zero, organism dies (starvation).

---

---

## Simulation Tick Rules (CORE ENGINE)

Each tick executes the following steps **in this exact order** 
(order matters for determinism, must deterministically resolve conflicts):

### Step 1: Resource Generation
- Energy generating cells (photosynthetic, root) compute energy gain and add to organism energy pool.
- NPC cells (Food) regenerate at a configured rate.

### Step 2: Sensor Aggregation
- For each organism, collect what its sensor cells detect.
- Eye cells: ray-cast (or hex-walk) in their facing direction to detect entities.
- Chemical sensors: read local chemical concentrations.
- Touch sensors: detect adjacent foreign cells.
- Terrain awareness: cells detect the terrain type of their own tile and adjacent tiles (available to brain for movement decisions).
- Compile into the body-invariant sensor input vector.

### Step 3: Brain Evaluation
- For each organism, pass sensor vector through brain (rule-based or NN).
- Receive action outputs: movement, growth, reproduction.

### Step 4: Movement Execution
Process in sub-steps. Each sub-step resolves its own conflicts before the next begins.

IMPORTANT: Organisms 'speed' is manifest by the delay between their last movement and next movement. Larger delay = slower.
This way fast organisms move multiple times before slower things. However overall movement per tick is just 1 cell in a given direction.

#### **4a. Movement Point Accumulation**
locomotion_power = sum(cell.locomotion_power for each cell in organism)
total_cell_mass = sum(cell.mass for each cell in organism)

Each organism stores `movement_points` (integer, initialized to 0 at birth).

Each tick, for every organism with locomotion_power > 0:
    movement_points += locomotion_power
    movement_points = min(movement_points, total_cell_mass)

The cap (`min`) is critical: it prevents organisms from banking movement points while stationary.
An organism that stands still for 100 ticks gains no speed advantage over one that stood still for 1 tick.
Once capped, excess accumulation is simply discarded.

An organism is **movement-ready** when `movement_points >= total_cell_mass`.

This is Bresenham-style integer accumulation: an organism with locomotion_power=3 and total_cell_mass=10
moves exactly 3 times per 10 ticks, evenly distributed. Doubling locomotion_power exactly doubles
movement frequency. No floating-point arithmetic needed.

Organisms with zero locomotion_power are **immobile** — they never accumulate movement points.


### Step 5: Growth
Border cells of growing organisms attempt to fill adjacent empty hexes. **Growth is always additive at the border**. 
No interior expansion; no pushing existing cells outward. (consider changing later)

Like movement, growth may also have conflicts with other organisms. Conflicts are resolved in a similar manor, computing
claims and breaking ties, however quickness is not considered for growth conflicts.

Execution:
1. The body plan provider (CPPN or lookup table) is queried for each empty hex adjacent to the organism's border cells,
   and the organisms existing cells: "should a cell exist here, and what type?"
2. Validate Energy Costs:
    - For each new cell to grow, the organism must have sufficient energy to pay for the cost.
    - If not all cells can be grown, grow the most possible in a deterministic order.
    - Discard any growth requests that cannot be paid for before moving on to validation and conflict resolution.
3. Validate ALL destination hexes:
    - Destination terrain is eligible for cell placement.
    - Destination is not occupied by another cell.
    - Destinations that are occupied by the current organism's cells are valid, but will destroy the existing cell.
4. Compute Organism's Growth Priority Rank (summing these values):
    - (+) (MAX_ORGANISM_CELLS - organism_cell_count) * 1,000 (medium priority: fewer cells move first)
    - (+) organism_id (low priority: higher id (younger) organisms move first)
    - IMPORTANT: follow similar logic/constraints as in movement.
5. Generate claims for each destination hex:
   - Follow similar logic as in movement.
6. Generate new cells + Deduct Costs.
   - Update the grid state with the new cell-type and organism_id for each successful claim.
   - Deduct growth cost from the organism energy. 
   - Growth cost is cell-type-dependent (configured per cell type, e.g., pseudopod=2, skin=5, armor=15).

**NOTE: Amoeboid Flow (via Growth + Retraction)**

Amoeboid movement is NOT a separate movement mode. It emerges from combining two standard actions: 
**directed growth** at the leading edge + **voluntary retraction** at the trailing edge.

Organisms my also choose to absorb / delete their own cells. Some cell types are designed to be deleted, providing
a refunded energy cost to the organism when absorbed.

- The brain requests growth in a specific direction and retraction from the opposite side.
- **Retraction = cell death**: the retracted cell is destroyed, its hex becomes empty.
- **Regrowth at the leading edge** costs the full growth price of the new cell type.
- This means amoeboid flow is only economically viable for **cheap cell types**:
  - Pseudopod: grow cost ~2, retract+regrow ~3-4 total → affordable, designed for this
  - Soft tissue: grow cost ~3, retract+regrow ~5-6 → marginal
  - Armor: grow cost ~15, retract+regrow ~28-30 → economically unviable
- Organisms can use BOTH flagella shift AND amoeboid flow: shift the rigid body, then extend pseudopod tendrils at the leading edge. These are independent actions in the same tick.

IMPORTANT: this kind of movement / growth should be considered when designing brain growth interfaces.

#### **Reproduction** Special Kind of New Cell Growth
- Organisms that choose to reproduce and have sufficient energy spawn a seed cell of any cell type in an adjacent empty hex.
- This should happen simultaneously with growth, brains should decide on reproduction at the same time as growth.
- The seed carries a copy of the parent's genome (no mutation during match — clones only).
- Parent's energy is reduced by the reproduction cost + new cell cost + chosen starting energy of seed.
The seed is a new organism (new ID, same genome) that begins its own developmental growth on subsequent ticks.

### Step 6: Action Resolution
- After all movement and growth are complete, any cell types which have possible actions are executed.
- Each cell type has a list of possible actions, each with a different set of conditions.
- For simplicity, actions are always performed if possible not decided by the brain.
- Consider combining this step with Step 1: Resource Generation.
- Example Actions:
  - Mouth: consume valid hex next to it IF its NOT the same genome and alive, destroying it and transferring energy to the organism's pool.
  - Spikes: destroy valid adjacent hexes of organisms with different genomes, or dead cells.

### Step 7: Death and Cleanup
- Organisms with zero cells are marked as dead and removed from live organism list.
- Organisms with zero energy are marked as dead, their cells remain on the grid, but are detectable as dead cells.

### Determinism Guarantee
- All random decisions use a seeded PRNG.
- Movement conflicts resolved by deterministic priority: fastest organism wins, ties broken by lighter mass, then by youngest organism.
- Growth conflicts use the same priority system, except speed is not considered.
- No floating-point non-determinism: use fixed-point arithmetic or careful GPU reduction ordering where needed.

---

## 7. Organism Lifecycle

### Birth (from reproduction)
1. Parent organism decides to reproduce (brain output).
2. Parent expends reproduction energy cost.
3. A **seed cell** is placed in an adjacent empty hex. This is a single cell carrying the parent's genome and a starting energy allotment.
4. The seed is a new organism with its own ID but the same genome as parent.

### Developmental Growth
1. The seed cell begins executing the body plan provider (CPPN or template).
2. Each tick, border cells query the body plan: "what should grow next to me?"
3. New cells are added at the border, consuming energy.
4. Growth proceeds outward from the seed, guided by the body plan, over many ticks.
5. The CPPN's developmental time input (`t` / organism age) enables temporal staging: grow core survival cells first (photosynthesis, basic sensors), then specialized cells later (weapons, armor).
6. Growth continues until the body plan says "stop" (mature size) OR the organism runs out of energy.
7. **Unbounded growth** is possible for plant-like organisms whose body plan never says stop — they expand indefinitely as long as energy allows. Natural limits: available space, maintenance costs exceeding photosynthesis income, predation.

### Damage and Regeneration
1. When cells are destroyed, the organism loses those cells.
2. Remaining border cells at the wound site detect empty adjacent hexes.
3. These border cells query the body plan provider: "should a cell exist here?"
4. If yes, and the organism has energy, the cell regrows.
5. Regeneration uses **the same growth mechanism** as initial development — no separate regeneration system needed.
6. Regeneration may not perfectly restore the original shape (depends on body plan encoding and local context), which is biologically realistic.

### Death
An organism dies when:
- All cells are destroyed.
- Energy reaches zero.
- 
#### Partial Death
- When an organisms cells are destroyed leaving it in two or more disconnected clusters of hexes, the largest
  of the divided components remains alive, while the others are removed from the organism and marked as dead.

---

## 8. Genome & Body Plan & Mutation

### Starting Implementation: Fixed-Topology CPPN

A **Compositional Pattern Producing Network** (CPPN) is a small neural network that maps spatial coordinates to cell properties.

**CPPN Inputs** (per hex position query):
- `d`: distance from organism center
- `a`: angle from organism center (discretized to hex directions or continuous)
- Symmetry-transformed coordinates (see below)
- `t`: developmental time (organism age in ticks)
- Neighbor context: count and types of filled neighbors (for context-sensitive growth)

**CPPN Outputs** (per hex position):
- `cell_exists`: float (thresholded to bool — should a cell be here?)
- `cell_type`: float vector (softmax over cell types — which type?)
- `growth_priority`: float (how urgently to grow here vs. other border positions)

**Fixed topology**: All CPPNs have the same architecture. Suggested starting point: 6 inputs → 16 hidden (layer 1) → 16 hidden (layer 2) → outputs. ~400 trainable weights. This is the genome for the body plan.

**Activation functions per neuron**: Each neuron has an evolved activation function selected from: {sin, cos, gaussian, sigmoid, tanh, relu, step, linear, abs}. These are encoded as an integer per neuron in the genome. Sine produces repeating patterns (segments, stripes). Gaussian produces localized features. Step produces sharp boundaries.

### Symmetry Modes
The genome includes a small set of parameters controlling how hex coordinates are transformed before being fed to the CPPN:

- **Bilateral symmetry**: CPPN receives `|q|` instead of `q`. Left-right mirroring.
- **Radial symmetry (N-fold)**: CPPN receives `(d, a mod (2π/N))`. N is evolvable (2, 3, 4, 5, 6).
- **Asymmetric**: CPPN receives raw `(q, r)`. No symmetry constraint.

The symmetry mode itself is part of the genome and can mutate.

### Full Genome Structure
```
Genome = {
  body_plan: {
    cppn_weights: float[~400]        # connection weights
    activation_functions: int[~32]    # per-neuron activation choice
    symmetry_mode: enum               # bilateral, radial-N, asymmetric
    symmetry_params: float[2]         # e.g., N for radial symmetry
  },
  brain: {
    rule_parameters: float[~30]       # thresholds and weights (rule-based)
    # OR nn_weights: float[~500]      # if using NN brain
  },
  metabolism: {
    reproduction_energy_threshold: float   # energy level to trigger reproduction
    offspring_energy_allocation: float      # energy given to offspring
    growth_rate_ceiling: float             # max cells grown per tick
    maintenance_cost_multiplier: float     # scales base maintenance costs
    movement_willingness: float            # how readily the organism moves
  }
}
```

### Mutation Operators
- **Weight perturbation**: add Gaussian noise to CPPN weights and brain parameters.
- **Activation function swap**: randomly change a neuron's activation function.
- **Symmetry mode change**: rare mutation that switches symmetry type.
- **Metabolic parameter perturbation**: small adjustments to life-history parameters.
- **Mutation rates are configurable** per operator.

### Crossover
With fixed-topology CPPNs, crossover is straightforward:
- For each weight: randomly select from parent A or parent B (uniform crossover).
- Or: interpolate between parents (blend crossover).
- Activation functions: randomly select from either parent per neuron.
- Metabolic parameters: blend or select per parameter.

### Future Upgrade Path: NEAT
If fixed topology proves limiting, NEAT-style variable topology can be added:
- Genomes encode network topology (nodes + connections) with innovation numbers.
- Crossover aligns by innovation number.
- Structural mutations: add node, add connection.
- GPU batching: pad all networks to maximum size with zeroed weights.
- This is a drop-in replacement for the body plan portion of the genome — nothing else in the architecture changes.

---

## 9. Brain Architecture

### Starting Implementation: Rule-Based with Evolved Parameters

The brain is a fixed set of prioritized behavioral rules. The genome encodes **threshold and weight parameters** that tune behavior, not the rules themselves.

**Rule structure** (evaluated top-to-bottom, first matching rule fires):
```
1. IF recent_damage > damage_flee_threshold AND has_locomotion:
     → move away from nearest_threat at max speed
2. IF nearest_threat_distance < threat_distance_threshold AND has_locomotion:
     → move away from nearest_threat at speed = threat_response_speed
3. IF internal_energy < hunger_threshold AND nearest_food_distance < food_seek_range:
     → move toward nearest_food, activate mouth
4. IF internal_energy > reproduction_threshold AND organism_age > min_reproduce_age:
     → reproduce
5. IF internal_energy > growth_threshold:
     → grow
6. IF nearest_food_distance < food_seek_range AND has_locomotion:
     → move toward nearest_food
7. ELSE:
     → wander randomly (or stay still if no locomotion)
```

All capitalized parameters are genome-encoded floats. Different parameter values produce dramatically different behaviors:
- **Predator**: low threat_response, high food_seek_range, low reproduction_threshold (reproduce fast to swarm).
- **Plant**: no locomotion cells, growth_threshold near zero (always grow), high reproduction_threshold (grow large before reproducing).
- **Defensive**: high threat_response, high damage_flee_threshold, moderate growth.

### Why Rule-Based First
- Eliminates the brain-body coupling problem — rules are body-invariant.
- No training loop needed — parameters are directly evolved.
- Simple to debug and understand when observing organism behavior.
- GPU-friendly — just parameter lookups and comparisons, no matrix math.
- Sufficient behavioral diversity through parameter evolution + body plan diversity.

### Sensor Abstraction (Body-Invariant Inputs)
The sensor aggregator normalizes raw cell data into a fixed-size input vector that means the same thing regardless of body shape:
- More eye cells → better detection range and angular accuracy.
- More chemical sensors → more precise gradient reading.
- But the brain input format is always the same size and semantics.

This ensures that body mutations don't break the brain.

### Future Upgrade Path: Neural Network Brain
When ready, replace rule-based brain with a fixed-topology NN:
- Same input vector (sensor abstraction layer already built).
- Same output vector (movement, grow, reproduce, attack, etc.).
- Genome encodes NN weights instead of rule parameters.
- Batch all brain evaluations as one matrix multiply on GPU.
- Consider Option 2 (sensor abstraction, already implemented) to handle brain-body coupling.
- Consider Option 3 (Lamarckian inner loop — short practice phase per genome) if coupling is still problematic.

---

## 10. Evolution & Tournament System

### Outer Loop
```
1. Initialize population of N genomes (e.g., N=100), randomly generated.
2. Repeat for G generations:
   a. Run M matches per generation (e.g., M=200).
      - Each match: select 2-5 genomes, instantiate in petri dish, simulate for tick_limit ticks.
      - Offspring during match are clones (same genome, no mutation).
      - Fitness Function = surface area (cell count) per genome at match end.
   b. Aggregate fitness across matches (average score, or Elo rating).
   c. Select survivors (top K genomes by fitness).
   d. Produce next generation via mutation and crossover of survivors.
   e. Persist generation state (checkpoint).
3. Output: evolved population of genomes.
```

### Matchmaking Strategies (swappable)
- **Random**: randomly sample 2-5 genomes per match. Simple, unbiased.
- **Elo-rated**: match genomes with similar ratings for more informative matches. Genome ratings update after each match using Elo/TrueSkill formula.
- **Round-robin bracket**: ensure every genome faces every other genome at least once per generation. More expensive but thorough.

### Fitness Scoring
- Primary metric: **total cell count of genome at end of match** (sum across all clones of that genome alive at match end).
- Averaged (or Elo-rated) across all matches the genome participated in during a generation.
- Fitness should be evaluated against diverse opponents to avoid overfitting to one adversary.

### Selection and Reproduction
- **Tournament selection**: randomly pick K genomes, the fittest advances. Repeat to fill next generation.
- **Elitism**: top E genomes pass to next generation unchanged.
- **Mutation**: apply mutation operators to offspring genomes.
- **Crossover**: optionally combine two parent genomes (configurable crossover rate).

### Map Variation
- Vary arena parameters across matches within a generation: different grid sizes, resource distributions, light levels.
- This evolves **generalist** organisms rather than specialists for one map.

### Match Length
- Tunable parameter with significant strategic implications.
- Short matches favor fast growers and aggressive strategies.
- Long matches favor sustainable, defensive, regenerative strategies.
- Consider increasing match length as evolution progresses (curriculum).

---

## 11. Manual Organisms & NPCs

### Purpose
- **Testing**: manually designed organisms verify that simulator mechanics work before evolution is layered on.
- **Seeding**: manual archetypes can bootstrap evolutionary populations.
- **NPCs**: simple organisms (plants, mice) that serve as environment/resources for evolved organisms.

### Manual Body Plan
- Skip the CPPN. Provide a **literal hex cell map** — a template where each hex position has a cell type (or empty).
- Built with a hex grid editor tool (or defined in JSON/config).
- Growth uses the template: organism grows from seed outward, filling in cells that match the template, layer by layer.
- Regeneration: border cells check the template and regrow missing cells.
- Optional **growth order** layer map: layer 0 = seed, layer 1 = first ring, etc.

### Manual/Scripted Brain
- **Behavior tree** or **priority rule list** authored by hand.
- Simple examples: plant brain = "always grow, never move"; mouse brain = "flee from bigger organisms, eat food, reproduce when energy high".
- Implements the same BrainProvider interface as evolved brains.

### Organism Contract (unified interface)
The simulator doesn't know or care whether an organism is evolved, manual, or NPC. All satisfy the same interface:

|              | Evolved           | Manual             | NPC              |
|--------------|-------------------|--------------------|------------------|
| Body plan    | CPPN query        | Lookup table       | Lookup table     |
| Brain        | Rule-based / NN   | Behavior tree      | Hardcoded script |
| Metabolism   | Evolved params    | Hand-tuned         | Hand-tuned       |
| Reproduction | Clones (in match) | Clones or disabled | Clones           |

### Seeding Evolution from Manual Designs
- Optionally train a CPPN to approximate a manual body plan (supervised: minimize cell-type error across hex positions). This produces a CPPN genome that reproduces the manual design and can then be evolved via normal mutation. Gives evolution a head start.

---

## 13. Key Design Decisions Log

Decisions made during design, with rationale. Do not revisit without good reason.

| Decision                                                         | Rationale                                                                                                                                                                                                                                                     |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hex grid, not square                                             | Uniform distance in all directions, natural organic shapes, clean rotation (6 orientations), eliminates diagonal ambiguity                                                                                                                                    |
| Toroidal grid topology                                           | Eliminates edge effects and corner asymmetry. Every tile has exactly 6 neighbors. Organisms cannot be trapped against walls. Simulates an unbounded surface without infinite space.                                                                           |
| Organisms are cells on the grid, not floating agents             | Cell-level destruction requires physical grid positions. Growth, combat, and regeneration are all grid operations. Surface area is trivially measured. GPU-friendly (uniform grid update).                                                                    |
| Organisms are also logical entities with shared energy and brain | Cells alone can't coordinate. Shared energy pool and brain enable organism-level strategy while cells handle local mechanics.                                                                                                                                 |
| CPPN for body plan encoding                                      | Compact genome, natural symmetry support, continuous/smooth fitness landscape, temporal development via age input. Superior to direct encoding (too large, not evolvable) and GRN (too complex to evolve from scratch).                                       |
| Fixed CPPN topology to start                                     | Simple, GPU-batchable, straightforward crossover. Upgrade to NEAT later if needed.                                                                                                                                                                            |
| Rule-based brain to start                                        | Eliminates brain-body coupling problem, no training needed, sufficient for diverse strategies, simple to debug. Upgrade to NN later.                                                                                                                          |
| Body-invariant sensor abstraction                                | Decouples brain from body. Body mutations don't break the brain. More sensor cells = better accuracy, not different input format.                                                                                                                             |
| Evolution between matches, not during                            | Clean separation of fitness evaluation (simulation) and evolution (outer loop). Offspring in-match are clones. Simpler to implement and reason about.                                                                                                         |
| Tournament-based evolution                                       | Well-understood, configurable, supports diverse matchmaking strategies. Fitness is relative (competitive), not absolute.                                                                                                                                      |
| Disconnected cell clusters die                                   | Avoids complex multi-entity splitting. Keeps organism integrity simple. Severing an organism is tactically meaningful (kills the smaller piece).                                                                                                              |
| Kotlin orchestrator + Python simulation via Jep                  | Best of both worlds: Kotlin's type safety and expressiveness for the complex orchestration logic; Python's GPU ecosystem (Taichi) for the simulation kernel. Jep avoids subprocess overhead.                                                                  |
| Docker deployment                                                | Reproducible environment. Solves Jep/CUDA dependency management. GPU passthrough on Linux/Windows, CPU fallback on Mac.                                                                                                                                       |
| CPU-first development                                            | Get architecture right before optimizing. Taichi CPU mode is fast enough for development. GPU port is a well-defined later phase.                                                                                                                             |
| Deterministic simulation                                         | Enables exact replay of matches. Required for debugging and visualization. Achieved via seeded PRNG and deterministic tiebreaking.                                                                                                                            |
| Terrain as separate grid layer                                   | Terrain is a static background layer; organisms/resources sit on top. Minimal GPU cost (one extra int per tile). Design into data structure from Phase 1, implement terrain types in Phase 8. Enables aquatic/terrestrial specialization and niche diversity. |

---

## 14. Performance & Optimization Strategy

### CPU Phase Targets
- Grid size: 512×512 (~260K tiles) for development.
- Organism count: up to 1,000 simultaneously.
- Taichi CPU backend (LLVM) with flat array data layout.

### GPU Phase Targets (RTX 3090)
- Grid size: 2048×2048 (~4.2M tiles).
- Organism count: up to 10,000 simultaneously.
- Target: optimize ticks/second.
- Taichi CUDA backend.
- Spatial hashing for neighbor queries.
- Batch CPPN evaluation as parallel kernel.

### Memory Budget (GPU)
- Total: well within 24GB VRAM of RTX 3090.

---

## 17. Open Questions & Future Considerations

Items explicitly deferred for later decision:

- **NN brain architecture details**: hidden layer sizes, activation functions, recurrent connections. Decide when upgrading from rule-based brain.
- **NEAT integration**: if/when fixed-topology CPPN proves limiting. Drop-in replacement for body plan genome.
- **Chemical signaling system**: how organisms emit and sense chemicals. Adds communication and territory marking but increases simulation complexity.
- **Multi-resource types**: light, organic matter, minerals, etc. Enables niche specialization but requires balancing multiple resource economies.
- **Terrain features**: barriers, resource patches, light gradients are planned (Phase 8). Open questions: optimal terrain map generation algorithms, balance of terrain type frequencies, whether terrain should be static or slowly change during a match.
- **Speciation tracking**: phylogenetic trees, genome clustering, species identification during evolution. Useful for analysis but not core to simulation.
- **Distributed evolution**: running evolution across multiple machines/GPUs. Only needed if single-GPU throughput is insufficient.
