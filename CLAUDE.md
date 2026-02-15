# CLAUDE.md — Artificial Life Evolution Simulator

## 1. Project Overview

This project is a **2D artificial life evolution simulator** where genetically distinct organisms compete for surface area dominance on a hex grid battlefield. Think of it as simulating a petri dish: organisms are placed on the grid, consume resources, grow, reproduce, fight, and the genome that occupies the most surface area (cell count) at the end of a match wins.

### Core Goals
- **Diverse meta-strategies must be viable**: predators, resource gatherers, plants/growers/amoebas, armored fortresses, fast swarming reproducers, parasites, etc.
- **The genotype encodes everything**: organism shape (body plan), features (cell types), and behavioral parameters.
- **Rich feature set**: ~100 cell types with tunable parameters (spikes, mouths, teeth, armor, skin, eyes, feet/flagella, photosynthetic membranes, chemical sensors, signal emitters, etc.) enabling different strategies to emerge.
- **Organisms are physically embodied on the grid**: individual cells can be destroyed, severed, or regrown. A bite removes a specific cell. Shape matters tactically.
- **Evolution happens between matches**, not during. The simulation is a fitness evaluation function. Tournament-based evolutionary algorithm drives the outer loop.
- **Deterministic simulation**: given the same inputs and random seed, a match replays identically. This enables batch evolution runs followed by replay visualization of interesting matches.

### Design Philosophy
- **Modularity is paramount**: genome encoding, brain architecture, evolution strategy, body plan generation, and sensor systems are all swappable modules behind clean interfaces. We are uncertain which approaches will produce the best emergent behavior, so the system must support experimentation.
- **CPU-first, GPU-later**: initial implementation runs on CPU. Architecture uses flat arrays and pure-function cell updates so the hot simulation loop can be ported to GPU (Taichi) without restructuring.
- **Build incrementally**: each implementation phase produces a runnable, testable system. Never more than one phase away from something you can observe.

---

## 2. Tech Stack & Environment

### Languages
- **Kotlin (JVM)**: Primary language. Owns the orchestrator layer — evolution loop, tournament management, genome management, configuration, match coordination, serialization, CLI/entry point.
- **Python**: Simulation computation layer. Owns the GPU-acceleratable simulation kernel, CPPN evaluation, and any ML workloads. Called from Kotlin via Jep (Java Embedded Python).

### Coding Standards:
- Always use 4 spaces for indentation.

#### **Python**:
 - ALWAYS include type hints and type annotations were possible.

### Key Libraries & Frameworks
- **Jep (Java Embedded Python)**: Embeds a Python interpreter inside the JVM process. Kotlin calls Python functions directly with near-zero overhead. NumPy arrays can be shared with minimal copying. This avoids subprocess/serialization overhead of a client-server architecture.
- **Taichi Lang** (Python): GPU-acceleratable simulation kernels. Compiles Python-like code to CUDA/OpenCL/CPU-LLVM. Supports hex grid simulations natively. CPU mode is fast enough for development; GPU mode for production evolution runs.
- **kotlinx.serialization**: Genome and configuration persistence (JSON format).
- **kotlinx.coroutines**: Parallel match execution on CPU cores.
- **Multik or ejml**: Matrix operations for neural network evaluation if/when NN brains are implemented.
- **Kotest or JUnit 5**: Testing framework.
- **Visualization**: TBD — candidates are OpenRNDR (Kotlin-native), a web-based viewer, or Taichi's built-in GUI. Deferred to later phase.

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
│  ┌─────────────┐  ┌──────────────┐             │
│  │  Evolution   │  │  Tournament  │             │
│  │  Strategy    │  │  Manager     │             │
│  └──────┬──────┘  └──────┬───────┘             │
│         │                │                      │
│  ┌──────┴──────┐  ┌──────┴───────┐             │
│  │  Genome     │  │  Match       │             │
│  │  Manager    │  │  Coordinator │             │
│  └─────────────┘  └──────┬───────┘             │
│                          │ Jep                  │
├──────────────────────────┼──────────────────────┤
│  PYTHON SIMULATION (embedded via Jep)           │
│                          │                      │
│  ┌──────────────┐  ┌────┴──────────┐           │
│  │  Body Plan   │  │  Simulation   │           │
│  │  Providers   │  │  Engine       │           │
│  │  (CPPN etc)  │  │  (Taichi)     │           │
│  └──────────────┘  └──────┬────────┘           │
│                           │                     │
│  ┌──────────────┐  ┌─────┴─────────┐           │
│  │  Brain       │  │  Hex Grid     │           │
│  │  Providers   │  │  (core data)  │           │
│  └──────────────┘  └───────────────┘           │
└─────────────────────────────────────────────────┘
```

### Separation of Concerns
- **Kotlin** owns: evolution logic, tournament brackets, genome CRUD, population management, configuration, persistence, entry point, replay management.
- **Python** owns: simulation execution, hex grid state, cell update kernels, CPPN evaluation, sensor aggregation, brain evaluation, combat resolution, energy accounting.
- **The boundary**: Kotlin sends match configuration + serialized genomes to Python. Python runs the match and returns fitness scores + optional replay data. This is a function call via Jep, not a network request.

### Six Swappable Modules
All behind clean interfaces. Implementations can be swapped via configuration:

1. **Body Plan Provider** — genome → cell layout. Implementations: CPPN, lookup table, L-system, GRN.
2. **Brain Provider** — sensor inputs → action outputs. Implementations: rule-based, fixed NN, NEAT NN, behavior tree, hardcoded script.
3. **Sensor Aggregator** — raw cell sensor data → abstracted brain input vector. Implementations: basic (nearest-entity summaries), detailed (full spatial map).
4. **Genome Representation** — the heritable data structure. Implementations: flat float vector, NEAT graph genome, composite genome.
5. **Evolution Strategy** — match results → next generation. Implementations: tournament selection, Elo-rated, MAP-Elites, manual selection.
6. **Match Configuration** — data-driven arena setup. Grid size, tick limit, resource distribution, competitor count, placement strategy.

---

## 4. Core Domain Model

### Hex Grid
- **Coordinate system**: Axial coordinates (q, r). Each hex has exactly 6 equidistant neighbors.
- **Toroidal topology**: the grid wraps in both axes — coordinate `width` maps back to `0`, and `-1` maps to `width-1` (same for height). All coordinate functions (`neighbors`, `hex_distance`, `line_of_sight`) account for wrapping and return the shortest toroidal path. There are no edges or corners; every tile has exactly 6 neighbors.
- **Why hex**: eliminates diagonal movement ambiguity of square grids, produces natural organic shapes, rotation is a clean coordinate transform (6 orientations at 60° increments), uniform distance in all directions.
- **Grid contents**: each hex tile has two layers: a **terrain type** (ground, water, rock, fertile soil, toxic, etc.) and **contents** (empty, resource, or organism cell). Terrain is the floor; organisms and resources sit on top. A water tile can contain a swimming organism's cell. A rock tile is impassable.
- **Tile state**: terrain type enum, contents enum, cell type enum, owner organism ID.

### Organisms
An organism is a **coherent entity** composed of contiguous cells on the hex grid. It is NOT an agent floating above the grid — its cells physically occupy hex tiles.

Each organism has:
- **Unique ID**: assigned at birth.
- **Genome**: heritable data encoding body plan, brain parameters, and metabolic parameters.
- **Cell set**: the set of hex tiles this organism's cells occupy. Cells have types (mouth, spike, armor, photosynthetic, eye, flagella, etc.).
- **Energy pool**: shared across all cells. Contributed to by photosynthetic cells and eating. Drained by cell maintenance, movement, growth, reproduction.
- **Brain state**: the current output of the brain (movement direction, grow/attack/reproduce decisions). Updated each tick from sensor inputs.

### Cells (~100 types, examples)
Each cell type has defined interactions with neighbors and the environment:

**Structural**: skin (basic, cheap), armor (damage resistant, expensive), membrane (permeable, allows resource absorption).

**Offense**: mouth (consumes adjacent food/foreign cells), spike (damages adjacent foreign cells passively), teeth (enhances mouth damage).

**Sensing**: eye (detects entities at range along line-of-sight), chemical sensor (detects environmental chemical gradients), touch sensor (detects adjacent contact).

**Locomotion**: flagella (enables whole-organism movement), cilia (enables slow amoeboid movement), pseudopod (enables directional extension).

**Metabolism**: photosynthetic membrane (generates energy from light), root (absorbs nutrients from resource tiles), storage vacuole (increases max energy capacity).

**Reproduction**: reproductive cell (required to produce offspring seeds).

**Signaling**: signal emitter (broadcasts chemical signals), pigment (visual signaling to other organisms).

The full cell type roster is a design/balance parameter — start with ~20 core types and expand.

### Energy System
- Every cell costs maintenance energy per tick (varies by type: armor costs more than skin).
- Photosynthetic cells generate energy proportional to light availability. Terrain modifies yield (fertile soil boosts, water reduces).
- Mouth cells transfer energy from consumed food/cells to the organism's pool.
- Movement costs energy proportional to organism mass (cell count). Terrain modifies cost (water may reduce movement cost for aquatic organisms).
- Growth (adding a new cell) costs energy dependent on cell type.
- Reproduction costs energy proportional to the new cells generated, + energy transferred to offspring as starting energy, + base reproduction cost.
- If energy reaches zero, cells begin dying (starvation).

---

## 5. Module Interfaces

These interfaces define the contracts between swappable modules. Implementations vary but inputs/outputs are fixed.

### Body Plan Provider
```
Interface: BodyPlanProvider
Input:
  - hex_position: (q, r) relative to organism center
  - organism_age: int (developmental time steps since birth)  
  - neighbor_context: array of 6 neighbor states (filled/empty, cell types)
  - genome_data: opaque blob (CPPN weights, lookup table, etc.)
Output:
  - should_cell_exist: bool
  - cell_type: enum (or null if no cell)
  - growth_priority: float (how urgently to grow here)
  - growth_direction_bias: float[6] (preference for expanding in each hex direction)
```

### Brain Provider
```
Interface: BrainProvider
Input (body-invariant sensor vector):
  - nearest_food_direction: float (angle)
  - nearest_food_distance: float (0-1 normalized)
  - nearest_threat_direction: float
  - nearest_threat_distance: float
  - nearest_kin_direction: float
  - nearest_kin_distance: float
  - chemical_gradient: float[N] (environmental signals)
  - terrain_ahead: float[T] (terrain type distribution in movement direction)
  - internal_energy_fraction: float (0-1)
  - organism_size: float (0-1 normalized)
  - organism_age: int
  - recent_damage: float (cells lost recently, 0-1)
  - recurrent_state: float[M] (memory neurons, fed back from previous tick)
Output:
  - movement_direction: float (angle, or null for no movement)
  - movement_speed: float (0-1)
  - grow: bool
  - growth_direction_bias: float (angle)
  - reproduce: bool
  - attack: bool
  - chemical_emission: float[N]
  - recurrent_state_out: float[M]
```

### Sensor Aggregator
```
Interface: SensorAggregator
Input:
  - organism's sensor cells (positions, types, orientations)
  - local world state visible to those cells
Output:
  - the body-invariant sensor vector (as defined in BrainProvider input)
```

### Genome Representation
```
Interface: Genome
Methods:
  - mutate(mutation_rate: float) -> Genome
  - crossover(other: Genome) -> Genome
  - serialize() -> bytes
  - deserialize(bytes) -> Genome
  - get_body_plan_data() -> opaque blob for BodyPlanProvider
  - get_brain_data() -> opaque blob for BrainProvider  
  - get_metabolic_params() -> MetabolicParams
```

### Evolution Strategy
```
Interface: EvolutionStrategy
Methods:
  - initialize_population(size: int) -> List<Genome>
  - select_match_participants(population: List<Genome>) -> List<Genome>
  - record_match_result(participants: List<Genome>, scores: List<float>)
  - produce_next_generation(population: List<Genome>) -> List<Genome>
```

### Match Configuration
```
Data class: MatchConfig
Fields:
  - grid_width: int
  - grid_height: int  
  - tick_limit: int
  - resource_distribution: ResourceConfig (initial placement, regeneration rate)
  - terrain_map: TerrainConfig (terrain type generation strategy and parameters)
  - competitor_count: int
  - initial_placement: PlacementStrategy (spaced, random, clustered)
  - random_seed: long (for determinism)
  - light_level: float (affects photosynthesis yield)
```

---

## 6. Simulation Tick Mechanics

Each tick executes the following steps **in this exact order** (order matters for determinism and balance):

### Step 1: Resource Regeneration
- Food tiles regenerate at a configured rate. Terrain modifies regeneration (fertile soil has higher rate, rock has none, water may support different resource types).
- Light level is constant (or varies by region if configured).
- Chemical signals diffuse and decay.

### Step 2: Sensor Aggregation
- For each organism, collect what its sensor cells detect.
- Eye cells: ray-cast (or hex-walk) in their facing direction to detect entities.
- Chemical sensors: read local chemical concentrations.
- Touch sensors: detect adjacent foreign cells.
- Terrain awareness: cells detect the terrain type of their own tile and adjacent tiles (available to brain for movement decisions).
- Compile into the body-invariant sensor input vector.

### Step 3: Brain Evaluation
- For each organism, pass sensor vector through brain (rule-based or NN).
- Receive action outputs: movement, growth, reproduction, attack decisions.

### Step 4: Action Execution
Process in sub-steps to avoid order-dependent conflicts:

**4a. Movement**: Organisms with flagella/locomotion cells attempt to move. Two movement modes:
- *Flagella shift*: if organism has sufficient flagella relative to mass, shift ALL cells one hex in the movement direction (atomic operation). Costs energy proportional to mass. Terrain modifies cost (e.g., water reduces cost for aquatic-capable organisms).
- *Amoeboid*: grow cells at leading edge, retract cells at trailing edge. Slower, lower energy cost, doesn't require flagella cells.
- **Terrain constraints**: organisms cannot move into or through rock tiles. Movement into water tiles only succeeds if the organism has water-compatible cell types. Terrain type of destination tile is checked before movement resolves.

**4b. Growth**: Border cells of growing organisms attempt to fill adjacent empty hexes. Cell type determined by body plan provider (CPPN query or template lookup). Costs energy per new cell. Growth rate limited by energy availability and configured maximum. **Terrain constraints**: growth into a tile is blocked if the terrain type forbids that cell type (e.g., armor cannot grow on water tiles, roots cannot grow on rock). Growth into rock tiles is always blocked.

**4c. Reproduction**: Organisms that choose to reproduce and have sufficient energy spawn a **seed cell** in an adjacent empty hex. The seed carries a copy of the parent's genome (no mutation during match — clones only). Parent's energy is reduced by the reproduction cost; offspring receives a configured starting energy amount.

**4d. Attack activation**: Mouth cells adjacent to foreign cells attempt to consume them. Spike cells passively damage adjacent foreign cells.

### Step 5: Combat Resolution
- For each mouth-vs-foreign-cell interaction: calculate damage based on mouth strength, teeth bonuses, target armor. If damage exceeds cell durability, the target cell is **destroyed** (removed from grid). Energy from the destroyed cell transfers to the attacker's pool.
- For each spike-vs-foreign-cell interaction: similar but passive (no energy transfer, just damage).
- Destroyed cells leave empty hexes. The owning organism loses those cells but remaining cells persist.
- **Disconnection check**: if cell destruction severs an organism into disconnected components, the smaller component(s) die (cells removed, energy lost). This keeps organism integrity without complex multi-entity splitting logic.

### Step 6: Energy Accounting
- Each cell costs maintenance energy (type-dependent). Terrain may modify maintenance costs (e.g., toxic terrain adds damage per tick to cells on it).
- Photosynthetic cells generate energy. Terrain modifies yield (fertile soil multiplier, water reduction).
- Root cells on fertile soil generate bonus energy.
- Net energy change applied to each organism's pool.
- Organisms at zero energy begin starvation: remove outermost cells until energy stabilizes or organism dies.

### Step 7: Death and Cleanup
- Organisms with zero cells are removed.
- Dead cell positions become food tiles (energy recycling).

### Step 8: Score Update
- Count cells per genome (not per organism — clones of the same genome contribute to the same score).
- Record for fitness evaluation.

### Determinism Guarantee
- All random decisions use a seeded PRNG.
- Conflict resolution (two organisms trying to grow into the same empty hex) resolved by deterministic tiebreaking (e.g., lower organism ID wins).
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
1. When cells are destroyed (combat, starvation), the organism loses those cells.
2. Remaining border cells at the wound site detect empty adjacent hexes.
3. These border cells query the body plan provider: "should a cell exist here?"
4. If yes, and the organism has energy, the cell regrows.
5. Regeneration uses **the same growth mechanism** as initial development — no separate regeneration system needed.
6. Regeneration may not perfectly restore the original shape (depends on body plan encoding and local context), which is biologically realistic.

### Death
An organism dies when:
- All cells are destroyed.
- Energy reaches zero and all cells are consumed by starvation.
- A disconnection event leaves no viable connected component.

Dead organisms' cells become food tiles on the grid.

---

## 8. Genome & Body Plan

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
      - Record surface area (cell count) per genome at match end.
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

## 12. Implementation Phases

Build in this order. Each phase produces a runnable, testable system.

### Phase 1: Hex Grid & Core Data Structures
- Implement hex grid with axial coordinates in Python (Taichi-compatible data layout).
- Tile state: terrain type, cell type, organism ID, energy. **Include terrain layer in data structure from day one** (initialized to "ground" everywhere) so adding terrain types later is purely additive.
- Hex coordinate math: neighbors, distance, line-of-sight.
- Basic visualization, simple web server to render a hex grid state. React.js that interacts with Kotlin. 
- **Deliverable**: a hex grid you can programmatically place cells on and display.

### Phase 2: Manual Organisms & Basic Simulation
- Implement organism entity (ID, cell set, energy pool).
- Lookup-table body plan provider.
- Hardcoded/scripted brains for test organisms.
- Implement the full tick loop: resource regen, sensor aggregation (basic), brain eval (scripted), action execution (growth, movement), combat resolution, energy accounting, death/cleanup.
- Create 3-4 manual test organisms: a stationary plant, a moving herbivore, an armored defender, a predator.
- Run matches between them. Verify mechanics work: growth, combat, movement, energy flow, starvation, regeneration.
- **Deliverable**: a working petri dish simulator with hand-designed creatures competing.

### Phase 3: Rule-Based Brain & Sensor Aggregation
- Implement the body-invariant sensor aggregator.
- Implement rule-based brain with configurable parameters.
- Replace hardcoded brains on test organisms with rule-based brains (hand-tune parameters to replicate intended behavior).
- Verify that organisms behave reasonably with the rule-based system.
- **Deliverable**: autonomous organisms competing with parameterized behavior.

### Phase 4: Genome, Mutation, and Basic Evolution
- Implement the genome data structure (float vector encoding rule parameters + metabolic parameters).
- Implement mutation and crossover operators.
- Implement simple tournament evolution loop (Kotlin orchestrator).
- Wire up Jep bridge: Kotlin sends genomes to Python, Python runs match, returns scores.
- Use lookup-table body plans initially (randomly generated simple shapes, or evolve just brain/metabolism parameters with fixed bodies).
- Run evolution for many generations. Verify fitness improves over time.
- **Deliverable**: working evolutionary loop producing increasingly fit organisms.

### Phase 5: CPPN Body Plan
- Implement CPPN evaluation in Python.
- Add CPPN weights and activation functions to genome.
- Implement developmental growth from seed using CPPN queries.
- Implement symmetry modes.
- Evolve body plans along with brain parameters and metabolism.
- **Deliverable**: organisms with evolved body shapes competing. This is where it gets interesting.

### Phase 6: GPU Acceleration
- Port the simulation tick kernel to Taichi.
- Spatial hashing for neighbor queries.
- Batch CPPN evaluation on GPU.
- Benchmark: measure ticks/second at various grid sizes and organism counts.
- **Deliverable**: GPU-accelerated simulation running significantly faster than CPU.

### Phase 7: Visualization & Replay
- Implement match replay recording (serialize grid state each tick, or record actions for deterministic replay).
- Build a visualization tool to watch replays (OpenRNDR, web viewer, or Taichi GUI).
- **Deliverable**: ability to watch evolved organisms compete visually.

### Phase 8: Refinement & Expansion
- Expand cell type roster toward ~100 types.
- **Implement terrain types**: water, rock, fertile soil, toxic, etc. with interaction rules defined in config. Implement terrain map generation strategies (procedural generation of lakes, rock formations, fertile patches). Terrain data structure is already in grid from Phase 1.
- Tune balance parameters (cell costs, damage values, energy rates, terrain modifiers).
- Experiment with evolution strategies (MAP-Elites for diversity, Elo matchmaking).
- Experiment with NN brains as upgrade from rule-based.
- Add more visualization features (genome browser, phylogenetic trees, statistics).

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
- Target: 50-100 ticks/second on modern CPU.
- Taichi CPU backend (LLVM) with flat array data layout.

### GPU Phase Targets (RTX 3090)
- Grid size: 2048×2048 (~4.2M tiles).
- Organism count: up to 10,000 simultaneously.
- Target: 500-2000 ticks/second.
- Taichi CUDA backend.
- Spatial hashing for neighbor queries.
- Batch CPPN evaluation as parallel kernel.

### Data Layout for GPU Readiness
Even during CPU phase, use data structures that map cleanly to GPU:
- **Flat arrays** for grid state (not objects-per-tile).
- **Struct-of-arrays** layout: separate arrays for cell_type[], organism_id[], etc. rather than array of cell structs. This is more cache-friendly and maps to GPU memory coalescing.
- **Fixed-size organism data**: cap organism data at fixed sizes (e.g., max 32×32 body plan). Pad smaller organisms. Enables uniform GPU processing.
- **No per-cell heap allocation**: everything in pre-allocated arrays.

### Memory Budget (GPU)
- Grid state: 4.2M tiles × ~20 bytes (including terrain layer) = ~84MB
- Organism state: 10K organisms × ~4KB each = ~40MB
- CPPN weights: 10K organisms × ~1600 bytes = ~16MB
- Working memory, spatial hash, etc.: ~100MB
- Total: well within 24GB VRAM of RTX 3090.

---

## 15. Project Directory Structure

```
alife-simulator/
├── CLAUDE.md                          # This file
├── build.gradle.kts                   # Gradle build with Kotlin DSL
├── settings.gradle.kts
├── docker/
│   ├── Dockerfile                     # JDK + Python + Jep + Taichi + CUDA
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── kotlin/                            # Kotlin sources (orchestrator)
│   └── src/
│       ├── main/kotlin/
│       │   ├── Main.kt                # Entry point
│       │   ├── config/
│       │   │   ├── MatchConfig.kt
│       │   │   ├── EvolutionConfig.kt
│       │   │   └── SimulatorConfig.kt
│       │   ├── genome/
│       │   │   ├── Genome.kt          # Interface
│       │   │   ├── FloatVectorGenome.kt
│       │   │   ├── CompositeGenome.kt
│       │   │   ├── Mutation.kt
│       │   │   └── Crossover.kt
│       │   ├── evolution/
│       │   │   ├── EvolutionStrategy.kt  # Interface
│       │   │   ├── TournamentSelection.kt
│       │   │   ├── EloTournament.kt
│       │   │   └── MapElites.kt
│       │   ├── tournament/
│       │   │   ├── TournamentManager.kt
│       │   │   ├── MatchCoordinator.kt
│       │   │   └── FitnessAggregator.kt
│       │   ├── bridge/
│       │   │   ├── JepBridge.kt       # Kotlin↔Python interface via Jep
│       │   │   └── DataConversion.kt  # Kotlin types ↔ Python dicts/arrays
│       │   └── persistence/
│       │       ├── GenomeDatabase.kt
│       │       ├── ReplayStore.kt
│       │       └── Checkpoint.kt
│       └── test/kotlin/
│           └── ...
├── python/                            # Python sources (simulation)
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── engine.py                  # Main simulation entry point (called from Kotlin)
│   │   ├── hex_grid.py                # Hex grid data structure, coordinate math
│   │   ├── world.py                   # Tick loop orchestration
│   │   ├── combat.py                  # Damage resolution
│   │   ├── organism.py                # Organism entity management
│   │   └── movement.py               # Movement mechanics
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── body_plan.py               # BodyPlanProvider ABC
│   │   ├── brain.py                   # BrainProvider ABC
│   │   └── sensor.py                  # SensorAggregator ABC
│   ├── body_plans/
│   │   ├── __init__.py
│   │   ├── cppn.py                    # CPPN body plan provider
│   │   └── lookup_table.py            # Manual/template body plan
│   ├── brains/
│   │   ├── __init__.py
│   │   ├── rule_based.py              # Rule-based brain with evolved parameters
│   │   ├── scripted.py                # Hardcoded NPC brains
│   │   └── fixed_nn.py               # Fixed-topology neural network brain
│   ├── sensors/
│   │   ├── __init__.py
│   │   └── basic_aggregator.py        # Body-invariant sensor aggregation
│   └── tests/
│       └── ...
├── config/
│   ├── default_match.json
│   ├── default_evolution.json
│   ├── cell_types.json                # Cell type definitions, costs, interactions
│   └── terrain_types.json             # Terrain type definitions, interaction rules
└── data/                              # Mounted volume for persistence
    ├── genomes/
    ├── replays/
    └── checkpoints/

```

---

## 16. Cell Type Balance Configuration

Cell types and their interactions should be defined in a **data file** (`config/cell_types.json`), not hardcoded. This enables rapid balance tuning without code changes.

Example schema:
```json
{
  "cell_types": {
    "skin": {
      "maintenance_cost": 1.0,
      "growth_cost": 5.0,
      "durability": 10,
      "description": "Basic structural cell. Cheap, low durability."
    },
    "armor": {
      "maintenance_cost": 3.0,
      "growth_cost": 15.0,
      "durability": 50,
      "description": "High durability structural cell. Expensive."
    },
    "mouth": {
      "maintenance_cost": 2.0,
      "growth_cost": 10.0,
      "durability": 8,
      "bite_damage": 15,
      "energy_transfer_rate": 0.8,
      "description": "Consumes adjacent foreign cells. Transfers energy."
    },
    "photosynthetic": {
      "maintenance_cost": 1.5,
      "growth_cost": 8.0,
      "durability": 5,
      "energy_generation": 2.0,
      "description": "Generates energy from light. Fragile."
    }
  }
}
```

All numeric values are tunable balance parameters. Start with a small roster (~20 types), playtest, expand.

### Terrain Type Configuration

Terrain types and their interaction rules are also defined in config (`config/cell_types.json` or a separate `config/terrain_types.json`). This enables experimenting with terrain effects without code changes.

Example schema:
```json
{
  "terrain_types": {
    "ground": {
      "passable": true,
      "description": "Default terrain. No special effects."
    },
    "water": {
      "passable": true,
      "allowed_cell_types": ["membrane", "flagella", "cilia", "mouth", "eye", "photosynthetic", "skin"],
      "blocked_cell_types": ["root", "armor", "spike"],
      "movement_cost_multiplier": 0.7,
      "photosynthesis_multiplier": 0.5,
      "resource_regen_multiplier": 0.3,
      "description": "Aquatic terrain. Only lightweight cell types can exist here. Faster movement, less light."
    },
    "rock": {
      "passable": false,
      "description": "Impassable barrier. No growth, no movement, no resources."
    },
    "fertile_soil": {
      "passable": true,
      "photosynthesis_multiplier": 2.0,
      "root_energy_multiplier": 3.0,
      "resource_regen_multiplier": 2.0,
      "description": "Rich terrain. Boosts photosynthesis and root energy. Higher resource regeneration."
    },
    "toxic": {
      "passable": true,
      "damage_per_tick": 2.0,
      "resource_regen_multiplier": 0.0,
      "description": "Damages cells each tick. No resources grow here. Passable but costly."
    }
  }
}
```

**Terrain map generation** is part of MatchConfig. Strategies include: uniform (all ground), random noise, perlin/simplex noise (natural-looking biomes), hand-designed templates, or hybrid approaches. Varying terrain across matches within a generation encourages generalist organisms that can handle diverse environments.

---

## 17. Open Questions & Future Considerations

Items explicitly deferred for later decision:

- **Visualization technology**: create a web view for overall status of simulations, and a visualizer for one match.
- **NN brain architecture details**: hidden layer sizes, activation functions, recurrent connections. Decide when upgrading from rule-based brain.
- **NEAT integration**: if/when fixed-topology CPPN proves limiting. Drop-in replacement for body plan genome.
- **Chemical signaling system**: how organisms emit and sense chemicals. Adds communication and territory marking but increases simulation complexity.
- **Multi-resource types**: light, organic matter, minerals, etc. Enables niche specialization but requires balancing multiple resource economies.
- **Terrain features**: barriers, resource patches, light gradients are planned (Phase 8). Open questions: optimal terrain map generation algorithms, balance of terrain type frequencies, whether terrain should be static or slowly change during a match.
- **Speciation tracking**: phylogenetic trees, genome clustering, species identification during evolution. Useful for analysis but not core to simulation.
- **Distributed evolution**: running evolution across multiple machines/GPUs. Only needed if single-GPU throughput is insufficient.
