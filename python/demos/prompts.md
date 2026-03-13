# AI Video Generation Prompts

Use these with **Kling 2.1** (start+end frame image-to-video) or **Runway Gen-3 Alpha** (image-to-video with text).

## Visual Style (apply to all scenes)

> Hyper-realistic 3D render, Minecraft voxel art style, hexagonal cells as small translucent cubes on a dark carbon-fiber substrate, microscope view, bioluminescent glow emanating from living cells, subtle depth-of-field blur at edges, volumetric lighting from above, wet organic sheen on cell surfaces

## Step 1: Create 3D Reference

Take the first PNG frame of each scene and generate a 3D version with Midjourney/DALL-E:

> A top-down microscopic view of hexagonal organisms made of glowing voxel cubes on a dark grid substrate. Each cell type has a distinct color: green for photosynthetic, red for mouths, purple for flagella, orange for spikes, beige for skin. Minecraft-style 3D blocks, bioluminescent, shallow depth of field, dark background, cinematic lighting

---

## Scene 1: "Genesis" (5 seconds)

**Start frame**: Single glowing green cell on dark grid
**End frame**: Full radial flower pattern of green/beige cells

**Motion prompt**:
> Slow cinematic zoom-in on a single bioluminescent green cube cell sitting on a dark hexagonal grid. The cell pulses with light, then begins dividing outward in a radial bloom pattern. New cells emerge symmetrically in all six directions, creating concentric rings — inner cells glow bright green (photosynthetic), outer cells warm beige (skin). Camera slowly rotates 15 degrees during growth. Volumetric light rays. Microscope depth of field.

**Camera**: Slow zoom in + 15° rotation

---

## Scene 2: "The Hunt" (7 seconds)

**Start frame**: Predator (red/purple) far from prey (green/beige)
**End frame**: Predator adjacent to prey, prey missing cells

**Motion prompt**:
> A sleek bilateral predator organism (red mouth cells at front, purple flagella on sides) propels itself across a dark hexagonal grid toward a rounder prey organism (green core, beige shell). The predator moves with visible thrust from its flagella, leaving a faint wake. The prey detects the approach and begins fleeing but is slower. Camera pans to follow the chase. As the predator reaches the prey, its red mouth cells begin consuming the prey's outer cells one by one, each consumed cell releasing a small burst of particle energy. Bioluminescent glow, microscope view, cinematic.

**Camera**: Tracking pan following the predator

---

## Scene 3: "Spike Battle" (5 seconds)

**Start frame**: Two spike warriors facing each other from opposite sides
**End frame**: Both warriors in contact, cells destroyed, debris

**Motion prompt**:
> Two armored organisms with protruding orange spike extensions charge toward each other across a dark hex grid. They collide at the center in a burst of particle effects. Orange spikes pierce through cells on contact, destroying them — fragments of destroyed cells scatter as small glowing particles. Both organisms sustain damage, losing cells from their edges. The collision is dramatic but in slow-motion microscope scale. Volumetric lighting, bioluminescent sparks on impact.

**Camera**: Static center frame, slight zoom on collision moment

---

## Scene 4: "Life Cycle" (5 seconds)

**Start frame**: Single seed cell
**End frame**: Parent organism with 1-2 offspring nearby

**Motion prompt**:
> A single cell on a dark hex grid grows into a small multicellular organism over time — first green photosynthetic cells, then pink tissue, then beige skin border. Once mature, the organism pulses with accumulated energy (bright internal glow), then buds off a single daughter cell to one side. The daughter cell begins its own growth cycle, forming a miniature copy of the parent. Camera slowly pulls back to reveal both organisms growing side by side. Warm bioluminescent lighting, microscope depth of field.

**Camera**: Start close, slow pull-back to reveal reproduction

---

## Scene 5: "Ecosystem" (8 seconds)

**Start frame**: Multiple organisms spread across large grid — plants, herbivores, predator
**End frame**: Dynamic scene with interactions visible

**Motion prompt**:
> A thriving microscopic ecosystem on a dark hexagonal grid. Four large radial plant organisms (bright green) photosynthesize in the corners, slowly growing outward. Three medium herbivore organisms (green core, beige shell, purple flagella) roam between plants, grazing on scattered food cells (bright green dots). A single aggressive predator (red fronted, purple flanked) stalks and pursues the nearest herbivore. Camera slowly pulls back to reveal the full ecosystem. Plants pulse gently with photosynthetic energy. Herbivores leave subtle movement trails. The predator moves noticeably faster. Bioluminescent, volumetric lighting, cinematic microscope view.

**Camera**: Wide establishing shot, very slow zoom out

---

## Post-Processing Notes

1. Generate each scene as a 5-10 second clip at 24fps
2. Stitch with crossfade transitions (0.5s each)
3. Add title cards between scenes:
   - "GENESIS" / "THE HUNT" / "SPIKE BATTLE" / "LIFE CYCLE" / "ECOSYSTEM"
4. Background music: ambient electronic, similar to "Cells" by Ludovico Einaudi
5. Total runtime target: ~30 seconds
6. Export at 1920x1080 or 4K for quality
