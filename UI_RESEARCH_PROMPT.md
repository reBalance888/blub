# Deep Research Prompt: BLUB Ocean Viewer UI/UX Redesign

## Your Role

You are a senior UI/UX designer and creative director specializing in real-time data visualization, game-like interfaces, and emergent behavior visualization. Your task is to deeply research and provide concrete, actionable recommendations for redesigning the visual layer of BLUB — a living ocean simulation where AI agents develop language through survival pressure.

## Project Context

**BLUB** is a Proof of Communication crypto experiment. An ocean simulation where lobster-agents mine BLUB tokens by developing useful communication. The simulation runs server-side (Python/FastAPI), the viewer is a single-page HTML5 Canvas app connected via WebSocket.

**The bigger vision:** Eventually, external developers will connect their own agents to a public server. The viewer is the PUBLIC FACE of the project — it needs to be visually compelling enough to make people want to participate. Think of it as a living aquarium that tells a story of emergent intelligence.

**Key entities to visualize:**
- **Lobsters** (33 agents) — the main actors, 3 tiers: shrimp (gray, beginner), lobster (orange-red, mid), kraken (purple, elite)
- **Rifts** — resource nodes (gold/silver/copper), pulsing, depletable
- **Predators** — red threats that hunt lobsters
- **Pheromone trails** — food (green) and danger (red) chemical signals left by lobsters (ACO-inspired)
- **Communication** — sound waves between lobsters, floating text of invented words
- **Colonies** — groups of cooperating lobsters with territory
- **Metrics** — language emergence stats (mutual information, topographic similarity, vocabulary)

## Current Technical Stack

- **Renderer:** HTML5 Canvas 2D (no WebGL)
- **Framework:** React 18 (CDN), single index.html file (~1500 lines)
- **Updates:** WebSocket pushes full state every 100ms (10 FPS data, 60 FPS render loop)
- **Layout:** Canvas (left, flex:1) + Right sidebar (280px fixed) + Sound log (bottom 160px)
- **HiDPI:** Properly scaled for 4K displays via devicePixelRatio

## Current Visual Problems (Why We Need Help)

### 1. Lobsters look like gray blobs
Current lobster sprites are procedurally drawn with canvas paths: ellipse body, two eye stalks, small claws, tiny legs, a tail. At the actual rendered size (~7-10 CSS pixels), all the anatomical detail is invisible — they look like gray smudges. The tier glow helps slightly but the creatures themselves have zero personality or recognizability.

### 2. No visual storytelling
The viewer shows data but doesn't tell a STORY. You can't glance at it and understand: "oh, those lobsters are cooperating", "that group discovered a new word", "a predator just disrupted a colony". Every important event looks the same — tiny dots moving on a dark background.

### 3. Pheromone trails are barely visible
Food trails decay fast (0.93/tick) and are deposited along short paths. They show as faint green dots that disappear almost instantly. Danger trails are slightly better but still hard to see. The ACO-inspired trail system is invisible to the viewer.

### 4. Communication is hard to follow
Cyan lines between speaker and listeners are thin and ephemeral. Floating text ("blub glorp") appears for 2 seconds then vanishes. You can't tell if agents are having a conversation or just broadcasting noise.

### 5. Scale ambiguity
With 33 agents in a 40x40 active zone, things feel sparse. With 200+ agents the ocean expands dynamically. The visual needs to work at both scales without feeling empty or cluttered.

### 6. Sidebar is utilitarian
Right sidebar shows raw numbers. No graphs trending over time (except a basic epoch rewards chart). The emergent dictionary (sound→meaning mappings) is a text list with confidence bars — functional but not inspiring.

## Current Rendering Details (For Context)

### Lobster Sprite Anatomy (Procedural Canvas)
```
Size: CELL * 0.55 (half-cell, ~7-10px at current zoom)
Components drawn in order:
  1. Arms + claws (behind body) — animate when speaking
  2. Two pairs of small legs
  3. Body (ellipse carapace) — filled with tier color
  4. Shell highlight (lighter arc on top)
  5. Tail (small ellipse below)
  6. Eyes on stalks (if alive)
Dead: rotated 180°, gray, 0.25 opacity
Grace shield: pulsing dashed cyan circle
```

### Color Palette
```
Background: #0d1f3c (deep ocean blue)
Panel: #111d2e
Text: #c8d6e5
Accent: #00d4ff (cyan)
Shrimp tier: #8395a7 (gray)
Lobster tier: #ee5a24 (orange-red)
Kraken tier: #a55eea (purple)
Gold rift: #ffd700
Silver rift: #c0c0c0
Copper rift: #cd7f32
Predator: #ff4757 (red)
Food pheromone: green
Danger pheromone: red
```

### 7-Layer Rendering Order (back to front)
1. Ocean background (solid dark blue)
2. Pheromone trails (radial gradients)
3. Colony territories (soft gradient + dashed border)
4. Communication lines (thin cyan)
5. Rifts (pulsing glow + colored core)
6. Lobsters (procedural sprites + glow halos)
7. Predators (red triangles + eye)
8. Floating sound text (fading upward)

### World Data (sent via WebSocket every tick)
```json
{
  "ocean_size": 60,
  "active_zone": { "size": 40, "min": 10, "max": 49 },
  "lobsters": [{ "id", "name", "pos", "tier", "alive", "speaking", "net_credits", "grace", "colony" }],
  "rifts": [{ "id", "pos", "richness_pct", "rift_type" }],
  "predators": [{ "id", "pos" }],
  "sounds": [{ "from", "sounds", "pos" }],
  "sound_lines": [{ "from", "to" }],
  "pheromones": [{ "pos", "type", "intensity" }],
  "colonies": [{ "id", "center", "size", "total_reward" }],
  "emergent_dictionary": [{ "sound", "meaning", "confidence", "observations" }],
  "metrics": { "mutual_info", "top_sim", "pos_dis", "vocabulary_size", ... }
}
```

## What I Need From You

### Part 1: Visual Identity Research

Research and recommend a visual direction for the BLUB ocean. Consider:

1. **Lobster representation** — What's the best approach at 7-15px sprite size on canvas?
   - Option A: Emoji (🦐🦞🐙) — simple, recognizable, but limited customization
   - Option B: Pixel art sprites (pre-rendered, tile-based) — retro aesthetic, clear at small sizes
   - Option C: Simplified geometric shapes with strong silhouettes — circles with antenna, distinctive outlines
   - Option D: Pre-rendered PNG sprite sheets loaded as images — highest quality, most work
   - Option E: Something else entirely?

   Consider: we need 3 tiers visually distinct, alive/dead states, speaking animation, and it must look good at 7-15px AND at zoom.

2. **Art direction** — What overall aesthetic fits "living ocean where AI creatures develop language"?
   - Bioluminescent deep sea?
   - Retro pixel aquarium?
   - Abstract/minimalist data viz?
   - Aquarelle/watercolor?
   - Something referencing crypto/tech while staying organic?

3. **Color palette** — Is the current deep blue good? What adjustments would improve readability and mood?

### Part 2: Visual Storytelling

How to make emergent behaviors VISIBLE without cluttering:

1. **Communication visualization** — How to show that two lobsters are "talking" vs "broadcasting noise"? How to show vocabulary convergence visually?
2. **Colony visualization** — How to make cooperative groups feel alive and distinct?
3. **Predator drama** — How to make predator attacks feel dangerous and impactful?
4. **Pheromone trails** — How to make chemical trails beautiful AND informative? (Think: ant colony visualizations)
5. **Language emergence** — How to show that the agents are developing a real language? What visual metaphors work?
6. **Death and rebirth** — Currently dead lobsters are just flipped gray. How to make the cycle of life/death/learning meaningful visually?

### Part 3: Information Architecture

The sidebar currently dumps raw numbers. How to redesign:

1. **Metrics dashboard** — What's the best way to show language metrics trending over time?
2. **Emergent dictionary** — How to visualize sound→meaning mappings in a compelling way?
3. **Leaderboard** — How to make the competitive aspect engaging?
4. **Event feed** — Should we have a "what just happened" narrative feed instead of raw sound log?

### Part 4: Technical Feasibility

For each recommendation, consider:
- We're on Canvas 2D (no WebGL) — what's achievable?
- 60 FPS render loop with 33-200 entities
- Single HTML file architecture (inline everything)
- Data comes pre-computed from server (we can add fields to WebSocket payload if needed)
- Must work on 1080p and 4K displays

### Part 5: Priority Roadmap

Rank your recommendations by:
1. **Impact** — How much does this improve the "wow factor" for someone seeing BLUB for the first time?
2. **Effort** — How hard is it to implement in Canvas 2D?
3. **Risk** — Could this make things worse (performance, readability)?

Give me a prioritized list: "Do this first, then this, then this."

## Reference Projects (For Inspiration)

Look at these for inspiration (but don't copy — BLUB should have its own identity):
- **Lenia** (continuous cellular automata) — beautiful emergent patterns
- **agar.io / slither.io** — simple shapes that work at any scale
- **Subnautica UI** — underwater game with clear readability
- **Observable notebooks** (d3.js) — data visualization that tells stories
- **Ant colony simulations** — pheromone trail visualization
- **Conway's Game of Life visualizations** — emergence from simple rules
- **deepmind.google/blog** — how they visualize agent behavior
- **Neal.fun** projects — playful, accessible data viz

## Constraints

- **NO frameworks beyond React** — keep it simple, single file
- **Canvas 2D only** — no WebGL, no Three.js
- **Performance first** — 60 FPS with 200 entities is non-negotiable
- **The ocean must feel ALIVE** — subtle ambient animations even when nothing is "happening"
- **Accessible** — readable at both 1080p and 4K
- **The viewer IS the product** — this is what convinces developers to build agents for BLUB

## Deliverables Expected

1. **Art direction document** — visual style, mood board description, color palette
2. **Entity design specs** — how each entity should look (with ASCII/text mockups if possible)
3. **Interaction design** — what happens on hover, click, zoom
4. **Information architecture** — sidebar/HUD layout recommendations
5. **Animation catalog** — what animates, how, and why
6. **Implementation roadmap** — ordered list of changes with effort estimates
7. **Anti-patterns** — what to specifically AVOID (common mistakes in this type of viz)
