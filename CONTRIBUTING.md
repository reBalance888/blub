# Contributing to BLUB Ocean

BLUB is currently a solo project, but I welcome contributions of all kinds.

## Ways to contribute

### Low barrier — no code needed
- **Watch and report** — Run the simulation, observe agent behavior, report anything unexpected in Issues
- **Spot patterns** — Look at metrics in `metrics_log.jsonl`, notice trends I might miss
- **Suggest papers** — Know a relevant paper on emergent communication, virtual economies, or complex systems? Open an issue
- **Improve docs** — Typos, unclear explanations, missing context

### Medium — some code
- **Analysis tools** — Better visualization of language emergence (message heatmaps, dialect maps, metric dashboards)
- **Viewer improvements** — The bioluminescent viewer can always be more beautiful and informative
- **Test coverage** — The project is under-tested

### High impact
- **New ecological mechanics** — Propose and implement new pressures (new predator types, environmental events)
- **Metric implementations** — New ways to measure language quality beyond TopSim/CIC/CSR
- **Performance** — The tick loop could be faster

## How to submit changes

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-idea`)
3. Make your changes
4. Test that the ocean still runs: `python -m server.main` + `python -u agents/run_agents.py --count 33 --type social`
5. Open a PR with a description of what changed and why

## Sacred principles

Please respect the five design principles when contributing:

1. **Never hardcode language or conventions** — if your change adds a predefined meaning to a sound, it violates the core premise
2. **No neural networks for agents** — we use simple learners under pressure
3. **Multiple pressures, not one** — don't remove a pressure source to make metrics look better
4. **No global information** — agents should only know what they can see/hear locally
5. **No single-metric optimization** — don't tune for TopSim at the expense of everything else

## Questions?

Open an issue. There are no stupid questions — this project is weird and that's the point.
