Read the AGENTS.MD file

### GPU / compute safety

NEVER launch multiple GPU-hungry scripts in parallel. This machine has a single GPU. Parallel GPU jobs will OOM, thrash the system, and risk losing work across the entire machine. Always run experiments SEQUENTIALLY — use a single bash loop or script that runs one job at a time. If you need to run N experiments, write a for-loop, not N parallel background commands.