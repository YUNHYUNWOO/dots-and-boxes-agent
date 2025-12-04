## Project Structure 

```
Haic/
 config/                            # Configuration and settings
    constant.py                     # Global constants
    load_config.py                  # Configuration loader
    exp_configs/                    # Experiment configuration JSONs
       version_comparison/          # Algorithm version comparison tests
       efficiency_test/             # Optimization techniques tests (move ordering, PVS, etc.)
       depth_test/                  # Search depth comparison
       evaluate_test/               # Evaluation function tests
       budget_manager_pdf_base/     # Time/resource budget management tests
    policy_configs/                 # AI policy configs (v0-v4)

 dotsandboxes/                      # Game environment core
    dnb.py                          # Main game class (board state management)
    dnb_engine.py                   # Game engine (rules & game flow)
    dnb_env.py                      # RL environment interface

 search/                            # Search algorithms
    search_engine.py                # Basic search (Minimax)
    ab_search_engine.py             # Alpha-Beta pruning implementation
    TranspositionTable.py           # Transposition table (memoization)

 policy/                            # Game playing strategies
    basepolicy.py                   # Base policy interface
    playable_policy.py              # Random move selection
    opening_policy.py               # Game opening phase policy
    search_policy.py                # Search-based policy
    mixed_policy.py                 # Combined policy

 heuristic/                         # Heuristic evaluation functions
    search_hearistic.py             # Board state evaluation for search
    evaluate.py                     # Evaluation function implementations

 util/                              # Utility functions
    dnb_util.py                     # General game utilities
    bit_dnb_util.py                 # Bit-operation optimized utilities
    budget_manager.py               # Time/compute budget manager
    time_manager.py                 # Time management
    scheduler.py                    # Task scheduling
    chain.py                        # Chain calculation 
    validate.py                     # Game state validation

 simulate/                          # Experiment simulation
    simulate.py                     # Two-AI match simulation
    ab_test.py                      # A/B testing logic
    logger.py                       # Simulation result logging
    sim_result/                     # Simulation results (organized by experiment)

 assets/                            # Images and resources
    (game images, benchmark charts)

 run_experiments.py                 # Main experiment runner
 play_with_ai.py                    # Interactive game interface
 requirements.txt                   # Python dependencies
 README.md                          # This file

```

### Key Components

**Game Engine** (dotsandboxes/)
- Manages board state and game rules for 5x5 Dots and Boxes
- Handles move validation and box capturing logic

**Search Algorithms** (search/)
- Alpha-Beta pruning for efficient game tree exploration
- Transposition table for caching evaluated positions
- Customizable search depth and time limits

**AI Policies** (policy/)
- From random moves to advanced search-based strategies
- Version progression: v0 (random)  v1 (opening)  v2 (search)  v3 (move ordering)  v4 (time control)

**Utilities** (util/)
- Chain detection and bonus calculation
- Time management and budget allocation
- Bit-optimized board representations

**Experiment Framework** (simulate/)
- Automated testing of different algorithm configurations
- Detailed logging and result analysis
- Supports large-scale parameter sweeps