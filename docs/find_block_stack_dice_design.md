# IMPLEMENTATION DESIGN: find_block_stack_dice

## 1. METADATA

```
new_env_name:      find_block_stack_dice
class_name:        FindBlockStackDiceEnv
file_path:         tampura_environments/tampura_environments/find_block_stack_dice/env.py
config_path:       env_configs/find_block_stack_dice.yml
base_class:        TampuraEnv (tampura.environment)
reference_env:     tampura_environments/find_dice/env.py  ← start by copying this
register_keys:     "find_block_stack_dice", "find_block_stack_dice_hard"
```

**Task**: Robot sees N objects on a table. Find target block (red box) and target dice (blue cube),
pick the dice, then stack (place on top of) the block. Some objects may initially be hidden under
an obstacle. Robot must move the obstacle first.

---

## 2. OBJECT SYSTEM

### 2.1 Object representation
All objects are **PyBullet primitives** (NOT YCB objects).
Use existing `create_block()` in `panda_utils/panda_env_utils.py` line 548:

```python
from tampura_environments.panda_utils.panda_env_utils import create_block
# Signature:
def create_block(color, position=(0,0,0.5), halfExtents=(0.02,0.02,0.02), client=None) -> int
```

### 2.2 Object categories (World.categories strings)
| category          | role                              |
|-------------------|-----------------------------------|
| `"target_block"`  | red box — dice must be placed ON this |
| `"target_dice"`   | blue cube — must pick and stack this |
| `"distractor_block"` | non-target block (same size, different color) |
| `"distractor_dice"`  | non-target dice (same size, different color)  |
| `"obstacle"`      | covers target_dice at init; must be moved first |

### 2.3 Sizes (halfExtents tuples)
```python
BLOCK_HALF_EXTENTS = (0.05, 0.05, 0.025)   # full 0.10 × 0.10 × 0.05 m
DICE_HALF_EXTENTS  = (0.02, 0.02, 0.02)    # full 0.04 × 0.04 × 0.04 m
OBSTACLE_HALF_EXTENTS = (0.03, 0.03, 0.025) # full 0.06 × 0.06 × 0.05 m
```

### 2.4 Colors (RGBA tuples)
```python
TARGET_BLOCK_COLOR  = (1.0, 0.0, 0.0, 1.0)   # red
TARGET_DICE_COLOR   = (0.0, 0.0, 1.0, 1.0)   # blue
OBSTACLE_COLOR      = (0.5, 0.5, 0.5, 1.0)   # gray

DISTRACTOR_COLORS = [
    (0.0, 0.8, 0.0, 1.0),  # green
    (1.0, 1.0, 0.0, 1.0),  # yellow
    (1.0, 0.5, 0.0, 1.0),  # orange
    (0.5, 0.0, 0.5, 1.0),  # purple
]
# distractor_block[i] uses DISTRACTOR_COLORS[i % 4]
# distractor_dice[i] uses DISTRACTOR_COLORS[(i+2) % 4]  ← offset to avoid same-color as block
```

---

## 3. GRASP MODE CHANGE — CRITICAL

**find_dice uses `GRASP_MODE = "saved"`** which reads pre-computed YCB grasp files.
This WILL FAIL for primitive shapes.

**New env must use `GRASP_MODE = "top"`** which uses AABB-based analytic grasps.
- Implementation in `panda_env_utils.py` lines 313-318:
  ```python
  elif grasp_mode == "top":
      aabb, pose = pbu.get_oobb(obj, **kwargs)
      cand = pbu.get_top_and_bottom_grasps(obj, aabb, pose, tool_pose=TOOL_POSE, grasp_length=0.01, **kwargs)
      return random.sample(cand, len(cand))
  ```
- Works for any convex shape with an AABB.
- Change: `GRASP_MODE = "top"` at top of new env.py

---

## 4. SCENE GENERATION

### 4.1 Config parameters
```python
N_DISTRACTOR_BLOCKS = config.get("n_distractor_blocks", 2)
N_DISTRACTOR_DICE   = config.get("n_distractor_dice",   2)
N_OBSTACLES         = config.get("n_obstacles",         1)  # default 1 (covers target_dice)
```

### 4.2 Table placement bounds
Table surface z ≈ 0.0 (from TABLE_POSE + TABLE_AABB).
Object base z = halfExtents[2] (sits on table surface).

Safe placement range (robot workspace subset):
```python
X_RANGE = (0.30, 0.65)   # robot can reach comfortably
Y_RANGE = (-0.25, 0.25)
```

Minimum distance between object centers: 0.12 m (rejection sampling).

### 4.3 setup_world_primitives() function — replaces setup_world()

```python
def setup_world_primitives(robot, client, n_distractor_blocks, n_distractor_dice,
                           n_obstacles, poses=None, **kwargs) -> Tuple[World, List[pbu.Pose]]:
    floor, obstacles_static = create_default_env(client=client, **kwargs)

    movable = []
    categories = []
    placed_xy = []

    def rand_pose(half_ext, min_dist=0.12):
        for _ in range(200):
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)
            z = half_ext[2]  # sit on table
            if all(np.linalg.norm([x - px, y - py]) > min_dist
                   for px, py in placed_xy):
                placed_xy.append((x, y))
                return pbu.Pose(pbu.Point(x, y, z))
        raise RuntimeError("Could not place object: workspace too crowded")

    # Creation order determines index → category alignment
    # 0: target_block
    b = create_block(TARGET_BLOCK_COLOR, halfExtents=BLOCK_HALF_EXTENTS, client=client)
    movable.append(b); categories.append("target_block")

    # 1: target_dice
    d = create_block(TARGET_DICE_COLOR, halfExtents=DICE_HALF_EXTENTS, client=client)
    movable.append(d); categories.append("target_dice")

    # 2..1+n: distractor blocks
    for i in range(n_distractor_blocks):
        color = DISTRACTOR_COLORS[i % len(DISTRACTOR_COLORS)]
        b = create_block(color, halfExtents=BLOCK_HALF_EXTENTS, client=client)
        movable.append(b); categories.append("distractor_block")

    # next n: distractor dice
    for i in range(n_distractor_dice):
        color = DISTRACTOR_COLORS[(i+2) % len(DISTRACTOR_COLORS)]
        d = create_block(color, halfExtents=DICE_HALF_EXTENTS, client=client)
        movable.append(d); categories.append("distractor_dice")

    # next n: obstacles
    for _ in range(n_obstacles):
        o = create_block(OBSTACLE_COLOR, halfExtents=OBSTACLE_HALF_EXTENTS, client=client)
        movable.append(o); categories.append("obstacle")

    # Pose assignment: use provided poses or generate new ones
    if poses is None:
        target_block_pose = rand_pose(BLOCK_HALF_EXTENTS)
        target_dice_pose  = rand_pose(DICE_HALF_EXTENTS)
        generated_poses   = [target_block_pose, target_dice_pose]

        for i in range(2, len(movable)):
            cat = categories[i]
            if cat == "obstacle" and i == 2 + n_distractor_blocks + n_distractor_dice:
                # First obstacle: place on top of target_dice
                td_pos = target_dice_pose[0]
                obs_z  = DICE_HALF_EXTENTS[2]*2 + OBSTACLE_HALF_EXTENTS[2]
                obs_pose = pbu.Pose(pbu.Point(td_pos[0], td_pos[1], obs_z))
                generated_poses.append(obs_pose)
                placed_xy.append((td_pos[0], td_pos[1]))
            else:
                h = (BLOCK_HALF_EXTENTS if "block" in cat else
                     DICE_HALF_EXTENTS  if "dice"  in cat else
                     OBSTACLE_HALF_EXTENTS)
                generated_poses.append(rand_pose(h))
        poses = generated_poses

    for obj, pose in zip(movable, poses):
        pbu.set_pose(obj, pose, client=client)

    client_id = len(CLIENT_MAP)
    CLIENT_MAP[client_id] = client

    pbu.set_joint_positions(robot, robot.get_group_joints(ARM_GROUP, client=client),
                            DEFAULT_ARM_POS, client=client)
    pbu.set_joint_positions(robot, robot.get_group_joints(GRIPPER_GROUP, client=client),
                            OPEN_GRIPPER_POS, client=client)

    world = World(
        client_id=client_id,
        robot=robot,
        environment=obstacles_static + movable,
        floor=floor,
        objects=movable,
        categories=categories,
    )
    return world, poses
```

---

## 5. BELIEF STATE (SceneBelief modifications)

### 5.1 New fields in __init__
```python
self.stacked_pair: Optional[Tuple[str, str]] = None  # (dice_sym, block_sym) after stack
```
Keep `self.placed = False` (reuse as "terminal action completed").

### 5.2 update() additions
```python
if action.name == "stack":
    new_belief.placed = True
    new_belief.stacked_pair = (action.args[0], action.args[2])  # (dice_sym, block_sym)
```

### 5.3 abstract() changes
Replace `is-target` block with:
```python
for obj_sym, cat in zip(self.world.objects, self.world.categories):
    if cat == "target_block":
        ab.items.append(Atom("is-target-block", [obj_sym]))
    elif cat == "target_dice":
        ab.items.append(Atom("is-target-dice", [obj_sym]))
```

Add `on` and `target-stacked`:
```python
if self.stacked_pair is not None:
    dice_sym, block_sym = self.stacked_pair
    ab.items.append(Atom("on", [dice_sym, block_sym]))
    # Check if it's the target pair
    objs = self.world.objects
    cats = self.world.categories
    if (dice_sym in objs and block_sym in objs and
        cats[objs.index(dice_sym)] == "target_dice" and
        cats[objs.index(block_sym)] == "target_block"):
        ab.items.append(Atom("target-stacked"))
```

---

## 6. PDDL SPEC (get_problem_spec())

### 6.1 Predicates (complete list)
```python
predicates = [
    Predicate("known-pose",       ["physical"]),
    Predicate("holding",          ["physical"]),
    Predicate("at-start",         ["physical"]),
    Predicate("object-pose",      ["physical", "pose"]),
    Predicate("is-target-block",  ["physical"]),     # NEW — replaces is-target
    Predicate("is-target-dice",   ["physical"]),     # NEW — replaces is-target
    Predicate("target-stacked",   []),               # NEW — zero-arg goal flag
    Predicate("on",               ["physical", "physical"]),  # NEW: on(top, bottom)
    Predicate("at-home",          []),
    Predicate("looking-conf",     ["physical", "conf"]),
    Predicate("moved",            ["physical"]),
    Predicate("object-grasp",     ["physical", "grasp"]),
    Predicate("at-grasp",         ["physical", "grasp"]),
    Predicate("stacked-pose",     ["physical", "physical", "pose"]),  # NEW: (top, bottom, sp)
]
```

### 6.2 Stream schemas (complete list)
```python
stream_schemas = [
    # UNCHANGED from find_dice:
    StreamSchema(
        name="look-conf-sample",
        inputs=["?o"], input_types=["physical"],
        output="?q", output_type="conf",
        certified=[Atom("looking-conf", ["?o", "?q"])],
        sample_fn=look_sample_fn_wrapper(self.starting_belief),
    ),
    StreamSchema(
        name="grasp-sample",
        inputs=["?o"], input_types=["physical"],
        output="?g", output_type="grasp",
        certified=[Atom("object-grasp", ["?o", "?g"])],
        sample_fn=grasp_sample_fn_wrapper(self.sim_world),  # uses GRASP_MODE="top"
    ),
    StreamSchema(
        name="place-sample",
        inputs=["?o"], input_types=["physical"],
        output="?p", output_type="pose",
        certified=[Atom("object-pose", ["?o", "?p"])],
        sample_fn=placement_sample_fn_wrapper(self.sim_world),
    ),
    # NEW:
    StreamSchema(
        name="stack-pose-sample",
        inputs=["?top", "?bottom"],
        input_types=["physical", "physical"],
        output="?sp",
        output_type="pose",
        certified=[Atom("stacked-pose", ["?top", "?bottom", "?sp"])],
        sample_fn=stack_pose_sample_fn_wrapper(self.sim_world),
    ),
]
```

### 6.3 Action schemas (complete list)

**pick** — UNCHANGED from find_dice

**place** — UNCHANGED from find_dice (used to move obstacles out of the way)

**stack** — NEW:
```python
ActionSchema(
    name="stack",
    inputs=["?top", "?g", "?bottom", "?sp"],
    input_types=["physical", "grasp", "physical", "pose"],
    preconditions=[
        Atom("holding",      ["?top"]),
        Atom("at-grasp",     ["?top", "?g"]),
        Atom("known-pose",   ["?bottom"]),
        Atom("stacked-pose", ["?top", "?bottom", "?sp"]),
    ],
    effects_fn=stack_effects_fn,
    execute_fn=stack_execute_fn,
    effects=[
        Atom("on",           ["?top", "?bottom"]),
        Atom("moved",        ["?top"]),
        Atom("known-pose",   ["?top"]),
        Not(Atom("at-home")),
        Not(Atom("holding",  ["?top"])),
        Not(Atom("at-grasp", ["?top", "?g"])),
    ],
    # No verify_effects: deterministic if IK succeeds
),
```

**look** — UNCHANGED from find_dice

**go-home** — UNCHANGED from find_dice

**NoOp()** — UNCHANGED

### 6.4 Reward
```python
reward = Atom("target-stacked")   # zero-arg predicate; set in abstract() when stacking succeeded
```

---

## 7. STREAM IMPLEMENTATION: stack-pose-sample

```python
def stack_pose_sample_fn_wrapper(sim_world):
    def sample_fn(args: List[str], store: AliasStore):
        top_sym, bottom_sym = args
        top_body    = store.get(top_sym)
        bottom_body = store.get(bottom_sym)

        client = sim_world.client

        bottom_aabb = pbu.get_aabb(bottom_body, client=client)
        top_aabb    = pbu.get_aabb(top_body,    client=client)

        top_half_h = (top_aabb.upper[2] - top_aabb.lower[2]) / 2.0
        stack_z    = bottom_aabb.upper[2] + top_half_h + 0.002  # 2mm clearance

        margin = 0.005  # stay within block footprint
        x_lo = bottom_aabb.lower[0] + margin
        x_hi = bottom_aabb.upper[0] - margin
        y_lo = bottom_aabb.lower[1] + margin
        y_hi = bottom_aabb.upper[1] - margin

        if x_lo >= x_hi or y_lo >= y_hi:
            # target smaller than margin: place at center
            cx = (bottom_aabb.lower[0] + bottom_aabb.upper[0]) / 2
            cy = (bottom_aabb.lower[1] + bottom_aabb.upper[1]) / 2
        else:
            cx = random.uniform(x_lo, x_hi)
            cy = random.uniform(y_lo, y_hi)

        yaw = random.uniform(0, 2 * np.pi)
        pose = pbu.Pose(pbu.Point(cx, cy, stack_z), pbu.Euler(yaw=yaw))
        return pose

    return sample_fn
```

**NOTE**: `bottom_aabb` is in world frame. This is correct IF `belief.set_sim(store)` is called
before stream sampling (it is, in `flat_stream_sample` and `progressive_widening`).

---

## 8. ACTION IMPLEMENTATIONS: stack

### 8.1 stack_effects_fn (copy of place_effects_fn, change action name check)
```python
def stack_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    (top_sym, grasp_sym, bottom_sym, stack_pose_sym) = action.args
    grasp      = store.get(grasp_sym)
    top_body   = store.get(top_sym)
    stack_pose = store.get(stack_pose_sym)
    belief.set_sim(store)
    new_belief = None

    pre_confs = grasp_ik(
        belief.world,
        top_body,
        stack_pose,
        grasp,
        obstacles=list(set(belief.world.environment) - {top_body}) + [belief.world.floor],
    )
    if pre_confs is not None:
        new_belief = belief.update(
            action,
            SceneObservation(
                moved=top_sym,
                poses={top_sym: stack_pose},
                conf=pre_confs,
            ),
            store,
        )

    if new_belief is None:
        return AbstractBeliefSet.from_beliefs([belief], store)
    return AbstractBeliefSet.from_beliefs([new_belief], store)
```

### 8.2 stack_execute_fn (copy of place_execute_fn, same logic)
```python
def stack_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    (top_sym, grasp_sym, bottom_sym, stack_pose_sym) = action.args
    stack_pose = store.get(stack_pose_sym)
    g          = store.get(grasp_sym)
    obs = SceneObservation(
        conf=belief.current_conf, grasp=g, grasp_body=belief.grasp_body
    )
    belief.set_sim(store)
    conf = ik(belief.world, store.get(top_sym), stack_pose, g)

    if conf is None:
        return state, obs

    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        conf,
        obstacles=[belief.world.floor]
            + list(set(belief.world.environment) - {store.get(top_sym)}),
        attachments=[grasp_attachment(belief.world, g)],
    )
    if motion_plan is None:
        return state, obs

    state = place_execute(state, motion_plan)   # reuse place_execute
    obs.poses[top_sym] = stack_pose
    obs.conf    = conf
    obs.grasp   = None
    obs.grasp_body = None
    obs.moved   = top_sym
    return state, obs
```

---

## 9. initialize() METHOD

Copy from find_dice's initialize() and replace:
- `get_scene_data()` + `setup_world()` → `setup_world_primitives()`
- Remove JSON loading, remove EXCLUDE_CLASSES check

```python
def initialize(self) -> Tuple[SceneBelief, AliasStore]:
    store = AliasStore()

    state_vis  = self.vis
    belief_vis = False

    world_robot,     world_client     = setup_robot(vis=state_vis)
    sim_world_robot, sim_world_client = setup_robot(
        vis=belief_vis,
        camera_matrix=world_robot.camera.camera_matrix,
    )

    n_db = self.config.get("n_distractor_blocks", 2)
    n_dd = self.config.get("n_distractor_dice",   2)
    n_ob = self.config.get("n_obstacles",         1)

    # Generate poses once; pass to sim_world to ensure both worlds share identical layout
    self.world, poses = setup_world_primitives(
        world_robot, world_client,
        n_distractor_blocks=n_db, n_distractor_dice=n_dd, n_obstacles=n_ob,
        poses=None, vis=state_vis,
    )
    self.state = SceneState(self.world)

    self.sim_world, _ = setup_world_primitives(
        sim_world_robot, sim_world_client,
        n_distractor_blocks=n_db, n_distractor_dice=n_dd, n_obstacles=n_ob,
        poses=poses, vis=belief_vis,
    )

    self.sim_world.objects = [
        store.add_typed(o, "physical") for o in self.sim_world.objects
    ]

    camera_image = self.world.robot.get_image(client=self.world.client)
    obs = obs_from_camera_image(
        self.sim_world.objects, self.world, camera_image, DEFAULT_ARM_POS, store
    )

    b = SceneBelief(self.sim_world, vis_grid=True)
    b.update_visibility(camera_image, store, include_unseen=True)
    b.visibility_grid.draw_intervals(b.world.client)
    b = b.update(NoOp(), obs, store=store)
    b.set_sim(store)
    self.starting_belief = b

    if self.vis:
        b.visibility_grid.draw_intervals(self.world.client)

    return b, store
```

---

## 10. VISUALIZATION FIX

**See `docs/vis_fix_design.md` for the full design and root cause analysis.**

Implement and verify the fix in find_dice first. Once verified, apply only the
`step()` pattern change to this environment:

```python
# find_block_stack_dice/env.py — apply after find_dice fix is verified
def step(self, action, belief, store):
    result = super().step(action, belief, store)
    self._recorder.capture_frame()   # synchronous; no background thread
    return result
```

FrameRecorder constructor call and camera setup are identical to find_dice
(see `docs/vis_fix_design.md` Section 3).

---

## 11. wrapup() METHOD

Unchanged from find_dice:
```python
def wrapup(self):
    self._recorder.make_gif()
```

---

## 12. CONFIG FILE (env_configs/find_block_stack_dice.yml)

```yaml
task: find_block_stack_dice
max_steps: 35
policy: tampura_policy
learning_strategy: bayes_optimistic
record: false
record_camera: external
record_interval: 0.1
n_distractor_blocks: 2
n_distractor_dice: 2
n_obstacles: 1
```

---

## 13. FILE STRUCTURE

```
tampura_environments/
  tampura_environments/
    find_block_stack_dice/
      __init__.py    (empty)
      env.py         (main implementation)

env_configs/
  find_block_stack_dice.yml

docs/
  vis_fix_design.md              (visualization fix — implement in find_dice first)
  find_block_stack_dice_design.md  (this file)
```

---

## 14. IMPLEMENTATION CHECKLIST (3-pass verification)

### Pass 1: PDDL Planning Trace

Minimal plan to achieve `target-stacked`:

```
1. pick(?obstacle, ?g_obs)              → pick up obstacle covering dice
2. place(?obstacle, ?g_obs, ?p_obs)     → place obstacle elsewhere; moved(obstacle) ✓
3. look(?target_dice, ?obstacle, ?q)    → known-pose(target_dice) ✓
4. pick(?target_dice, ?g_dice)          → holding(target_dice), at-grasp(target_dice, g_dice)
5. stack(?target_dice, ?g_dice, ?target_block, ?sp) → target-stacked ✓
```

target_block is visible at init → `known-pose(target_block)` already in initial abstract belief.

Precondition check for step 3 (look):
- `looking-conf(obstacle, q)` ← certified by look-conf-sample(obstacle) ✓
- `Not(known-pose(target_dice))` ✓ (hidden at init)
- `known-pose(obstacle)` ← from place effects ✓
- `Not(holding)` ✓ (placed in step 2)

Precondition check for step 5 (stack):
- `holding(target_dice)` ✓ from step 4
- `at-grasp(target_dice, g_dice)` ✓ from step 4
- `known-pose(target_block)` ✓ visible at init
- `stacked-pose(target_dice, target_block, sp)` ✓ from stack-pose-sample stream

PDDL plan trace verified. ✓

### Pass 2: Python implementation feasibility

| Component | Status | Notes |
|-----------|--------|-------|
| `create_block()` | ✓ exists | panda_env_utils.py:548 |
| `GRASP_MODE="top"` | ✓ exists | panda_env_utils.py:313 |
| `stack_pose_sample_fn` | ✓ straightforward | uses pbu.get_aabb |
| `stack_effects_fn` | ✓ copy of place_effects_fn | identical structure |
| `stack_execute_fn` | ✓ copy of place_execute_fn | reuses place_execute |
| `SceneBelief.stacked_pair` | ✓ new field | simple Optional[Tuple] |
| `Atom("target-stacked")` | ✓ zero-arg predicate | same pattern as at-home |
| Sim world pose sync | ✓ fixed in Section 4.3 | poses passed as parameter |
| FrameRecorder threading | ✓ fixed in vis_fix_design.md | synchronous capture_frame() |

### Pass 3: TAMPURA framework compatibility

| Interface | Requirement | Satisfied? |
|-----------|-------------|------------|
| `initialize()` return | `(Belief, AliasStore)` | ✓ |
| `get_problem_spec()` return | `ProblemSpec` | ✓ |
| `step()` signature | `step(action, belief, store)` | ✓ (delegates to super) |
| `wrapup()` | cleanup | ✓ |
| `register_env()` | called at module bottom | ✓ |
| `ActionSchema.effects_fn` | `(action, belief, store) → AbstractBeliefSet` | ✓ |
| `ActionSchema.execute_fn` | `(action, belief, state, store) → (state, obs)` | ✓ |
| `StreamSchema.sample_fn` | `(args: List[str], store) → output_obj` | ✓ |
| `Predicate("target-stacked", [])` | zero-arg valid | ✓ — at-home uses same pattern |

All 3 verification passes complete. ✓

---

## 15. KNOWN RISKS

1. **`get_top_and_bottom_grasps` for small cubes**: The dice (0.04m) is close to gripper width
   limits. If all grasps fail IK, the planner will never pick the dice. Mitigation: try
   `grasp_length=0.005` instead of 0.01 in the top-mode call.

2. **`pbu.get_aabb` in `stack_pose_sample_fn`**: Returns world-frame AABB. Correct only if
   `belief.set_sim(store)` has been called — this is guaranteed by flat_stream_sample and
   progressive_widening call order.

3. **`look` action `depends` field**: find_dice uses `depends=[Atom("moved", ["?o2"])]`.
   This must be preserved in the new env so the planner correctly links look(target_dice, obstacle)
   to the prior place(obstacle) action.

---

## 16. IMPORTS REQUIRED IN env.py

Same as find_dice, plus `create_block`:
```python
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, GRIPPER_GROUP, OPEN_GRIPPER_POS,
    SceneState, World, create_block, create_default_env,
    dimensions_from_camera_image, get_grasp, get_shortened_table_dims,
    grasp_attachment, grasp_ik, ik, pick_execute, pixel_from_point,
    place_execute, placement_sample, plan_motion, plan_workspace_motion,
    pose_to_vec, setup_robot, transformation_to_pose,
)
```

---

*Implementation order:*
*1. [find_dice] docs/vis_fix_design.md — fix FrameRecorder threading in find_dice and verify*
*2. [this env] Section 4 (setup_world_primitives with pose sync built-in)*
*3. [this env] Section 5-6 (SceneBelief + PDDL spec)*
*4. [this env] Section 8 (stack action fns)*
*5. [this env] Section 10 (copy verified recorder pattern from find_dice)*
