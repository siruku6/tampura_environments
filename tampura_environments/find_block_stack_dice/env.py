from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, AbstractBeliefSet, Action,
                             ActionSchema, AliasStore, Belief, NoOp,
                             Observation, StreamSchema)
from tampura.symbolic import And, Atom, Exists, Not, Predicate

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, GRIPPER_GROUP, OPEN_GRIPPER_POS,
    SceneState, World, create_block, create_default_env,
    dimensions_from_camera_image, get_grasp, get_shortened_table_dims,
    grasp_attachment, grasp_ik, ik, pick_execute, pixel_from_point,
    place_execute, placement_sample, plan_motion, plan_workspace_motion,
    pose_to_vec, setup_robot, transformation_to_pose,
)
from tampura_environments.panda_utils.primitives import Grasp
from tampura_environments.panda_utils.robot import (CAMERA_FRAME,
                                                    DEFAULT_ARM_POS,
                                                    PANDA_TOOL_TIP)
from tampura_environments.panda_utils.voxel_utils import VoxelGrid
from tampura_environments.panda_utils.frame_recorder import (
    FrameRecorder,
    make_external_capture_fn,
    make_robot_capture_fn,
)


GRID_RESOLUTION = 0.015
GRASP_MODE = "top"
EXECUTE_ATTEMPTS = 100

TARGET_BLOCK_COLOR  = (1.0, 0.0, 0.0, 1.0)
TARGET_DICE_COLOR   = (0.0, 0.0, 1.0, 1.0)
OBSTACLE_COLOR      = (0.5, 0.5, 0.5, 1.0)

DISTRACTOR_COLORS = [
    (0.0, 0.8, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.5, 0.0, 1.0),
    (0.5, 0.0, 0.5, 1.0),
]

BLOCK_HALF_EXTENTS    = (0.05, 0.05, 0.025)
DICE_HALF_EXTENTS     = (0.02, 0.02, 0.02)
OBSTACLE_HALF_EXTENTS = (0.03, 0.03, 0.025)

X_RANGE = (0.30, 0.65)
Y_RANGE = (-0.25, 0.25)


@dataclass
class SceneObservation(Observation):
    camera_image: Optional[pbu.CameraImage] = None
    poses: Dict[str, pbu.Pose] = field(default_factory=lambda: {})
    grasp: Optional[Grasp] = None
    grasp_body: Optional[str] = None
    conf: Optional[str] = None
    moved: Optional[str] = None
    possible_locations: List[Any] = field(default_factory=lambda: [])

    def __repr__(self):
        return f"SceneObservation(...)"


def obs_from_camera_image(
    object_syms: List[str],
    world: World,
    camera_image: pbu.CameraImage,
    conf,
    store: AliasStore,
) -> SceneObservation:
    seg_bodies = camera_image.segmentationMaskBuffer[:, :, 0]
    visible_bodies = set(np.unique(seg_bodies.astype(int)))

    # Obstacle centers used to determine whether a dice is physically blocked.
    # When an obstacle is directly above a dice (overlapping XY footprint, higher Z center),
    # the dice is treated as hidden regardless of camera visibility.
    obstacle_centers = [
        pbu.get_pose(obj, client=world.client)[0]
        for obj, cat in zip(world.objects, world.categories)
        if cat == "obstacle"
    ]

    object_poses = {}
    for obj_sym, gt_pose, cat in zip(object_syms, world.poses, world.categories):
        body = store.get(obj_sym)
        if body not in visible_bodies:
            continue

        if cat in ("target_dice", "distractor_dice") and obstacle_centers:
            dx, dy, dz = gt_pose[0]
            x_tol = OBSTACLE_HALF_EXTENTS[0] + DICE_HALF_EXTENTS[0]
            y_tol = OBSTACLE_HALF_EXTENTS[1] + DICE_HALF_EXTENTS[1]
            if any(
                abs(ox - dx) < x_tol
                and abs(oy - dy) < y_tol
                and oz > dz + DICE_HALF_EXTENTS[2]
                for ox, oy, oz in obstacle_centers
            ):
                continue  # dice is hidden under an obstacle

        object_poses[obj_sym] = gt_pose

    return SceneObservation(camera_image=camera_image, poses=object_poses, conf=conf)


class SceneBelief(Belief):
    def __init__(self, world: World, vis_grid=False, **kwargs):
        self.objects = {}
        self.camera_intrinsics = None
        self.world = world
        floor_aabb = pbu.get_aabb(self.world.floor, client=self.world.client)
        self.grasp = None
        self.grasp_body = None
        self.current_conf = None
        self.possible_locations = None
        self.placed = False
        self.stacked_pair: Optional[Tuple[str, str]] = None
        visibility_aabb = pbu.AABB(
            lower=[
                0.37,
                floor_aabb.lower[1],
                floor_aabb.upper[2] + 0.02,
            ],
            upper=[
                floor_aabb.upper[0] - 0.05,
                floor_aabb.upper[1],
                floor_aabb.upper[2] + 0.1,
            ],
        )
        self.moved = []
        self.object_poses = {}
        self.vis_grid = vis_grid
        self.visibility_grid = self.setup_visibility_grid(visibility_aabb)
        # Permanent structural fact: maps dice_sym → obstacle_sym for dice
        # that were initially covered by an obstacle at scene initialization.
        # Set once by initialize() and copied through update(); never cleared.
        self.initially_covered: Dict[str, str] = {}

    def vectorize(self):
        obj_pose_vecs = []
        for obj in self.world.objects:
            if self.get_pose(obj) is None:
                obj_pose_vecs.append(pose_to_vec(pbu.unit_pose()))
            else:
                obj_pose_vecs.append(pose_to_vec(self.get_pose(obj)))
        vectorized_vis_grid = np.array(
            [self.visibility_grid.is_occupied(voxel) for voxel in self.get_all_voxels()]
        )
        b_vec = np.concatenate([vectorized_vis_grid] + obj_pose_vecs)
        return b_vec

    def get_pose(self, obj):
        if obj in self.object_poses:
            return self.object_poses[obj]
        return None

    def update_objects(self, obs: SceneObservation):
        for obj_sym, pose in obs.poses.items():
            self.object_poses[obj_sym] = pose

    def get_all_voxels(self):
        return list(self.visibility_grid.voxels_from_aabb(self.visibility_grid.aabb))

    def set_sim(self, store: AliasStore):
        for obj_sym, pose in self.object_poses.items():
            if pose is None:
                pose = pbu.Pose(pbu.Point(x=100))
            pbu.set_pose(store.get(obj_sym), pose, client=self.world.client)

    def get_new_seen_voxels(
        self, camera_image: pbu.CameraImage, include_unseen=False
    ) -> List[Any]:
        voxels = []
        width, height = dimensions_from_camera_image(camera_image)

        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):
                center_world = self.visibility_grid.to_world(
                    self.visibility_grid.center_from_voxel(voxel)
                )
                center_camera = pbu.tform_point(
                    pbu.invert(camera_image.camera_pose), center_world
                )
                distance = center_camera[2]
                pixel = pixel_from_point(
                    camera_image.camera_matrix, center_camera, width, height
                )
                if pixel is not None:
                    depth = camera_image.depthPixels[pixel.row, pixel.column]
                    if distance <= depth:
                        voxels.append(voxel)
                elif include_unseen:
                    voxels.append(voxel)

        return voxels

    def update_visibility(
        self,
        camera_image: pbu.CameraImage,
        store: AliasStore,
        possible_locations=None,
        include_unseen=False,
    ):
        new_voxels = self.get_new_seen_voxels(
            camera_image, include_unseen=include_unseen
        )
        for voxel in new_voxels:
            self.visibility_grid.set_free(voxel=voxel)

        radius = 0.001
        point_collision_shape_id = self.world.client.createCollisionShape(
            p.GEOM_SPHERE, radius=radius
        )
        point_obj = self.world.client.createMultiBody(
            0, point_collision_shape_id, -1, [0, 0, 0]
        )
        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):
                point = self.visibility_grid.center_from_voxel(voxel)

                if possible_locations is not None:
                    distances = []
                    for possible_point in possible_locations:
                        transformed_pose = transformation_to_pose(possible_point)[0]
                        transformed_array = np.array(transformed_pose)
                        point_array = np.array(point)
                        distance = np.linalg.norm(point_array - transformed_array)
                        distances.append(distance)

                    min_distance = min(float(distance) for distance in distances)

                    if min_distance > 0.05:
                        self.visibility_grid.set_free(voxel=voxel)
                        continue

                for obj_sym in self.object_poses:
                    pbu.set_pose(
                        point_obj,
                        pbu.Pose(point=pbu.Point(*point)),
                        client=self.world.client,
                    )

                    distance_threshold = GRID_RESOLUTION / 2.0
                    closest_points = self.world.client.getClosestPoints(
                        store.get(obj_sym), point_obj, distance_threshold
                    )
                    if len(closest_points) > 0:
                        self.visibility_grid.set_free(voxel=voxel)
        pbu.remove_body(point_obj, client=self.world.client)

    def setup_visibility_grid(self, surface: pbu.AABB) -> VoxelGrid:
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = pbu.Pose(pbu.Point(z=0.01))
        surface_aabb = pbu.AABB(
            lower=surface.lower,
            upper=[surface.upper[0], surface.upper[1], GRID_RESOLUTION * 2],
        )

        grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=pbu.BLUE,
            client=self.world.client,
        )
        static_grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=pbu.BLACK,
            client=self.world.client,
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)
            static_grid.set_occupied(voxel)

        return grid

    def update(
        self,
        action: Action,
        observation: SceneObservation,
        store: AliasStore,
    ) -> SceneBelief:
        new_belief = copy.deepcopy(self)

        if action.name == "place":
            new_belief.placed = True

        if action.name == "stack":
            new_belief.placed = True
            new_belief.stacked_pair = (action.args[0], action.args[2])

        if observation.moved is not None:
            new_belief.moved = list(set(self.moved + [observation.moved]))

        new_belief.current_conf = observation.conf
        new_belief.update_objects(observation)
        new_belief.set_sim(store)

        if observation.possible_locations is not None:
            new_belief.possible_locations = observation.possible_locations

        if observation.camera_image is not None:
            new_belief.camera_intrinsics = observation.camera_image.camera_matrix
            new_belief.update_visibility(observation.camera_image, store)

        if observation.grasp is not None:
            new_belief.grasp = observation.grasp
            new_belief.grasp_body = observation.grasp_body
        else:
            new_belief.grasp = None
            new_belief.grasp_body = None

        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        ab = AbstractBelief([Atom("known-pose", [o]) for o in self.object_poses.keys()])

        if self.grasp is not None:
            ab.items.append(Atom("holding", [self.grasp_body]))
            ab.items.append(Atom("at-grasp", [self.grasp_body, self.grasp]))

        for obj_sym, cat in zip(self.world.objects, self.world.categories):
            if cat == "target_block":
                ab.items.append(Atom("is-target-block", [obj_sym]))
            elif cat == "target_dice":
                ab.items.append(Atom("is-target-dice", [obj_sym]))

        ab.items += [Atom("moved", [p]) for p in self.moved]

        # Permanent link: dice that were initially covered by an obstacle.
        # This predicate is never removed; it guides the look action to use
        # the correct reference object (the covering obstacle).
        for dice_sym, obs_sym in self.initially_covered.items():
            ab.items.append(Atom("hidden-under", [dice_sym, obs_sym]))

        if self.current_conf is not None and np.allclose(
            self.current_conf, DEFAULT_ARM_POS, rtol=1e-4, atol=1e-4
        ):
            ab.items.append(Atom("at-home"))

        if self.stacked_pair is not None:
            dice_sym, block_sym = self.stacked_pair
            ab.items.append(Atom("on", [dice_sym, block_sym]))
            objs = self.world.objects
            cats = self.world.categories
            if (dice_sym in objs and block_sym in objs and
                    cats[objs.index(dice_sym)] == "target_dice" and
                    cats[objs.index(block_sym)] == "target_block"):
                ab.items.append(Atom("target-stacked"))

        return ab


def setup_world_primitives(
    robot, client, n_distractor_blocks, n_distractor_dice,
    n_obstacles, poses=None, **kwargs
) -> Tuple[World, List[pbu.Pose]]:
    floor, obstacles_static = create_default_env(client=client, **kwargs)

    movable = []
    categories = []
    placed_xy = []

    def rand_pose(half_ext, min_dist=0.12):
        for _ in range(200):
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)
            z = half_ext[2]
            if all(np.linalg.norm([x - px, y - py]) > min_dist
                   for px, py in placed_xy):
                placed_xy.append((x, y))
                return pbu.Pose(pbu.Point(x, y, z))
        raise RuntimeError("Could not place object: workspace too crowded")

    b = create_block(TARGET_BLOCK_COLOR, halfExtents=BLOCK_HALF_EXTENTS, client=client)
    movable.append(b); categories.append("target_block")

    d = create_block(TARGET_DICE_COLOR, halfExtents=DICE_HALF_EXTENTS, client=client)
    movable.append(d); categories.append("target_dice")

    for i in range(n_distractor_blocks):
        color = DISTRACTOR_COLORS[i % len(DISTRACTOR_COLORS)]
        b = create_block(color, halfExtents=BLOCK_HALF_EXTENTS, client=client)
        movable.append(b); categories.append("distractor_block")

    for i in range(n_distractor_dice):
        color = DISTRACTOR_COLORS[(i + 2) % len(DISTRACTOR_COLORS)]
        d = create_block(color, halfExtents=DICE_HALF_EXTENTS, client=client)
        movable.append(d); categories.append("distractor_dice")

    for _ in range(n_obstacles):
        o = create_block(OBSTACLE_COLOR, halfExtents=OBSTACLE_HALF_EXTENTS, client=client)
        movable.append(o); categories.append("obstacle")

    if poses is None:
        target_block_pose = rand_pose(BLOCK_HALF_EXTENTS)
        target_dice_pose  = rand_pose(DICE_HALF_EXTENTS)
        generated_poses   = [target_block_pose, target_dice_pose]

        first_obstacle_placed = False

        for i in range(2, len(movable)):
            cat = categories[i]
            if cat == "obstacle" and not first_obstacle_placed:
                td_pos = target_dice_pose[0]
                obs_z  = DICE_HALF_EXTENTS[2] * 2 + OBSTACLE_HALF_EXTENTS[2]
                obs_pose = pbu.Pose(pbu.Point(td_pos[0], td_pos[1], obs_z))
                generated_poses.append(obs_pose)
                placed_xy.append((td_pos[0], td_pos[1]))
                first_obstacle_placed = True
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


def pick_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    (obj, grasp) = action.args
    g = store.get(grasp)
    belief.set_sim(store)
    new_belief = None

    if obj in belief.object_poses:
        pre_confs = grasp_ik(
            belief.world,
            store.get(obj),
            belief.get_pose(obj),
            g,
            obstacles=list(set(belief.world.environment) - {store.get(obj)})
            + [belief.world.floor],
        )
        if pre_confs is not None:
            new_belief = belief.update(
                action, SceneObservation(grasp=grasp, grasp_body=action.args[0]), store
            )

    if new_belief is None:
        return AbstractBeliefSet.from_beliefs([belief], store)

    return AbstractBeliefSet.from_beliefs([new_belief], store)


def pick_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    (obj, grasp) = action.args
    g = store.get(grasp)

    obs = SceneObservation(conf=belief.current_conf)
    belief.set_sim(store)

    pre_confs = grasp_ik(
        belief.world,
        store.get(obj),
        belief.get_pose(obj),
        g,
        obstacles=list(set(belief.world.environment) - {store.get(obj)})
        + [belief.world.floor],
    )

    if pre_confs is None:
        logging.debug("Pick ik fail")
        return state, obs
    else:
        logging.debug("Pick ik success")

    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        pre_confs[0],
        obstacles=[belief.world.floor] + list(set(belief.world.environment)),
    )
    if motion_plan is None:
        return state, obs

    state = pick_execute(state, g, motion_plan, pre_confs, full_close=False)
    obs.grasp = grasp
    obs.grasp_body = obj
    obs.conf = pre_confs[0]
    state.grasp = g
    return state, obs


def place_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    (o, g, p) = action.args
    grasp = store.get(g)
    obj = store.get(o)
    placement_pose = store.get(p)
    belief.set_sim(store)
    new_belief = None

    pre_confs = grasp_ik(
        belief.world,
        obj,
        placement_pose,
        grasp,
        obstacles=list(set(belief.world.environment) - {obj})
        + [belief.world.floor],
    )
    if pre_confs is not None:
        new_belief = belief.update(
            action,
            SceneObservation(moved=o, poses={o: placement_pose}, conf=pre_confs),
            store,
        )

    if new_belief is None:
        return AbstractBeliefSet.from_beliefs([belief], store)

    return AbstractBeliefSet.from_beliefs([new_belief], store)


def place_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    (obj, grasp, place_pose) = action.args
    p = store.get(place_pose)
    g = store.get(grasp)
    obs = SceneObservation(
        conf=belief.current_conf, grasp=g, grasp_body=belief.grasp_body
    )
    belief.set_sim(store)
    conf = ik(belief.world, store.get(obj), p, g)

    if conf is None:
        return state, obs

    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        conf,
        obstacles=[belief.world.floor]
        + list(set(belief.world.environment) - {store.get(obj)}),
        attachments=[grasp_attachment(belief.world, g)],
    )
    if motion_plan is None:
        return state, obs

    state = place_execute(state, motion_plan)
    obs.poses[obj] = p
    obs.conf = conf
    obs.grasp = None
    obs.grasp_body = None
    obs.moved = action.args[0]

    return state, obs


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

    state = place_execute(state, motion_plan)
    obs.poses[top_sym] = stack_pose
    obs.conf    = conf
    obs.grasp   = None
    obs.grasp_body = None
    obs.moved   = top_sym
    return state, obs


def stack_pose_sample_fn_wrapper(sim_world):
    def sample_fn(args: List[str], store: AliasStore):
        top_sym, bottom_sym = args
        top_body    = store.get(top_sym)
        bottom_body = store.get(bottom_sym)

        client = sim_world.client

        bottom_aabb = pbu.get_aabb(bottom_body, client=client)
        top_aabb    = pbu.get_aabb(top_body,    client=client)

        top_half_h = (top_aabb.upper[2] - top_aabb.lower[2]) / 2.0
        stack_z    = bottom_aabb.upper[2] + top_half_h + 0.002

        margin = 0.005
        x_lo = bottom_aabb.lower[0] + margin
        x_hi = bottom_aabb.upper[0] - margin
        y_lo = bottom_aabb.lower[1] + margin
        y_hi = bottom_aabb.upper[1] - margin

        if x_lo >= x_hi or y_lo >= y_hi:
            cx = (bottom_aabb.lower[0] + bottom_aabb.upper[0]) / 2
            cy = (bottom_aabb.lower[1] + bottom_aabb.upper[1]) / 2
        else:
            cx = random.uniform(x_lo, x_hi)
            cy = random.uniform(y_lo, y_hi)

        yaw = random.uniform(0, 2 * np.pi)
        pose = pbu.Pose(pbu.Point(cx, cy, stack_z), pbu.Euler(yaw=yaw))
        return pose

    return sample_fn


def look_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    belief.set_sim(store)

    q = store.get(action.args[2])

    pbu.set_joint_positions(
        belief.world.robot,
        belief.world.robot.get_group_joints(ARM_GROUP, client=belief.world.client),
        q,
        client=belief.world.client,
    )

    camera_image = belief.world.robot.get_image(client=belief.world.client)

    new_seen_voxels = belief.get_new_seen_voxels(camera_image)
    new_seen_points = [
        belief.visibility_grid.center_from_voxel(voxel) for voxel in new_seen_voxels
    ]
    num_occupied = len(belief.visibility_grid.occupied)
    success_count = len(new_seen_points)
    fail_count = num_occupied - success_count
    new_fail_belief = belief.update(action, SceneObservation(camera_image), store)
    abstract_fail = new_fail_belief.abstract(store)

    belief_map = {abstract_fail: [new_fail_belief]}
    ab_counts = {abstract_fail: fail_count}

    if len(new_seen_points) > 0:
        new_success_belief = copy.deepcopy(new_fail_belief)
        new_success_belief.object_poses[action.args[0]] = pbu.Pose(
            pbu.Point(*random.choice(new_seen_points))
        )
        new_success_belief.current_conf = None
        abstract_success = new_success_belief.abstract(store)
        belief_map[abstract_success] = [new_success_belief]
        ab_counts[abstract_success] = success_count

    abstract_bs = AbstractBeliefSet(
        ab_counts=ab_counts,
        belief_map=belief_map,
    )

    return abstract_bs


def look_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    observation = SceneObservation(conf=belief.current_conf)
    belief.set_sim(store)
    look_conf = store.get(action.args[2])
    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        look_conf,
        obstacles=[belief.world.floor] + belief.world.environment,
    )
    if motion_plan is None:
        return state, observation

    if len(motion_plan.path) > 0:
        state.apply_sequence([motion_plan])

    camera_image = state.world.robot.get_image(client=state.world.client)
    obs = obs_from_camera_image(
        belief.world.objects, state.world, camera_image, look_conf, store
    )
    obs.conf = look_conf

    return state, obs


def home_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore
) -> AbstractBeliefSet:
    new_belief = belief.update(
        action,
        SceneObservation(
            grasp=belief.grasp, grasp_body=belief.grasp_body, conf=DEFAULT_ARM_POS
        ),
        store=store,
    )
    return AbstractBeliefSet.from_beliefs([new_belief], store)


def home_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    observation = SceneObservation(
        grasp=belief.grasp, grasp_body=belief.grasp_body, conf=DEFAULT_ARM_POS
    )
    belief.set_sim(store)
    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        DEFAULT_ARM_POS,
        obstacles=[belief.world.floor]
        + list(set(belief.world.environment) - {store.get(belief.grasp_body)}),
        attachments=[grasp_attachment(belief.world, store.get(belief.grasp))],
    )
    if motion_plan is None:
        return state, observation
    state.apply_sequence([motion_plan])
    return state, observation


def look_sample_fn_wrapper(b0, max_attempts=100):
    def look_sample_fn(args: List[str], store: AliasStore):
        (obj_sym,) = args
        look_at_obj = store.get(obj_sym)

        def look_at(camera_position, target_position, up_direction=np.array([0, 0, 1])):
            forward = np.array(target_position) - np.array(camera_position)
            forward = forward / np.linalg.norm(forward)
            left = np.cross(up_direction, forward)
            left = left / np.linalg.norm(left)
            up = np.cross(forward, left)
            up = up / np.linalg.norm(up)
            rotation_matrix = np.array([left, up, forward]).T
            return rotation_matrix

        num_attempts = 0
        while num_attempts < max_attempts:
            num_attempts += 1
            client = b0.world.client
            table_aabb = pbu.get_aabb(b0.world.floor, client=client)
            object_pose = pbu.get_pose(look_at_obj, client=client)

            target_world_T_camera = pbu.multiply(
                pbu.Pose(
                    pbu.Point(
                        x=np.random.normal(0, 0.1),
                        y=np.random.normal(0, 0.1),
                        z=np.random.uniform(0.2, 0.3),
                    )
                ),
                object_pose,
            )
            r = look_at(target_world_T_camera[0], object_pose[0])
            rotation = R.from_matrix(r)
            quaternion = rotation.as_quat()

            target_world_T_camera = pbu.multiply(
                pbu.Pose(target_world_T_camera[0]), ([0, 0, 0], quaternion)
            )
            target_world_T_gripper = target_world_T_camera
            current_world_T_camera = pbu.get_link_pose(
                b0.world.robot,
                pbu.link_from_name(b0.world.robot, CAMERA_FRAME, client=client),
                client=client,
            )
            current_world_T_gripper = pbu.get_link_pose(
                b0.world.robot,
                pbu.link_from_name(b0.world.robot, PANDA_TOOL_TIP, client=client),
                client=client,
            )
            camera_T_gripper = pbu.multiply(
                pbu.invert(current_world_T_camera), current_world_T_gripper
            )
            target_world_T_gripper = pbu.multiply(
                target_world_T_camera, camera_T_gripper
            )

            arm_path = plan_workspace_motion(
                b0.world.robot,
                [target_world_T_gripper],
                obstacles=[b0.world.floor],
                client=b0.world.client,
            )
            if arm_path is None:
                continue
            else:
                pbu.set_joint_positions(
                    b0.world.robot,
                    b0.world.robot.get_group_joints(ARM_GROUP, client=b0.world.client),
                    arm_path[0],
                    client=b0.world.client,
                )
                return arm_path[0]

        return None

    return look_sample_fn


def grasp_sample_fn_wrapper(world):
    def grasp_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)
        grasp = get_grasp(
            world,
            obj,
            world.environment,
            grasp_mode=GRASP_MODE,
            use_saved=False,
            client=world.client,
        )
        return grasp

    return grasp_sample_fn


def placement_sample_fn_wrapper(world):
    def place_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)

        table_pose, table_width, table_length, thickness = get_shortened_table_dims()
        table_aabb = pbu.AABB(
            lower=[-table_width / 2, -table_length / 2, -thickness / 2],
            upper=[table_width / 2, table_length / 2, thickness / 2],
        )

        placement = placement_sample(
            world,
            obj,
            world.floor,
            world.environment,
            table_aabb=table_aabb,
            table_pose=table_pose,
            client=world.client,
        )
        return placement

    return place_sample_fn


class FindBlockStackDiceEnv(TampuraEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(FindBlockStackDiceEnv, self).__init__(*args, **kwargs)
        self.world = None

        world_getter = lambda: self.world
        camera_mode = self.config.get("record_camera", "external")
        if camera_mode == "robot":
            capture_fn = make_robot_capture_fn(world_getter)
        else:
            capture_fn = make_external_capture_fn(
                world_getter,
                camera_pos=(0.55, -0.65, 0.60),
                target_pos=(0.25, 0.0, 0.25),
                vertical_fov=30.0,
            )
        self._recorder = FrameRecorder(
            capture_fn=capture_fn,
            save_dir=self.save_dir,
            interval=self.config.get("record_interval", 0.1),
            enabled=self.config.get("record", False),
        )

    def step(self, action, belief, store):
        self.state.frame_callback = self._recorder.make_step_callback()
        result = super().step(action, belief, store)
        self.state.frame_callback = None
        return result

    def wrapup(self):
        self._recorder.make_gif()

    def vis_updated_belief(self, belief: SceneBelief, store: AliasStore):
        belief.visibility_grid.draw_intervals(self.world.client)

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

        # Identify which dice are initially hidden under an obstacle.
        # This mapping is stored permanently so abstract() can emit hidden-under atoms.
        x_tol = OBSTACLE_HALF_EXTENTS[0] + DICE_HALF_EXTENTS[0]
        y_tol = OBSTACLE_HALF_EXTENTS[1] + DICE_HALF_EXTENTS[1]
        for dice_sym, dice_cat in zip(self.sim_world.objects, self.sim_world.categories):
            if dice_cat not in ("target_dice", "distractor_dice"):
                continue
            if dice_sym in b.object_poses:
                continue  # visible dice are not hidden
            dice_pose = pbu.get_pose(store.get(dice_sym), client=self.world.client)
            dx, dy, dz = dice_pose[0]
            for obs_sym, obs_cat in zip(self.sim_world.objects, self.sim_world.categories):
                if obs_cat != "obstacle":
                    continue
                obs_pose = pbu.get_pose(store.get(obs_sym), client=self.world.client)
                ox, oy, oz = obs_pose[0]
                if abs(ox - dx) < x_tol and abs(oy - dy) < y_tol and oz > dz + DICE_HALF_EXTENTS[2]:
                    b.initially_covered[dice_sym] = obs_sym
                    break

        self.starting_belief = b

        if self.vis:
            b.visibility_grid.draw_intervals(self.world.client)

        return b, store

    def get_problem_spec(self):
        predicates = [
            Predicate("known-pose",      ["physical"]),
            Predicate("holding",         ["physical"]),
            Predicate("at-start",        ["physical"]),
            Predicate("object-pose",     ["physical", "pose"]),
            Predicate("is-target-block", ["physical"]),
            Predicate("is-target-dice",  ["physical"]),
            Predicate("target-stacked",  []),
            Predicate("on",              ["physical", "physical"]),
            Predicate("at-home",         []),
            Predicate("looking-conf",    ["physical", "conf"]),
            Predicate("moved",           ["physical"]),
            Predicate("object-grasp",    ["physical", "grasp"]),
            Predicate("at-grasp",        ["physical", "grasp"]),
            Predicate("stacked-pose",    ["physical", "physical", "pose"]),
            Predicate("hidden-under",    ["physical", "physical"]),
        ]

        stream_schemas = [
            StreamSchema(
                name="look-conf-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?q",
                output_type="conf",
                certified=[Atom("looking-conf", ["?o", "?q"])],
                sample_fn=look_sample_fn_wrapper(self.starting_belief),
            ),
            StreamSchema(
                name="grasp-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?g",
                output_type="grasp",
                certified=[Atom("object-grasp", ["?o", "?g"])],
                sample_fn=grasp_sample_fn_wrapper(self.sim_world),
            ),
            StreamSchema(
                name="place-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?p",
                output_type="pose",
                certified=[Atom("object-pose", ["?o", "?p"])],
                sample_fn=placement_sample_fn_wrapper(self.sim_world),
            ),
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

        holding = Exists(Atom("holding", ["?obj"]), ["?obj"], ["physical"])
        action_schemas = [
            ActionSchema(
                name="pick",
                inputs=["?o", "?g"],
                input_types=["physical", "grasp"],
                preconditions=[
                    Not(Atom("moved", ["?o"])),
                    Not(holding),
                    Atom("known-pose", ["?o"]),
                    Atom("object-grasp", ["?o", "?g"]),
                ],
                effects_fn=pick_effects_fn,
                execute_fn=pick_execute_fn,
                effects=[Not(Atom("at-home"))],
                verify_effects=[
                    Atom("holding", ["?o"]),
                    Atom("at-grasp", ["?o", "?g"]),
                ],
            ),
            ActionSchema(
                name="place",
                inputs=["?o", "?g", "?p"],
                input_types=["physical", "grasp", "pose"],
                preconditions=[
                    Atom("holding", ["?o"]),
                    Atom("at-grasp", ["?o", "?g"]),
                    Atom("object-pose", ["?o", "?p"]),
                ],
                effects_fn=place_effects_fn,
                execute_fn=place_execute_fn,
                effects=[
                    Atom("moved", ["?o"]),
                    Not(Atom("at-home")),
                    Not(Atom("holding", ["?o"])),
                    Not(Atom("at-grasp", ["?o", "?g"])),
                    Atom("known-pose", ["?o"]),
                ],
            ),
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
            ),
            ActionSchema(
                name="look",
                inputs=["?o1", "?o2", "?q"],
                input_types=["physical", "physical", "conf"],
                preconditions=[
                    Atom("looking-conf",  ["?o2", "?q"]),
                    Not(Atom("known-pose", ["?o1"])),
                    Atom("known-pose",    ["?o2"]),
                    Atom("moved",         ["?o2"]),
                    Atom("hidden-under",  ["?o1", "?o2"]),
                    Not(holding),
                ],
                effects_fn=look_effects_fn,
                execute_fn=look_execute_fn,
                effects=[Not(Atom("at-home"))],
                verify_effects=[Atom("known-pose", ["?o1"])],
            ),
            ActionSchema(
                name="go-home",
                inputs=[],
                input_types=[],
                preconditions=[
                    holding,
                    Not(Atom("at-home")),
                ],
                effects_fn=home_effects_fn,
                execute_fn=home_execute_fn,
                effects=[Atom("at-home")],
            ),
            NoOp(),
        ]

        # Symk needs to derive the goal via PDDL-visible effects.
        # "target-stacked" is computed in abstract() but never set as an action effect,
        # so the planner cannot plan towards it.  Express the goal directly using the
        # PDDL-visible predicates that stack adds (on) and the initial atoms (is-target-*).
        reward = Exists(
            And([
                Atom("is-target-dice",  ["?top"]),
                Atom("is-target-block", ["?bottom"]),
                Atom("on",              ["?top", "?bottom"]),
            ]),
            ["?top", "?bottom"],
            ["physical", "physical"],
        )

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            stream_schemas=stream_schemas,
            reward=reward,
        )

        return spec


register_env("find_block_stack_dice", FindBlockStackDiceEnv)
register_env("find_block_stack_dice_hard", FindBlockStackDiceEnv)
