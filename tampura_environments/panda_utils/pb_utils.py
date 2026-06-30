from __future__ import print_function

import itertools
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple, TypeAlias

import imageio
import numpy as np
import pybullet as p  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from scipy.interpolate import CubicSpline, make_interp_spline, make_lsq_spline
from scipy.spatial import ConvexHull  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

DEFAULT_CLIENT = None
CLIENT = 0
BASE_LINK = -1
STATIC_MASS = 0
MAX_DISTANCE = 0
NULL_ID = -1
INFO_FROM_BODY = {}
UNKNOWN_FILE = "unknown_file"
DEFAULT_RADIUS = 0.5
DEFAULT_EXTENTS = [1, 1, 1]
DEFAULT_SCALE = [1, 1, 1]
DEFAULT_NORMAL = [0, 0, 1]
DEFAULT_HEIGHT = 1
GRASP_LENGTH = 0.04
MAX_GRASP_WIDTH = np.inf
DEFAULT_SPEED_FRACTION = 0.3
DEFAULT_MESH = ""
_EPS = np.finfo(float).eps * 4.0
GRAVITY = 9.8


PointType = Tuple[float, float, float]
EulerType = PointType
QuatType = Tuple[float, float, float, float]
PoseType = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


@dataclass
class RGB:
    """RGB カラー（各チャネル 0.0〜1.0）。"""

    red: float
    green: float
    blue: float

    def __iter__(self):
        return iter(asdict(self).values())


@dataclass
class RGBA:
    """RGBA カラー（各チャネル 0.0〜1.0、alpha は不透明度）。"""

    red: float
    green: float
    blue: float
    alpha: float

    def __iter__(self):
        return iter(asdict(self).values())


@dataclass
class Mesh:
    """3D メッシュ（頂点リストと三角面インデックスリスト）。"""

    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]


RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 0.1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)
LIGHT_GREY = RGBA(0.75, 0.75, 0.75, 1)

ACHROMATIC_COLORS = {
    "white": WHITE,
    "grey": GREY,
    "black": BLACK,
}

CHROMATIC_COLORS = {
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
}


@dataclass
class Interval:
    """スカラー値の区間 [lower, upper]。関節可動範囲などに使う。"""

    lower: float
    upper: float


UNIT_LIMITS = Interval(0.0, 1.0)
CIRCULAR_LIMITS = Interval(-np.pi, np.pi)
UNBOUNDED_LIMITS = Interval(-np.inf, np.inf)


def angle_between(vec1, vec2):
    """2つのベクトル間の角度（ラジアン）を返す。"""
    inner_product = np.dot(vec1, vec2) / (get_length(vec1) * get_length(vec2))
    return math.acos(clip(inner_product, min_value=-1.0, max_value=+1.0))


def get_image_aabb(camera_matrix):
    """カメラ行列から画像空間の AABB を返す。"""
    upper = np.array(dimensions_from_camera_matrix(camera_matrix)) - 1
    lower = np.zeros(upper.shape)
    return AABB(lower, upper)


def get_aabb_volume(aabb):
    """AABB の体積を返す。"""
    if aabb_empty(aabb):
        return 0.0
    return np.prod(get_aabb_extent(aabb))


def quat_from_euler(euler):
    """オイラー角（ロール・ピッチ・ヨー）をクォータニオンに変換する。"""
    return p.getQuaternionFromEuler(euler)


def pose_from_pose2d(pose2d, z=0.0):
    """2D 姿勢（x, y, theta）を 3D Pose に変換する。"""
    x, y, theta = pose2d
    return Pose(Point(x=x, y=y, z=z), Euler(yaw=theta))


def Euler(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    """ロール・ピッチ・ヨーからオイラー角配列を作成する。"""
    return np.array([roll, pitch, yaw])


def Point(x=0.0, y=0.0, z=0.0):
    """3D 座標点を numpy 配列として作成する。"""
    return np.array([x, y, z])


def ray_from_pixel(camera_matrix, pixel):
    """ピクセル座標からカメラ座標系のレイベクトルを求める。"""
    return np.linalg.inv(camera_matrix).dot(np.append(pixel, 1))


def Pose(point: Optional[PointType] = None, euler: Optional[EulerType] = None):
    """位置とオイラー角から (position, quaternion) の Pose タプルを作成する。"""
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)


@dataclass
class JointState:
    """PyBullet の getJointState 戻り値を保持するデータクラス。"""

    jointPosition: float
    jointVelocity: float
    jointReactionForces: Tuple[float, float, float, float, float, float]
    appliedJointMotorTorque: float


@dataclass
class Pixel:
    """画像上のピクセル座標（行・列）。"""

    row: int
    column: int


@dataclass
class CollisionInfo:
    """PyBullet の getContactPoints 戻り値を保持するデータクラス。"""

    contactFlag: int
    bodyUniqueIdA: int
    bodyUniqueIdB: int
    linkIndexA: int
    linkIndexB: int
    positionOnA: Tuple[float, float, float]
    positionOnB: Tuple[float, float, float]
    contactNormalOnB: Tuple[float, float, float]
    contactDistance: float
    normalForce: float
    lateralFriction1: float
    lateralFrictionDir1: Tuple[float, float, float]
    lateralFriction2: float
    lateralFrictionDir2: Tuple[float, float, float]


@dataclass
class CollisionPair:
    """衝突判定対象の物体 ID とリンクインデックスの組み合わせ。"""

    body: int
    links: List[int]

    def __int__(self):
        return int(self.body)


@dataclass
class AABB:
    """軸平行バウンディングボックス（Axis-Aligned Bounding Box）。

    ワールド座標系の軸に平行な辺を持つ直方体で、物体の占有空間を近似する。
    衝突判定・配置サンプリング・可視性チェックなどで広く使われる。

    Attributes:
        lower: バウンディングボックスの最小座標 (x_min, y_min, z_min)。
        upper: バウンディングボックスの最大座標 (x_max, y_max, z_max)。
    """

    lower: list
    upper: list


@dataclass
class OOBB:
    """物体姿勢に基づくバウンディングボックス（Object-Oriented Bounding Box）。

    ローカル座標系で定義した AABB と、それをワールド座標系に配置するための
    姿勢を組み合わせた表現。AABB と異なり物体の回転を考慮した形状近似が可能。

    Attributes:
        aabb: ローカル（物体）座標系での AABB。各軸の最小・最大座標。
        pose: ローカル座標系をワールド座標系に変換する姿勢 (position, quaternion)。
    """

    aabb: AABB
    pose: Pose


@dataclass
class CollisionShapeData:
    """PyBullet の getCollisionShapeData 戻り値を保持するデータクラス。"""

    object_unique_id: int
    linkIndex: int
    geometry_type: int
    dimensions: list
    filename: str
    local_frame_pos: List[float]
    local_frame_orn: List[float]


@dataclass
class BodyInfo:
    """PyBullet の getBodyInfo 戻り値を保持するデータクラス。"""

    base_name: str
    body_name: str


@dataclass
class JointInfo:
    """PyBullet の getJointInfo 戻り値を保持するデータクラス。"""

    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: List[str]
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: float
    jointAxis: float
    parentFramePos: float
    parentFrameOrn: float
    parentIndex: float


@dataclass
class LinkState:
    """PyBullet の getLinkState 戻り値を保持するデータクラス。"""

    linkWorldPosition: Tuple[float, float, float]
    linkWorldOrientation: Tuple[float, float, float, float]
    localInertialFramePosition: Tuple[float, float, float]
    localInertialFrameOrientation: Tuple[float, float, float, float]
    worldLinkFramePosition: Tuple[float, float, float]
    worldLinkFrameOrientation: Tuple[float, float, float, float]


@dataclass
class CameraImage:
    """カメラで撮影した RGB・深度・セグメンテーション画像と付属メタデータ。"""

    rgbPixels: Any
    depthPixels: Any
    segmentationMaskBuffer: Any
    camera_pose: PoseType
    camera_matrix: Any


@dataclass
class DynamicsInfo:
    """PyBullet の getDynamicsInfo 戻り値（質量・摩擦係数など）を保持するデータクラス。"""

    mass: float
    lateral_friction: float
    local_inertia_diagonal: Tuple[float, float, float]
    local_inertial_pos: Tuple[float, float, float]
    local_inertial_orn: Tuple[float, float, float, float]
    restitution: float
    rolling_friction: float
    spinning_friction: float
    contact_damping: float
    contact_stiffness: float


@dataclass
class VisualShapeData:
    """PyBullet の getVisualShapeData 戻り値を保持するデータクラス。"""

    objectUniqueId: int
    linkIndex: int
    visualGeometryType: int
    dimensions: Optional[Tuple[float, ...]]
    meshAssetFileName: str
    localVisualFrame_position: Tuple[float, float, float]
    localVisualFrame_orientation: Tuple[float, float, float, float]
    rgbaColor: Tuple[float, float, float, float]
    textureUniqueId: int


@dataclass
class ModelInfo:
    """ロードした URDF / メッシュモデルのメタ情報（パス・スケール・固定ベースフラグ）。"""

    name: str
    path: str
    fixed_base: bool
    scale: float


@dataclass
class MouseEvent:
    """PyBullet の getMouseEvents 戻り値を保持するデータクラス。"""

    eventType: str
    mousePosX: int
    mousePosY: int
    buttonIndex: int
    buttonState: str


@dataclass
class ConstraintInfo:
    """PyBullet の getConstraintInfo 戻り値を保持するデータクラス。"""

    parentBodyUniqueId: int
    parentJointIndex: int
    childBodyUniqueId: int
    childLinkIndex: int
    constraintType: int
    jointAxis: Tuple[float, float, float]
    jointPivotInParent: Tuple[float, float, float]
    jointPivotInChild: Tuple[float, float, float]
    jointFrameOrientationParent: Tuple[float, float, float, float]
    jointFrameOrientationChild: Tuple[float, float, float, float]
    maxAppliedForce: float


def remove_redundant(path, tolerance=1e-3):
    """パスから隣接する重複コンフィギュレーションを除去する。"""
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = get_difference(new_path[-1], np.array(conf))
        if not np.allclose(
            np.zeros(len(difference)), difference, atol=tolerance, rtol=0
        ):
            new_path.append(conf)
    return new_path


def compute_min_duration(distance, max_velocity, acceleration):
    """距離・最大速度・加速度から最小移動時間を計算する。"""
    if distance == 0:
        return 0
    max_ramp_duration = max_velocity / acceleration
    if acceleration == np.inf:
        # return distance / max_velocity
        ramp_distance = 0.0
    else:
        ramp_distance = 0.5 * acceleration * math.pow(max_ramp_duration, 2)
    remaining_distance = distance - 2 * ramp_distance
    if 0 <= remaining_distance:  # zero acceleration
        remaining_time = remaining_distance / max_velocity
        total_time = 2 * max_ramp_duration + remaining_time
    else:
        half_time = np.sqrt(distance / acceleration)
        total_time = 2 * half_time
    return total_time


def compute_position(ramp_time, max_duration, acceleration, t):
    """台形速度プロファイルにおける時刻 t の移動位置を返す。"""
    velocity = acceleration * ramp_time
    max_time = max_duration - 2 * ramp_time
    t1 = clip(t, 0, ramp_time)
    t2 = clip(t - ramp_time, 0, max_time)
    t3 = clip(t - ramp_time - max_time, 0, ramp_time)
    # assert t1 + t2 + t3 == t
    return (
        0.5 * acceleration * math.pow(t1, 2)
        + velocity * t2
        + velocity * t3
        - 0.5 * acceleration * math.pow(t3, 2)
    )


def is_center_on_aabb(
    body, bottom_aabb: AABB, above_epsilon=1e-2, below_epsilon=0.0, **kwargs
):
    """物体の底面中心が AABB の上面に乗っているかどうかを返す。"""
    assert (0 <= above_epsilon) and (0 <= below_epsilon)
    center, extent = get_center_extent(body, **kwargs)  # TODO: approximate_as_prism
    base_center = center - np.array([0, 0, extent[2]]) / 2
    top_z_min = base_center[2]
    bottom_z_max = bottom_aabb.upper[2]
    return (
        (bottom_z_max - abs(below_epsilon))
        <= top_z_min
        <= (bottom_z_max + abs(above_epsilon))
    ) and (aabb_contains_point(base_center[:2], aabb2d_from_aabb(bottom_aabb)))


def compute_ramp_duration(distance, acceleration, duration):
    """指定した移動時間内に距離を走行するためのランプ（加速）時間を計算する。"""
    discriminant = max(
        0, math.pow(duration * acceleration, 2) - 4 * distance * acceleration
    )
    velocity = 0.5 * (duration * acceleration - math.sqrt(discriminant))  # +/-
    # assert velocity <= max_velocity
    ramp_time = velocity / acceleration
    predicted_distance = velocity * (
        duration - 2 * ramp_time
    ) + acceleration * math.pow(ramp_time, 2)
    assert abs(distance - predicted_distance) < 1e-6
    return ramp_time


def aabb_from_oobb(oobb: OOBB):
    """OOBB をワールド座標系の AABB に変換する。"""
    return aabb_from_points(tform_points(oobb.pose, get_aabb_vertices(oobb.aabb)))


def add_ramp_waypoints(
    differences, accelerations, q1, duration, sample_step, waypoints, time_from_starts
):
    """台形プロファイルに基づいて中間ウェイポイントをサンプリングしてリストに追加する。"""
    dim = len(q1)
    distances = np.abs(differences)
    time_from_start = time_from_starts[-1]

    ramp_durations = [
        compute_ramp_duration(distances[idx], accelerations[idx], duration)
        for idx in range(dim)
    ]
    directions = np.sign(differences)
    for t in np.arange(sample_step, duration, sample_step):
        positions = []
        for idx in range(dim):
            distance = compute_position(
                ramp_durations[idx], duration, accelerations[idx], t
            )
            positions.append(q1[idx] + directions[idx] * distance)
        waypoints.append(positions)
        time_from_starts.append(time_from_start + t)
    return waypoints, time_from_starts


def is_center_stable(body, surface, **kwargs):
    """物体の底面中心がサーフェスの AABB 上に安定して乗っているかを返す。"""
    return is_center_on_aabb(body, get_aabb(surface), **kwargs)


def set_texture(body, texture=None, link=BASE_LINK, shape_index=NULL_ID, client=None):
    """物体のリンクにテクスチャを設定する。"""
    client = client or DEFAULT_CLIENT
    if texture is None:
        texture = NULL_ID
    return client.changeVisualShape(
        body,
        link,
        shapeIndex=shape_index,
        textureUniqueId=texture,
        physicsClientId=CLIENT,
    )


def ramp_retime_path(
    path, max_velocities, acceleration_fraction=np.inf, sample_step=None, **kwargs
):
    """台形速度プロファイルでパスを再タイミングし、ウェイポイントと開始時刻リストを返す。"""
    assert np.all(max_velocities)
    accelerations = max_velocities * acceleration_fraction
    dim = len(max_velocities)

    # Assuming instant changes in accelerations
    waypoints = [path[0]]
    time_from_starts = [0.0]
    for q1, q2 in get_pairs(path):
        differences = get_difference(q1, q2)  # assumes not circular anymore
        # differences = difference_fn(q1, q2)
        distances = np.abs(differences)
        duration = max(
            [
                compute_min_duration(
                    distances[idx], max_velocities[idx], accelerations[idx]
                )
                for idx in range(dim)
            ]
            + [0.0]
        )
        time_from_start = time_from_starts[-1]
        if sample_step is not None:
            waypoints, time_from_starts = add_ramp_waypoints(
                differences,
                accelerations,
                q1,
                duration,
                sample_step,
                waypoints,
                time_from_starts,
            )
        waypoints.append(q2)
        time_from_starts.append(time_from_start + duration)
    return waypoints, time_from_starts


def get_max_velocity(body, joint, **kwargs):
    """指定した関節の最大速度を返す。"""
    # Note that the maximum velocity is not used in actual motor control commands at the moment.
    return get_joint_info(body, joint, **kwargs).jointMaxVelocity


def get_max_velocities(body, joints, **kwargs):
    """複数関節の最大速度をタプルで返す。"""
    return tuple(get_max_velocity(body, joint, **kwargs) for joint in joints)


def retime_trajectory(
    robot,
    joints,
    path,
    only_waypoints=False,
    velocity_fraction=DEFAULT_SPEED_FRACTION,
    **kwargs,
):
    """
    :param robot:
    :param joints:
    :param path:
    :param velocity_fraction: fraction of max_velocity
    :return:
    """
    path = adjust_path(robot, joints, path, **kwargs)
    if only_waypoints:
        path = waypoints_from_path(path)
    max_velocities = velocity_fraction * np.array(
        get_max_velocities(robot, joints, **kwargs)
    )
    return ramp_retime_path(path, max_velocities, **kwargs)


def approximate_spline(time_from_starts, path, k=3, approx=np.inf):
    """タイムスタンプとパスから B スプライン近似曲線を生成する。"""
    x = time_from_starts
    if approx == np.inf:
        positions = make_interp_spline(
            time_from_starts, path, k=k, t=None, bc_type="clamped"
        )
        positions.x = positions.t[positions.k : -positions.k]
    else:
        assert approx <= len(x) - 2 * k
        t = np.r_[
            (x[0],) * (k + 1),
            np.linspace(x[0], x[-1], num=2 + approx, endpoint=True)[1:-1],
            (x[-1],) * (k + 1),
        ]
        w = None
        positions = make_lsq_spline(x, path, t, k=k, w=w)
    positions.x = positions.t[positions.k : -positions.k]
    return positions


def interpolate_path(
    robot,
    joints,
    path,
    velocity_fraction=DEFAULT_SPEED_FRACTION,
    k=1,
    bspline=False,
    dump=False,
    **kwargs,
):
    """ロボットのパスを補間して連続的な位置曲線を返す。"""
    path, time_from_starts = retime_trajectory(
        robot,
        joints,
        path,
        velocity_fraction=velocity_fraction,
        sample_step=None,
        **kwargs,
    )
    if k == 3:
        if bspline:
            positions = approximate_spline(time_from_starts, path, k=k, **kwargs)
        else:
            positions = CubicSpline(
                time_from_starts, path, bc_type="clamped", extrapolate=False
            )
    else:
        kinds = {1: "linear", 2: "quadratic", 3: "cubic"}  # slinear
        positions = interp1d(
            time_from_starts, path, kind=kinds[k], axis=0, assume_sorted=True
        )

    if not dump:
        return positions

    velocities = positions.derivative()
    accelerations = positions.derivative()

    return positions


def get_base_name(body, **kwargs):
    """物体のベースリンク名を文字列で返す。"""
    return get_body_info(body, **kwargs).base_name.decode(encoding="UTF-8")


def get_body_name(body, **kwargs):
    """物体のボディ名を文字列で返す。"""
    return get_body_info(body, **kwargs).body_name.decode(encoding="UTF-8")


def get_name(body, **kwargs):
    """物体のユニークな表示名（ボディ名 + ID）を返す。"""
    name = get_body_name(body, **kwargs)
    if name == "":
        name = "body"
    return "{}{}".format(name, int(body))


def add_body_name(body, name=None, **kwargs):
    """物体の AABB 上部に名前テキストをデバッグ表示する。"""
    if name is None:
        name = get_name(body, **kwargs)
    with PoseSaver(body, **kwargs):
        set_pose(body, unit_pose(), **kwargs)
        aabb = get_aabb(body, **kwargs)
    position = aabb.upper
    return add_text(name, position=position, parent=body, **kwargs)


def get_aabb_area(aabb):
    """AABB の XY 平面上の面積を返す。"""
    return get_aabb_volume(aabb2d_from_aabb(aabb))


def waypoints_from_path(path, difference_fn=None, tolerance=1e-3):
    """パスから方向変化のある点のみを抽出してウェイポイントリストを返す。"""
    if difference_fn is None:
        difference_fn = get_difference
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints


def get_com_pose(body, link):  # COM = center of mass
    """指定リンクの重心（CoM）のワールド座標系姿勢を返す。"""
    if link == BASE_LINK:
        return get_pose(body)
    link_state = get_link_state(body, link)
    # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse()
    return link_state.linkWorldPosition, link_state.linkWorldOrientation


def add_fixed_constraint(
    body, robot, robot_link=BASE_LINK, max_force=None, client=None, **kwargs
):
    """ロボットのリンクと物体の間に固定拘束を作成する。"""
    client = client or DEFAULT_CLIENT
    body_link = BASE_LINK
    body_pose = get_pose(body)
    end_effector_pose = get_com_pose(robot, robot_link)
    grasp_pose = multiply(invert(end_effector_pose), body_pose)
    point, quat = grasp_pose

    constraint = client.createConstraint(
        int(robot),
        robot_link,
        body,
        body_link,  # Both seem to work
        p.JOINT_FIXED,
        jointAxis=unit_point(),
        parentFramePosition=point,
        childFramePosition=unit_point(),
        parentFrameOrientation=quat,
        childFrameOrientation=unit_quat(),
    )
    if max_force is not None:
        client.changeConstraint(constraint, maxForce=max_force)
    return constraint


def remove_debug(debug, client=None, **kwargs):
    """指定したデバッグアイテム（線・テキスト等）を削除する。"""
    client = client or DEFAULT_CLIENT
    client.removeUserDebugItem(debug)


def remove_all_debug(client=None, **kwargs):
    """すべてのデバッグアイテムを削除する。"""
    client = client or DEFAULT_CLIENT
    client.removeAllUserDebugItems()


def is_fixed_base(body, **kwargs):
    """物体が固定ベース（質量ゼロ）かどうかを返す。"""
    return get_mass(body, **kwargs) == STATIC_MASS


def remove_constraint(constraint, client=None, **kwargs):
    """指定した拘束を削除する。"""
    client = client or DEFAULT_CLIENT
    client.removeConstraint(constraint)


def get_urdf_flags(cache=False, cylinder=False, merge=False, sat=False, **kwargs):
    """指定したオプションに対応する PyBullet URDF 読み込みフラグ値を返す。"""
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    if cylinder:
        flags |= p.URDF_USE_IMPLICIT_CYLINDER
    if merge:
        flags |= p.URDF_MERGE_FIXED_LINKS
    if sat:
        flags |= p.URDF_INITIALIZE_SAT_FEATURES
    # flags |= p.URDF_USE_INERTIA_FROM_FILE
    return flags


def load_pybullet(filename, fixed_base=False, scale=1.0, client=None, **kwargs):
    """URDF / SDF / MJCF / OBJ ファイルを PyBullet に読み込んで物体 ID を返す。"""
    # fixed_base=False implies infinite base mass
    client = client or DEFAULT_CLIENT
    with LockRenderer(client=client):
        flags = get_urdf_flags(**kwargs)
        if filename.endswith(".urdf"):
            body = client.loadURDF(
                filename, useFixedBase=fixed_base, flags=flags, globalScaling=scale
            )
        elif filename.endswith(".sdf"):
            body = client.loadSDF(filename)
        elif filename.endswith(".xml"):
            body = client.loadMJCF(filename, flags=flags)
        elif filename.endswith(".bullet"):
            body = client.loadBullet(filename)
        elif filename.endswith(".obj"):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, client=client, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body


def get_connection(client=None):
    """PyBullet クライアントの接続方法（GUI / DIRECT など）を返す。"""
    client = client or DEFAULT_CLIENT
    return client.getConnectionInfo()["connectionMethod"]


def has_gui(client=None, **kwargs):
    """PyBullet クライアントが GUI モードで動作しているかを返す。"""
    client = client or DEFAULT_CLIENT
    return get_connection(client=client) == p.GUI


class Saver(object):
    """シミュレーション状態を保存・復元するための基底クラス。"""

    # TODO: contextlib
    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        self.save()

    def __exit__(self, type, value, traceback):
        self.restore()


def set_renderer(enable, client=None):
    """PyBullet GUI のレンダリングを有効または無効にする。"""
    client = client or DEFAULT_CLIENT
    if not has_gui(client=client):
        return

    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable))


class LockRenderer(Saver):
    """コンテキスト内でレンダリングを一時停止し、オブジェクト追加を高速化する。"""

    # disabling rendering temporary makes adding objects faster
    def __init__(self, client=None, lock=True, **kwargs):
        self.client = client or DEFAULT_CLIENT
        # skip if the visualizer isn't active
        if has_gui(client=self.client) and lock:
            set_renderer(enable=False, client=self.client)

    def restore(self):
        if not has_gui(client=self.client):
            return

        set_renderer(enable=True, client=self.client)


def create_obj(path, scale=1.0, mass=STATIC_MASS, color=GREY, **kwargs):
    """OBJ ファイルからメッシュ物体を生成して物体 ID を返す。"""
    collision_id, visual_id = create_shape(
        get_mesh_geometry(path, scale=scale), color=color, **kwargs
    )
    body = create_body(collision_id, visual_id, mass=mass, **kwargs)
    fixed_base = mass == STATIC_MASS
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(
        None, path, fixed_base, scale
    )  # TODO: store geometry info instead?
    return body


def unit_point():
    """原点を表す 3D 点タプルを返す。"""
    return (0.0, 0.0, 0.0)


def unit_quat():
    """単位回転（回転なし）を表すクォータニオンを返す。"""
    return quat_from_euler([0, 0, 0])  # [X,Y,Z,W]


def unit_pose():
    """原点・回転なしの単位姿勢 (position, quaternion) を返す。"""
    return (unit_point(), unit_quat())


def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    """ジオメトリから衝突形状と視覚形状を作成して ID タプルを返す。"""
    collision_id = (
        create_collision_shape(geometry, pose=pose, **kwargs) if collision else NULL_ID
    )
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs)
    return collision_id, visual_id


def create_body(
    collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS, client=None, **kwargs
):
    """衝突形状・視覚形状・質量からマルチボディを作成して物体 ID を返す。"""
    client = client or DEFAULT_CLIENT
    return client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )


def get_box_geometry(width, length, height):
    """ボックスジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_BOX,
        "halfExtents": [width / 2.0, length / 2.0, height / 2.0],
    }


def create_box(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    """指定サイズのボックス物体を生成して物体 ID を返す。"""
    collision_id, visual_id = create_shape(
        get_box_geometry(w, l, h), color=color, **kwargs
    )
    return create_body(collision_id, visual_id, mass=mass, **kwargs)


def get_cylinder_geometry(radius, height):
    """円柱ジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_CYLINDER,
        "radius": radius,
        "length": height,
    }


def get_sphere_geometry(radius):
    """球ジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_SPHERE,
        "radius": radius,
    }


def get_capsule_geometry(radius, height):
    """カプセルジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_CAPSULE,
        "radius": radius,
        "length": height,
    }


def get_plane_geometry(normal):
    """平面ジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_PLANE,
        "planeNormal": normal,
    }


def get_mesh_geometry(path, scale=1.0):
    """OBJ ファイルパスからメッシュジオメトリの辞書を返す。"""
    return {
        "shapeType": p.GEOM_MESH,
        "fileName": path,
        "meshScale": scale * np.ones(3),
    }


def join_paths(*paths):
    """複数のパス要素を結合して絶対パスを返す。"""
    return os.path.abspath(os.path.join(*paths))


def list_paths(directory):
    """ディレクトリ内のファイルパスをソート済みリストで返す。"""
    return sorted(join_paths(directory, filename) for filename in os.listdir(directory))


def get_min_limit(body, joint, **kwargs):
    """指定した関節の下限値を返す。"""
    return get_joint_limits(body, joint, **kwargs).lower


def get_min_limits(body, joints, **kwargs):
    """複数関節の下限値をリストで返す。"""
    return [get_min_limit(body, joint, **kwargs) for joint in joints]


def get_max_limit(body, joint, **kwargs):
    """指定した関節の上限値を返す。"""
    return get_joint_limits(body, joint, **kwargs).upper


def get_max_limits(body, joints, **kwargs):
    """複数関節の上限値をリストで返す。"""
    return [get_max_limit(body, joint, **kwargs) for joint in joints]


def get_joint_limits(body, joint, **kwargs) -> Interval:
    """指定した関節の可動範囲を Interval で返す。"""
    if is_circular(body, joint, **kwargs):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint, **kwargs)
    return Interval(joint_info.jointLowerLimit, joint_info.jointUpperLimit)


def get_joint_state(body, joint, client=None, **kwargs):
    """指定した関節の状態（位置・速度・反力・トルク）を返す。"""
    client = client or DEFAULT_CLIENT
    return JointState(*client.getJointState(int(body), joint))


def get_joint_position(body, joint, **kwargs):
    """指定した関節の現在角度（位置）を返す。"""
    return get_joint_state(body, joint, **kwargs).jointPosition


def get_joint_velocity(body, joint, **kwargs):
    """指定した関節の現在速度を返す。"""
    return get_joint_state(body, joint, **kwargs).jointVelocity


def get_joint_velocities(body, joints, **kwargs):
    """複数関節の現在速度をタプルで返す。"""
    return tuple(get_joint_velocity(body, joint, **kwargs) for joint in joints)


def get_joint_positions(body, joints, **kwargs):
    """複数関節の現在角度（位置）をタプルで返す。"""
    return tuple(get_joint_position(body, joint, **kwargs) for joint in joints)


def get_camera_matrix(width, height, fx, fy=None):
    """画像サイズと焦点距離からカメラ内部パラメータ行列を作成する。"""
    if fy is None:
        fy = fx
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def set_pose(body, pose, client=None, **kwargs):
    """物体のベース姿勢（位置・クォータニオン）を設定する。"""
    client = client or DEFAULT_CLIENT
    (point, quat) = pose
    client.resetBasePositionAndOrientation(int(body), point, quat)


def pixel_from_ray(camera_matrix, ray):
    """カメラ座標系のレイベクトルを画像ピクセル座標に投影する。"""
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]


def pixel_from_point(camera_matrix, point_camera):
    """カメラ座標系の 3D 点を画像上の Pixel に変換する（画像外は None）。"""
    px, py = pixel_from_ray(camera_matrix, point_camera)
    width, height = dimensions_from_camera_matrix(camera_matrix)
    if (0 <= px < width) and (0 <= py < height):
        r, c = np.floor([py, px]).astype(int)
        return Pixel(r, c)
    return None


def aabb_from_extent_center(extent, center=None):
    """サイズと中心座標から AABB を生成する。"""
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.0
    lower = center - half_extent
    upper = center + half_extent
    return AABB(lower, upper)


def get_aabb_center(aabb):
    """AABB の中心座標を返す。"""
    return (np.array(aabb.lower) + np.array(aabb.upper)) / 2.0


def get_aabb_extent(aabb):
    """AABB の各軸方向のサイズ（幅・奥行き・高さ）を返す。"""
    return np.array(aabb.upper) - np.array(aabb.lower)


def buffer_aabb(aabb, buffer):
    """AABB を指定量だけ各軸方向に拡張した新しい AABB を返す。"""
    if (aabb is None) or (np.isscalar(buffer) and (buffer == 0.0)):
        return aabb
    extent = get_aabb_extent(aabb)
    if np.isscalar(buffer):
        # buffer = buffer - DEFAULT_AABB_BUFFER # TODO: account for the default
        buffer = buffer * np.ones(len(extent))
    new_extent = np.add(2 * buffer, extent)
    center = get_aabb_center(aabb)
    return aabb_from_extent_center(new_extent, center)


def get_link_state(body, link, kinematics=True, velocity=True, client=None, **kwargs):
    """指定リンクの位置・姿勢などの状態を LinkState として返す。"""
    client = client or DEFAULT_CLIENT
    return LinkState(*client.getLinkState(int(body), link))


def get_constraints(client=None, **kwargs):
    """シミュレーション内のすべての拘束 ID リストを返す。"""
    client = client or DEFAULT_CLIENT
    return [client.getConstraintUniqueId(i) for i in range(client.getNumConstraints())]


def get_constraint_info(constraint, client=None, **kwargs):  # getConstraintState
    """指定した拘束の詳細情報を ConstraintInfo として返す。"""
    # TODO: four additional arguments
    client = client or DEFAULT_CLIENT
    return ConstraintInfo(*client.getConstraintInfo(constraint)[:11])


def get_fixed_constraints():
    """シミュレーション内の固定（JOINT_FIXED）拘束の ID リストを返す。"""
    fixed_constraints = []
    for constraint in get_constraints():
        constraint_info = get_constraint_info(constraint)
        if constraint_info.constraintType == p.JOINT_FIXED:
            fixed_constraints.append(constraint)
    return fixed_constraints


def get_link_pose(body, link, **kwargs):
    """指定リンクのワールド座標系姿勢 (position, quaternion) を返す。"""
    if link == BASE_LINK:
        return get_pose(body, **kwargs)
    link_state = get_link_state(body, link, **kwargs)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def aabb_overlap(aabb1, aabb2):
    """2 つの AABB が重なっているかどうかを返す。"""
    if (aabb1 is None) or (aabb2 is None):
        return False
    lower1, upper1 = aabb1
    lower2, upper2 = aabb2
    return all(l1 <= u2 for l1, u2 in zip(lower1, upper2)) and all(
        l2 <= u1 for l2, u1 in zip(lower2, upper1)
    )


def get_buffered_aabb(body, link=None, max_distance=MAX_DISTANCE, **kwargs):
    """物体の AABB を max_distance 分だけ拡張したバッファ済み AABB を返す。"""
    body, links = parse_body(body, link=link)
    return buffer_aabb(
        aabb_union(get_aabbs(body, links=links, **kwargs)), buffer=max_distance
    )


def set_dynamics(body, link=BASE_LINK, client=None, **kwargs):
    """指定リンクの動力学パラメータ（摩擦・質量など）を変更する。"""
    # TODO: iterate over all links
    client = client or DEFAULT_CLIENT
    client.changeDynamics(int(body), link)


def apply_alpha(color, alpha=1.0):
    """RGBA カラーのアルファ値を指定値で上書きした新しいカラーを返す。"""
    if color is None:
        return None
    return RGBA(color.red, color.green, color.blue, alpha)


def interpolate(value1, value2, num_steps=2):
    """value1 から value2 まで num_steps 段階で線形補間した値をジェネレータとして返す。"""
    num_steps = max(num_steps, 2)
    yield value1
    for w in np.linspace(0, 1, num=num_steps, endpoint=True)[1:-1]:
        yield convex_combination(value1, value2, w=w)
    yield value2


def create_visual_shape(
    geometry, pose=unit_pose(), color: RGBA = RED, specular=None, client=None, **kwargs
):
    """ジオメトリと色から PyBullet の視覚形状を作成して ID を返す。"""
    client = client or DEFAULT_CLIENT
    if color is None:  # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        "rgbaColor": list(color),
        "visualFramePosition": point,
        "visualFrameOrientation": quat,
    }
    visual_args.update(geometry)
    # if specular is not None:
    visual_args["specularColor"] = [0, 0, 0]
    return client.createVisualShape(**visual_args)


def create_collision_shape(geometry, pose=unit_pose(), client=None, **kwargs):
    """ジオメトリから PyBullet の衝突形状を作成して ID を返す。"""
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    client = client or DEFAULT_CLIENT
    point, quat = pose
    collision_args = {
        "collisionFramePosition": point,
        "collisionFrameOrientation": quat,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if "length" in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args["height"] = collision_args["length"]
        del collision_args["length"]
    return client.createCollisionShape(**collision_args)


def get_closest_points(
    body1,
    body2,
    link1=None,
    link2=None,
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    client=None,
    **kwargs,
):
    """2 物体間の最近接点情報リストを返す。"""
    client = client or DEFAULT_CLIENT

    if use_aabb and not aabb_overlap(
        get_buffered_aabb(body1, link1, max_distance=max_distance / 2.0),
        get_buffered_aabb(body2, link2, max_distance=max_distance / 2.0),
    ):
        return []
    if (link1 is None) and (link2 is None):
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), distance=max_distance
        )
    elif link2 is None:
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), linkIndexA=link1, distance=max_distance
        )
    elif link1 is None:
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), linkIndexB=link2, distance=max_distance
        )
    else:
        results = client.getClosestPoints(
            bodyA=int(body1),
            bodyB=int(body2),
            linkIndexA=link1,
            linkIndexB=link2,
            distance=max_distance,
        )

    if results is None:
        results = []  # Strange pybullet failure case

    return [CollisionInfo(*info) for info in results]


def body_collision(body1, body2, **kwargs):
    """2 つの物体が衝突しているかどうかを返す。"""
    return len(get_closest_points(body1, body2, **kwargs)) != 0


def get_joint_name(body, joint, **kwargs):
    """指定した関節の名前を文字列で返す。"""
    return get_joint_info(body, joint, **kwargs).jointName.decode("UTF-8")


def joint_from_name(body, name, **kwargs):
    """名前から関節インデックスを取得する。見つからない場合は ValueError を送出する。"""
    for joint in get_joints(body, **kwargs):
        if get_joint_name(body, joint, **kwargs) == name:
            return joint
    raise ValueError(body, name)


def joints_from_names(body, names, **kwargs):
    """名前リストから関節インデックスのタプルを返す。"""
    return tuple(joint_from_name(body, name, **kwargs) for name in names)


def invert(pose):
    """姿勢の逆変換（逆行列に相当する姿勢）を返す。"""
    point, quat = pose
    return p.invertTransform(point, quat)


def get_joint_info(body, joint, client=None, **kwargs):
    """指定した関節の詳細情報を JointInfo として返す。"""
    client = client or DEFAULT_CLIENT
    return JointInfo(*client.getJointInfo(int(body), joint))


def get_link_name(body, link, **kwargs):
    """指定リンクの名前を文字列で返す。"""
    if link == BASE_LINK:
        return get_base_name(body, **kwargs)
    return get_joint_info(body, link, **kwargs).linkName.decode("UTF-8")


def get_num_joints(body, client=None, **kwargs):
    """物体の関節数を返す。"""
    client = client or DEFAULT_CLIENT
    return client.getNumJoints(int(body))


def get_joints(body, **kwargs):
    """物体のすべての関節インデックスをリストで返す。"""
    return list(range(get_num_joints(body, **kwargs)))


get_links = get_joints  # Does not include BASE_LINK


def dimensions_from_camera_matrix(camera_matrix: list):
    """カメラ行列から画像の幅と高さを返す。"""
    cx, cy = np.array(camera_matrix)[:2, 2]
    width, height = (2 * cx + 1), (2 * cy + 1)
    return width, height


def get_collision_data(body, link=BASE_LINK, client=None, **kwargs):
    """指定リンクの衝突形状データリストを返す。"""
    client = client or DEFAULT_CLIENT
    while True:
        try:
            tups = client.getCollisionShapeData(int(body), link)
            break
        except:
            print("Pybullet error getting collision shape. Trying again.")

    return [CollisionShapeData(*tup) for tup in tups]


def can_collide(body, link=BASE_LINK, **kwargs):
    """指定リンクに衝突形状が定義されているかどうかを返す。"""
    return len(get_collision_data(body, link=link, **kwargs)) != 0


def get_all_links(body, **kwargs):
    """ベースリンクを含む全リンクインデックスのリストを返す。"""
    return [BASE_LINK] + list(get_links(body, **kwargs))


def get_aabbs(body, links=None, only_collision=True, **kwargs):
    """指定リンク（または全リンク）の AABB リストを返す。"""
    if links is None:
        links = get_all_links(body, **kwargs)
    if only_collision:
        links = [link for link in links if can_collide(body, link, **kwargs)]
    return [get_aabb(body, link=link, **kwargs) for link in links]


def aabb_union(aabbs: list[AABB]):
    """複数の AABB を包含する最小の AABB を返す。"""
    if not aabbs:
        return None
    if len(aabbs) == 1:
        return aabbs[0]
    d = len(aabbs[0].lower)
    lower = [min(aabb.lower[k] for aabb in aabbs) for k in range(d)]
    upper = [max(aabb.upper[k] for aabb in aabbs) for k in range(d)]
    return AABB(lower, upper)


def get_aabb(body, link=None, client=None, **kwargs) -> AABB:
    """PyBullet から物体（またはリンク）の AABB をワールド座標系で取得する。

    link が None の場合は物体全リンクの AABB を合成して返す。

    Args:
        body: PyBullet の物体 ID。
        link: リンクインデックス。None の場合は物体全体の AABB を返す。

    Returns:
        ワールド座標系での AABB。
    """
    client = client or DEFAULT_CLIENT
    if link is None:
        return aabb_union(get_aabbs(body, client=client, **kwargs))
    return AABB(*client.getAABB(int(body), linkIndex=link))


def get_oobb(body, link=None, client=None, **kwargs) -> OOBB:
    """物体の AABB と姿勢から OOBB を返す。"""
    aabb = get_aabb(body, client=client, **kwargs)
    pose = get_pose(body, client=client, **kwargs)
    return OOBB(aabb, pose)


def get_body_info(body, client=None, **kwargs):
    """物体のベース名・ボディ名を含む BodyInfo を返す。"""
    client = client or DEFAULT_CLIENT
    return BodyInfo(*client.getBodyInfo(int(body)))


def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    """2 つの物体のリンクペアのうち、衝突しているペアが 1 つでもあるかを返す。"""
    if links1 is None:
        links1 = get_all_links(body1, **kwargs)
    if links2 is None:
        links2 = get_all_links(body2, **kwargs)
    for link1, link2 in itertools.product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False


def dict_from_kwargs(**kwargs):
    """キーワード引数を辞書として返す。"""
    return kwargs


def get_control_joint_kwargs(
    body,
    joint,
    position_gain=None,
    max_velocity=None,
    velocity_scale=None,
    max_force=None,
):
    """setJointMotorControl2 向けの制御パラメータ辞書を組み立てて返す。"""
    kwargs = {}
    if position_gain is not None:
        velocity_gain = 0.1 * position_gain
        kwargs.update(
            {
                "positionGain": position_gain,
                "velocityGain": velocity_gain,
            }
        )
    if velocity_scale is not None:
        max_velocity = velocity_scale * get_max_velocity(body, joint)
        kwargs.update(
            {
                "maxVelocity": max_velocity,
            }
        )
    if max_velocity is not None:
        kwargs.update(dict_from_kwargs(maxVelocity=max_velocity))
    if max_force is not None:
        kwargs.update(
            {
                "force": max_force,
            }
        )
    return kwargs


def get_duration_fn(body, joints, velocities=None, norm=np.inf, **kwargs):
    """2 つのコンフィギュレーション間の移動時間を推定するクロージャを返す。"""
    if velocities is None:
        velocities = np.array(get_max_velocities(body, joints, **kwargs))
    difference_fn = get_difference_fn(body, joints, **kwargs)

    def fn(q1, q2):
        distances = np.array(difference_fn(q2, q1))
        durations = np.divide(distances, np.abs(velocities))
        return np.linalg.norm(durations, ord=norm)

    return fn


def waypoint_joint_controller(
    body, joints, target, tolerance=1e-3, time_step=0.1, timeout=np.inf, **kwargs
):
    """目標関節角度へウェイポイントで近づく制御ジェネレータを返す。"""
    assert len(joints) == len(target)
    duration_fn = get_duration_fn(body, joints, **kwargs)
    dt = get_time_step(**kwargs)
    time_elapsed = 0.0
    while time_elapsed < timeout:
        positions = get_joint_positions(body, joints, **kwargs)
        remaining = duration_fn(positions, target)
        if all_close(positions, target, atol=tolerance):
            break
        w = min(remaining, time_step) / remaining
        waypoint = convex_combination(positions, target, w=w)
        control_joints(body, joints, waypoint, **kwargs)
        yield positions
        time_elapsed += dt


def control_joint(body, joint, position=None, velocity=0.0, client=None, **kwargs):
    """単一関節を位置制御モードで目標角度に追従させる。"""
    if position is None:
        position = get_joint_position(body, joint)  # TODO: remove?
    joint_kwargs = get_control_joint_kwargs(body, joint, **kwargs)
    return client.setJointMotorControl2(
        bodyIndex=int(body),  # bodyUniqueId
        jointIndex=joint,
        controlMode=p.POSITION_CONTROL,
        targetPosition=position,
        targetVelocity=velocity,  # Note that the targetVelocity is not the maximum joint velocity
        **joint_kwargs,
    )


def velocity_control_joint(body, joint, velocity=0.0, client=None, **kwargs):
    """単一関節を速度制御モードで目標速度に追従させる。"""
    joint_kwargs = get_control_joint_kwargs(body, joint, **kwargs)
    return client.setJointMotorControl2(
        int(body),
        joint,
        p.VELOCITY_CONTROL,
        targetVelocity=velocity,  # Note that the targetVelocity is not the maximum joint velocity
        **joint_kwargs,
    )


def control_joints(
    body,
    joints,
    positions=None,
    velocities=None,
    position_gain=None,
    velocity_scale=None,
    max_force=None,
    client=None,
    **kwargs,
):
    """複数関節を位置制御モードで目標角度に追従させる。"""
    if positions is None:
        positions = get_joint_positions(body, joints, client=client)
    if velocities is None:
        velocities = [0.0] * len(joints)

    if velocity_scale is not None:
        for i, joint in enumerate(joints):
            control_joint(
                body,
                joint,
                position=positions[i],
                velocity=velocities[i],
                position_gain=position_gain,
                velocity_scale=velocity_scale,
                max_force=max_force,
                client=client,
            )
        return None

    kwargs = {}
    if position_gain is not None:
        velocity_gain = 0.1 * position_gain
        kwargs.update(
            {
                "positionGains": [position_gain] * len(joints),
                "velocityGains": [velocity_gain] * len(joints),
            }
        )
    if max_force is not None:
        max_forces = [max_force] * len(joints)
        kwargs.update(
            {
                "forces": max_forces,
            }
        )
    return client.setJointMotorControlArray(
        bodyUniqueId=int(body),
        jointIndices=joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=positions,
        targetVelocities=velocities,
        **kwargs,
    )


def simulate_controller(controller, max_time=np.inf, **kwargs):
    """コントローラジェネレータを駆動しながらシミュレーションを進める。"""
    sim_dt = get_time_step(**kwargs)
    sim_time = 0.0
    for _ in controller:
        if max_time < sim_time:
            break
        step_simulation(**kwargs)
        sim_time += sim_dt
        yield sim_time


def step_simulation(client=None, **kwargs):
    """シミュレーションを 1 ステップ進める。"""
    client = client or DEFAULT_CLIENT
    client.stepSimulation()


def aabb_contains_point(point, aabb: AABB):
    """点が AABB の内部（境界含む）に存在するかどうかを返す。"""
    return (
        np.less_equal(aabb.lower, point).all()
        and np.less_equal(point, aabb.upper).all()
    )


def oobb_contains_point(point, container: OOBB):
    """点が OOBB の内部に存在するかどうかを返す。"""
    return aabb_contains_point(
        tform_point(invert(container.pose), point), container.aabb
    )


def get_oobb_vertices(oobb: OOBB):
    """OOBB の 8 頂点をワールド座標系で返す。"""
    return tform_points(oobb.pose, get_aabb_vertices(oobb.aabb))


def create_cylinder(radius, height, mass=STATIC_MASS, color=BLUE, **kwargs):
    """指定サイズの円柱物体を生成して物体 ID を返す。"""
    collision_id, visual_id = create_shape(
        get_cylinder_geometry(radius, height), color=color, **kwargs
    )
    return create_body(collision_id, visual_id, mass=mass, **kwargs)


def expand_links(body, **kwargs):
    """CollisionPair のリンクが未指定の場合、全リンクを展開して返す。"""
    pair = parse_body(body)
    if pair.links is None:
        pair.links = get_all_links(pair.body, **kwargs)
    return CollisionPair(pair.body, pair.links)


def pairwise_collision(body1, body2, **kwargs):
    """2 つの物体（または CollisionPair）間で衝突が発生しているかを返す。"""
    if isinstance(body1, CollisionPair) or isinstance(body2, CollisionPair):
        pair1 = expand_links(body1, **kwargs)
        pair2 = expand_links(body2, **kwargs)
        return any_link_pair_collision(
            pair1.body, pair1.links, pair2.body, pair2.links, **kwargs
        )

    return body_collision(body1, body2, **kwargs)


def link_from_name(body, name, **kwargs):
    """リンク名からリンクインデックスを返す。見つからない場合は ValueError を送出する。"""
    if name == get_base_name(body, **kwargs):
        return BASE_LINK
    for link in get_joints(body, **kwargs):
        if get_link_name(body, link, **kwargs) == name:
            return link
    raise ValueError(body, name)


def get_link_names(body, links, **kwargs):
    """リンクインデックスのリストからリンク名リストを返す。"""
    return [get_link_name(body, link, **kwargs) for link in links]


def parse_body(body, link=None):
    """物体 ID または CollisionPair を正規化して CollisionPair として返す。"""
    return body if isinstance(body, CollisionPair) else CollisionPair(body, link)


def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, **kwargs):
    """2 つの物体の指定リンク間で衝突が発生しているかを返す。"""
    return (
        len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0
    )


def ensure_dir(f):
    """ファイルパスの親ディレクトリが存在しない場合は作成する。"""
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def spaced_colors(n, s=1, v=1):
    """色相を均等に分割した n 色の RGBA カラーリストを返す。"""
    import colorsys

    return [
        RGBA(*colorsys.hsv_to_rgb(h, s, v), alpha=1.0)
        for h in np.linspace(0, 1, n, endpoint=False)
    ]


def get_bodies(client=None, **kwargs):
    """シミュレーション内のすべての物体 ID リストを返す。"""
    client = client or DEFAULT_CLIENT
    # Note that all APIs already return body unique ids, so you typically never need to use getBodyUniqueId if you keep track of them
    return [client.getBodyUniqueId(i) for i in range(client.getNumBodies())]


def save_image(filename, rgba):
    """RGBA 画像配列をファイルに保存する。"""
    # Ensure the image is scaled to 0–255 and converted to uint8
    if rgba.dtype == np.float32 or rgba.dtype == np.float64:
        rgba = np.clip(rgba, 0, 1)  # Assuming the float values are in range 0.0–1.0
        rgba = (rgba * 255).astype(np.uint8)
    elif rgba.dtype != np.uint8:
        raise ValueError("Unsupported image data type. Must be float or uint8.")

    # Save the image
    imageio.imwrite(filename, rgba)


def image_from_segmented(segmented, color_from_body=None, **kwargs):
    """セグメンテーションマスクから物体ごとに色付けした RGB 画像を生成する。"""
    if color_from_body is None:
        bodies = get_bodies(**kwargs)
        color_from_body = dict(zip(bodies, spaced_colors(len(bodies))))
    image = np.zeros(segmented.shape[:2] + (3,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            body, link = segmented[r, c, :]
            image[r, c, :] = list(color_from_body.get(body, BLACK))[:3]  # TODO: alpha
    return image


def save_camera_images(camera_image, directory="", prefix="", client=None, **kwargs):
    """カメラ画像（RGB・深度・セグメンテーション）をディレクトリに保存する。"""
    # safe_remove(directory)
    ensure_dir(directory)
    depth_image = camera_image.depthPixels
    seg_image = camera_image.segmentationMaskBuffer
    save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]
    depth_image = (
        (depth_image - np.min(depth_image))
        / (np.max(depth_image) - np.min(depth_image))
        * 255
    ).astype(np.uint8)
    save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]
    if seg_image is None:
        return None

    segmented_image = image_from_segmented(seg_image, client=client)
    save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image


def safe_zip(sequence1, sequence2):  # TODO: *args
    """長さが等しいことを確認してから 2 つのシーケンスをzipする。"""
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))


def violates_limit(body, joint, value, **kwargs):
    """関節の値が可動範囲を逸脱しているかどうかを返す。"""
    # TODO: custom limits
    if is_circular(body, joint, **kwargs):
        return False
    lower, upper = get_joint_limits(body, joint, **kwargs)
    return (value < lower) or (upper < value)


def violates_limits(body, joints, values, **kwargs):
    """いずれかの関節が可動範囲を逸脱しているかどうかを返す。"""
    return any(
        violates_limit(body, joint, value, **kwargs)
        for joint, value in zip(joints, values)
    )


def set_joint_positions(body, joints, values, **kwargs):
    """複数の関節角度を一括して設定する。"""
    for joint, value in safe_zip(joints, values):
        set_joint_position(body, joint, value, **kwargs)


def set_joint_position(body, joint, value, client=None, **kwargs):
    """指定した関節の角度を設定する。"""
    # TODO: remove targetVelocity=0
    client = client or DEFAULT_CLIENT
    client.resetJointState(int(body), joint, targetValue=value, targetVelocity=0)


def stable_z_on_aabb(body, aabb, **kwargs):
    """物体が AABB の上に安定して置かれるときの底面中心の Z 座標を返す。"""
    center, extent = get_center_extent(body, **kwargs)
    return (aabb.upper + extent / 2 + (get_point(body, **kwargs) - center))[2]


def get_center_extent(body, **kwargs):
    """物体の AABB から中心座標とサイズを返す。"""
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)


def get_point(body, **kwargs):
    """物体のベース位置（ワールド座標）を返す。"""
    return get_pose(body, **kwargs)[0]


def get_pose(body, client=None, **kwargs):
    """物体のベース姿勢 (position, quaternion) をワールド座標系で返す。"""
    client = client or DEFAULT_CLIENT
    return client.getBasePositionAndOrientation(int(body))


def multiply(*poses):
    """複数の姿勢を順に合成（行列積）した結果を返す。"""
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose


def point_from_pose(pose):
    """姿勢タプルから位置ベクトルを取り出して返す。"""
    return pose[0]


def quat_from_pose(pose):
    """姿勢タプルからクォータニオンを取り出して返す。"""
    return pose[1]


def tform_point(affine, point):
    """アフィン姿勢（変換行列相当）を点に適用してワールド座標に変換する。"""
    return point_from_pose(multiply(affine, Pose(point=point)))


def get_joint_names(body, joints, **kwargs):
    """関節インデックスのリストから関節名リストを返す。"""
    return [
        get_joint_name(body, joint, **kwargs) for joint in joints
    ]  # .encode('ascii')


def flatten_links(body, links=None, **kwargs):
    """物体とリンクの組み合わせを CollisionPair の集合として展開する。"""
    if links is None:
        links = get_all_links(body, **kwargs)
    return {CollisionPair(body, frozenset([link])) for link in links}


def child_link_from_joint(joint):
    """関節インデックスからその子リンクインデックスを返す（PyBullet では一致）。"""
    link = joint
    return link


def get_link_parent(body, link, **kwargs):
    """指定リンクの親リンクインデックスを返す（ベースリンクの場合は None）。"""
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, **kwargs).parentIndex


def get_all_link_parents(body, **kwargs):
    """各リンクとその親リンクのマッピング辞書を返す。"""
    return {
        link: get_link_parent(body, link, **kwargs)
        for link in get_links(body, **kwargs)
    }


def get_all_link_children(body, **kwargs):
    """各リンクとその子リンクリストのマッピング辞書を返す。"""
    children = {}
    for child, parent in get_all_link_parents(body, **kwargs).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, **kwargs):
    """指定リンクの直接の子リンクインデックスリストを返す。"""
    children = get_all_link_children(body, **kwargs)
    return children.get(link, [])


def get_link_descendants(body, link, test=lambda l: True, **kwargs):
    """指定リンクの子孫リンクインデックスリストを再帰的に返す。"""
    descendants = []
    for child in get_link_children(body, link, **kwargs):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test, **kwargs))
    return descendants


def get_link_subtree(body, link, **kwargs):
    """指定リンク自身とその全子孫リンクのインデックスリストを返す。"""
    return [link] + get_link_descendants(body, link, **kwargs)


def get_moving_links(body, joints, **kwargs):
    """指定した関節群が動かすリンクの全インデックスリストを返す。"""
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link, **kwargs))
    return list(moving_links)


def parent_joint_from_link(link):
    """リンクインデックスからその親関節インデックスを返す（PyBullet では一致）。"""
    # note that link index == joint index
    joint = link
    return joint


def get_field_of_view(camera_matrix):
    """カメラ行列から水平・垂直方向の視野角（ラジアン）を返す。"""
    dimensions = np.array(dimensions_from_camera_matrix(camera_matrix))
    focal_lengths = np.array([camera_matrix[i, i] for i in range(2)])
    return 2 * np.arctan(np.divide(dimensions, 2 * focal_lengths))


def get_joint_descendants(body, link, **kwargs):
    """指定リンクの子孫リンクに対応する関節インデックスリストを返す。"""
    return list(map(parent_joint_from_link, get_link_descendants(body, link, **kwargs)))


def get_movable_joint_descendants(body, link, **kwargs):
    """指定リンクの子孫のうち、可動関節のインデックスリストを返す。"""
    return prune_fixed_joints(
        body, get_joint_descendants(body, link, **kwargs), **kwargs
    )


def get_projection_matrix(
    width, height, vertical_fov, near, far, client=None, **kwargs
):
    """画像サイズ・視野角・クリッピング距離からプロジェクション行列を返す。"""
    client = client or DEFAULT_CLIENT
    aspect = float(width) / height
    fov_degrees = math.degrees(vertical_fov)
    projection_matrix = client.computeProjectionMatrixFOV(
        fov=fov_degrees, aspect=aspect, nearVal=near, farVal=far
    )
    return projection_matrix


def compiled_with_numpy():
    """PyBullet が NumPy サポート付きでコンパイルされているかを返す。"""
    return bool(p.isNumpyEnabled())


def get_image_flags(segment=False, segment_links=False):
    """getCameraImage に渡すセグメンテーション用フラグ値を返す。"""
    if segment:
        if segment_links:
            return p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        return 0  # TODO: adjust output dimension when not segmenting links
    return p.ER_NO_SEGMENTATION_MASK


def point_from_tform(tform):
    """4x4 変換行列から平行移動ベクトルを取り出して返す。"""
    return np.array(tform)[:3, 3]


def demask_pixel(pixel):
    """セグメンテーションピクセル値から物体 ID とリンクインデックスを取り出す。"""
    body = pixel & ((1 << 24) - 1)
    link = (pixel >> 24) - 1
    return body, link


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def matrix_from_tform(tform):
    """4x4 変換行列から 3x3 回転行列を取り出して返す。"""
    return np.array(tform)[:3, :3]


def quat_from_matrix(rot):
    """3x3 回転行列からクォータニオンを返す。"""
    matrix = np.eye(4)
    matrix[:3, :3] = rot[:3, :3]
    return quaternion_from_matrix(matrix)


def pose_from_tform(tform):
    """4x4 変換行列から (position, quaternion) の姿勢タプルを返す。"""
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))


def extract_segmented(seg_image):
    """セグメンテーション画像を (物体ID, リンクID) の 2 チャンネル配列に変換する。"""
    segmented = np.zeros(seg_image.shape + (2,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            pixel = seg_image[r, c]
            segmented[r, c, :] = demask_pixel(pixel)
    return segmented


def get_focal_lengths(dims, fovs):
    """画像サイズと視野角から焦点距離を返す。"""
    return np.divide(dims, np.tan(fovs / 2)) / 2


def get_image(
    camera_pos=None,
    target_pos=None,
    width=640,
    height=480,
    vertical_fov=60.0,
    near=0.02,
    far=5.0,
    tiny=False,
    segment=False,
    client=None,
    **kwargs,
):
    """カメラ位置とターゲット位置から CameraImage（RGB・深度・セグメント）を返す。"""
    client = client or DEFAULT_CLIENT
    up_vector = [0, 0, 1]  # up vector of the camera, in Cartesian world coordinates
    camera_flags = {}
    view_matrix = None
    if (camera_pos is None) or (target_pos is None):
        pass
    else:
        view_matrix = client.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector,
        )
        camera_flags["viewMatrix"] = view_matrix
    projection_matrix = get_projection_matrix(
        width, height, vertical_fov, near, far, client=client
    )

    flags = get_image_flags(segment=segment, **kwargs)
    renderer = p.ER_TINY_RENDERER if tiny else p.ER_BULLET_HARDWARE_OPENGL
    width, height, rgb, d, seg = client.getCameraImage(
        width,
        height,
        projectionMatrix=projection_matrix,
        shadow=False,
        flags=flags,
        renderer=renderer,
        **camera_flags,
    )
    if not compiled_with_numpy():
        rgb = np.reshape(rgb, [height, width, -1])  # 4
        d = np.reshape(d, [height, width])
        seg = np.reshape(seg, [height, width])

    depth = far * near / (far - (far - near) * d)
    segmented = None
    if segment:
        segmented = extract_segmented(seg)

    if view_matrix is None:
        view_matrix = np.identity(4)  # TODO: hack
    camera_tform = np.reshape(view_matrix, [4, 4])  # TODO: transpose?
    camera_tform[:3, 3] = camera_pos
    camera_tform[3, :3] = 0

    view_pose = multiply(pose_from_tform(camera_tform), Pose(euler=Euler(roll=np.pi)))

    focal_length = get_focal_lengths(height, vertical_fov)  # TODO: horizontal_fov
    camera_matrix = get_camera_matrix(width, height, focal_length)

    return CameraImage(rgb, depth, segmented, view_pose, camera_matrix)


def get_image_at_pose(camera_pose, camera_matrix, far=5.0, **kwargs):
    """カメラ姿勢とカメラ行列から CameraImage を取得する。"""
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    camera_point = point_from_pose(camera_pose)
    target_point = tform_point(camera_pose, np.array([0, 0, far]))
    return get_image(
        camera_point,
        target_point,
        width=width,
        height=height,
        vertical_fov=vertical_fov,
        far=far,
        **kwargs,
    )


def get_data_type(data):
    """形状データから PyBullet ジオメトリタイプを返す。"""
    return (
        data.geometry_type
        if isinstance(data, CollisionShapeData)
        else data.visualGeometryType
    )


def get_data_radius(data):
    """形状データからジオメトリの半径を返す。"""
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[1]
    return DEFAULT_RADIUS


def get_joint_inertial_pose(body, joint, **kwargs):
    """指定リンクの慣性フレームのローカル姿勢を返す。"""
    dynamics_info = get_dynamics_info(body, joint, **kwargs)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


def get_data_pose(data):
    """形状データからローカルフレームの姿勢 (position, orientation) を返す。"""
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)


def get_data_extents(data):
    """形状データからボックスの半サイズ（halfExtents）を返す。"""
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return DEFAULT_EXTENTS


def get_data_height(data):
    """形状データから円柱・カプセルの高さを返す。"""
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[0]
    return DEFAULT_HEIGHT


def get_data_scale(data):
    """形状データからメッシュのスケールを返す。"""
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return DEFAULT_SCALE


def get_data_normal(data):
    """形状データから平面の法線ベクトルを返す。"""
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return DEFAULT_NORMAL


def collision_shape_from_data(data, body, link, client=None, **kwargs):
    """CollisionShapeData から PyBullet の衝突形状を再作成して ID を返す。"""
    client = client or DEFAULT_CLIENT
    filename = data.filename.decode(encoding="UTF-8")
    if (data.geometry_type == p.GEOM_MESH) and (filename == UNKNOWN_FILE):
        return NULL_ID
    pose = multiply(
        get_joint_inertial_pose(body, link, client=client), get_data_pose(data)
    )
    point, quat = pose

    return client.createCollisionShape(
        shapeType=data.geometry_type,
        radius=get_data_radius(data),
        halfExtents=np.array(get_data_extents(data)) / 2,
        height=get_data_height(data),
        fileName=filename,
        meshScale=get_data_scale(data),
        planeNormal=get_data_normal(data),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        collisionFramePosition=point,
        collisionFrameOrientation=quat,
    )


def get_length(vec, norm=2):
    """ベクトルのノルム（デフォルトはユークリッドノルム）を返す。"""
    return np.linalg.norm(vec, ord=norm)


def get_difference(p1, p2):
    """2 点の差ベクトル (p2 - p1) を numpy 配列で返す。"""
    return np.array(list(p2)) - np.array(list(p1))


def get_distance(p1, p2, **kwargs):
    """2 点間のユークリッド距離を返す。"""
    return get_length(get_difference(p1, p2), **kwargs)


def get_relative_pose(body, link1, link2=BASE_LINK, **kwargs):
    """link2 から見た link1 の相対姿勢を返す。"""
    world_from_link1 = get_link_pose(body, link1, **kwargs)
    world_from_link2 = get_link_pose(body, link2, **kwargs)
    link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
    return link2_from_link1


def is_circular(body, joint, **kwargs):
    """指定した関節が連続回転（円形可動域）かどうかを返す。"""
    joint_info = get_joint_info(body, joint, **kwargs)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def get_custom_limits(
    body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS, **kwargs
):
    """カスタム制限を適用した各関節の (lower, upper) ペアのイテレータを返す。"""
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint, **kwargs):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint, **kwargs))
    return zip(*[(l.lower, l.upper) for l in joint_limits])


def get_movable_joints(body, **kwargs):
    """物体の可動関節インデックスリストを返す。"""
    return prune_fixed_joints(body, get_joints(body, **kwargs), **kwargs)


def get_joint_type(body, joint, **kwargs):
    """指定した関節のタイプ（FIXED / REVOLUTE / PRISMATIC など）を返す。"""
    return get_joint_info(body, joint, **kwargs).jointType


def is_fixed(body, joint, **kwargs):
    """指定した関節が固定関節かどうかを返す。"""
    return get_joint_type(body, joint, **kwargs) == p.JOINT_FIXED


def is_movable(body, joint, **kwargs):
    """指定した関節が可動（非固定）かどうかを返す。"""
    return not is_fixed(body, joint, **kwargs)


def prune_fixed_joints(body, joints, **kwargs):
    """関節リストから固定関節を除いた可動関節のみのリストを返す。"""
    return [joint for joint in joints if is_movable(body, joint, **kwargs)]


def set_joint_state(body, joint, position, velocity, client=None, **kwargs):
    """指定した関節の角度と速度を同時に設定する。"""
    client = client or DEFAULT_CLIENT
    client.resetJointState(
        int(body), joint, targetValue=position, targetVelocity=velocity
    )


def set_joint_states(body, joints, positions, velocities, **kwargs):
    """複数関節の角度と速度を一括して設定する。"""
    assert len(joints) == len(positions) == len(velocities)
    for joint, position, velocity in zip(joints, positions, velocities):
        set_joint_state(body, joint, position, velocity, **kwargs)


def get_dynamics_info(body, link=BASE_LINK, client=None, **kwargs):
    """指定リンクの動力学情報（質量・摩擦など）を DynamicsInfo として返す。"""
    client = client or DEFAULT_CLIENT
    return DynamicsInfo(*client.getDynamicsInfo(int(body), link)[:10])


def get_client(client=None):
    """クライアントが None の場合はグローバル CLIENT を返す。"""
    if client is None:
        return CLIENT
    return client


def clone_collision_shape(body, link, client=None):
    """指定リンクの衝突形状を複製して新しい衝突形状 ID を返す。"""
    client = get_client(client)
    collision_data = get_collision_data(body, link, client=client)
    if not collision_data:
        return NULL_ID
    assert len(collision_data) == 1
    # TODO: can do CollisionArray
    try:
        return collision_shape_from_data(collision_data[0], body, link, client=client)
    except p.error:
        return NULL_ID


def get_visual_data(body, link=BASE_LINK, client=None, **kwargs):
    """指定リンクの視覚形状データリストを返す。"""
    client = client or DEFAULT_CLIENT
    flags = p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS
    visual_data = [
        VisualShapeData(*tup) for tup in client.getVisualShapeData(int(body), flags)
    ]
    # return visual_data
    return list(filter(lambda d: d.linkIndex == link, visual_data))


def visual_shape_from_data(data, client=None, **kwargs):
    """VisualShapeData から PyBullet の視覚形状を再作成して ID を返す。"""
    client = client or DEFAULT_CLIENT
    if (data.visualGeometryType == p.GEOM_MESH) and (
        data.meshAssetFileName == UNKNOWN_FILE
    ):
        return NULL_ID
    point, quat = get_data_pose(data)
    return client.createVisualShape(
        shapeType=data.visualGeometryType,
        radius=get_data_radius(data),
        halfExtents=np.array(get_data_extents(data)) / 2,
        length=get_data_height(data),
        fileName=data.meshAssetFileName,
        meshScale=get_data_scale(data),
        planeNormal=get_data_normal(data),
        rgbaColor=data.rgbaColor,
        visualFramePosition=point,
        visualFrameOrientation=quat,
    )


def clone_visual_shape(body, link, client=None):
    """指定リンクの視覚形状を複製して新しい視覚形状 ID を返す。"""
    client = client or DEFAULT_CLIENT
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return NULL_ID
    assert len(visual_data) == 1
    return visual_shape_from_data(visual_data[0], client=client)


def get_joint_parent_frame(body, joint, **kwargs):
    """指定関節の親フレームの位置・姿勢を返す。"""
    joint_info = get_joint_info(body, joint, **kwargs)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def get_local_link_pose(body, joint, **kwargs):
    """指定リンクの親フレームから見たローカル姿勢を返す。"""
    parent_joint = get_link_parent(body, joint, **kwargs)
    parent_com = get_joint_parent_frame(body, joint, **kwargs)
    tmp_pose = invert(
        multiply(get_joint_inertial_pose(body, joint, **kwargs), parent_com)
    )
    parent_inertia = get_joint_inertial_pose(body, parent_joint, **kwargs)
    # return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)


def clone_body(body, links=None, collision=True, visual=True, client=None, **kwargs):
    """物体（または指定リンク群）を複製して新しい物体 ID を返す。"""
    client = client or DEFAULT_CLIENT
    if links is None:
        links = get_links(body)
    # movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = (
        get_link_parent(body, links[0], client=client, **kwargs) if links else BASE_LINK
    )
    new_from_original[base_link] = NULL_ID

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = []  # list of local link positions, with respect to parent
    orientations = []  # list of local link orientations, w.r.t. parent
    inertial_positions = []  # list of local inertial frame pos. in link frame
    inertial_orientations = []  # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link, client=client)
        dynamics_info = get_dynamics_info(body, link, client=client)
        masses.append(dynamics_info.mass)
        collision_shapes.append(
            clone_collision_shape(body, link, client=client) if collision else NULL_ID
        )
        visual_shapes.append(
            clone_visual_shape(body, link, client=client) if visual else NULL_ID
        )
        point, quat = get_local_link_pose(body, link, client=client)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1)
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)

    base_dynamics_info = get_dynamics_info(body, base_link, client=client)
    base_point, base_quat = get_link_pose(body, base_link, client=client)
    new_body = client.createMultiBody(
        baseMass=base_dynamics_info.mass,
        baseCollisionShapeIndex=(
            clone_collision_shape(body, base_link, client=client)
            if collision
            else NULL_ID
        ),
        baseVisualShapeIndex=(
            clone_visual_shape(body, base_link, client=client) if visual else NULL_ID
        ),
        basePosition=base_point,
        baseOrientation=base_quat,
        baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
        baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
        linkMasses=masses,
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=visual_shapes,
        linkPositions=positions,
        linkOrientations=orientations,
        linkInertialFramePositions=inertial_positions,
        linkInertialFrameOrientations=inertial_orientations,
        linkParentIndices=parent_indices,
        linkJointTypes=joint_types,
        linkJointAxis=joint_axes,
    )
    # set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(
        range(len(links)), get_joint_positions(body, links, client=client)
    ):
        # TODO: check if movable?
        client.resetJointState(int(new_body), joint, value, targetVelocity=0)
    return new_body


def remove_body(body, client=None, **kwargs):
    """シミュレーションから物体を削除する。"""
    client = client or DEFAULT_CLIENT
    if (CLIENT, body) in INFO_FROM_BODY:
        del INFO_FROM_BODY[CLIENT, body]
    return client.removeBody(int(body))


def set_color(
    body, color: RGBA, link=BASE_LINK, shape_index=NULL_ID, client=None, **kwargs
):
    """物体リンクの表示色を設定する。"""
    client = client or DEFAULT_CLIENT
    if link is None:
        return set_all_color(body, color, **kwargs)
    return client.changeVisualShape(
        int(body), link, shapeIndex=shape_index, rgbaColor=list(color)
    )


def get_color(body, **kwargs) -> RGBA:
    """物体ベースリンクの表示色を RGBA で返す。"""
    # TODO: average over texture
    visual_data = get_visual_data(body, **kwargs)
    if not visual_data:
        # TODO: no viewer implies no visual data
        return None
    return RGBA(*visual_data[0].rgbaColor)


def set_all_color(body, color, **kwargs):
    """物体の全リンクに同一の表示色を設定する。"""
    for link in get_all_links(body, **kwargs):
        set_color(body, color, link, **kwargs)


def get_aabb_vertices(aabb: AABB):
    """AABB の全頂点座標リストを返す（2^d 個）。"""
    d = len(aabb.lower)
    return [
        tuple([aabb.lower, aabb.upper][i[k]][k] for k in range(d))
        for i in itertools.product(range(2), repeat=d)
    ]


def wait_if_gui(*args, **kwargs):
    """GUI モードのときのみユーザー入力を待機する。"""
    if has_gui(**kwargs):
        wait_for_user(*args, **kwargs)


def get_mouse_events():
    """直近のマウスイベントリストを返す。"""
    return list(MouseEvent(*event) for event in p.getMouseEvents())


def update_viewer():
    """ビューアのマウスイベントを処理してフリーズを防ぐ。"""
    get_mouse_events()


def elapsed_time(start_time):
    """開始時刻からの経過時間（秒）を返す。"""
    return time.time() - start_time


def is_darwin():
    """実行環境が macOS かどうかを返す。"""
    return platform.system() == "Darwin"


def wait_for_duration(duration):
    """指定した秒数だけビューアを更新しながら待機する。"""
    t0 = time.time()
    while elapsed_time(t0) <= duration:
        update_viewer()


def randomize(iterable):  # TODO: bisect
    """イテラブルをリストに変換してシャッフルした結果を返す。"""
    sequence = list(iterable)
    random.shuffle(sequence)
    return sequence


def wait_for_user(message="Press enter to continue", **kwargs):
    """ユーザーの Enter キー入力を待機する（macOS GUI では別スレッドで実行）。"""
    if has_gui(**kwargs) and is_darwin():
        return threaded_input(message)
    return input(message)


def threaded_input(*args, **kwargs):
    """ビューア更新を続けながら別スレッドでユーザー入力を待機する。"""
    import threading

    data = []
    thread = threading.Thread(
        target=lambda: data.append(input(*args, **kwargs)), args=[]
    )
    thread.start()
    try:
        while thread.is_alive():
            update_viewer()
    finally:
        thread.join()
    return data[-1]


def get_data_path():
    """pybullet_data パッケージのデータディレクトリパスを返す。"""
    import pybullet_data

    return pybullet_data.getDataPath()


def add_data_path(data_path=None):
    """PyBullet の追加検索パスに pybullet_data のパスを設定する。"""
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path


def get_pitch(point):
    """3D 点の仰角（ピッチ角）をラジアンで返す。"""
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx**2 + dy**2))


def get_yaw(point):
    """3D 点の方位角（ヨー角）をラジアンで返す。"""
    dx, dy = point[:2]
    return np.math.atan2(dy, dx)


def set_camera_pose(camera_point, target_point=np.zeros(3), client=None, **kwargs):
    """PyBullet GUI のカメラをカメラ位置とターゲット位置に基づいて設定する。"""
    client = client or DEFAULT_CLIENT
    delta_point = np.array(target_point) - np.array(camera_point)
    distance = np.linalg.norm(delta_point)
    yaw = get_yaw(delta_point) - np.pi / 2
    pitch = get_pitch(delta_point)
    client.resetDebugVisualizerCamera(
        distance, math.degrees(yaw), math.degrees(pitch), target_point
    )


def get_data_filename(data):
    """形状データからメッシュファイルのパス文字列を返す。"""
    return (
        data.filename
        if isinstance(data, CollisionShapeData)
        else data.meshAssetFileName
    ).decode(encoding="UTF-8")


def convex_hull(points):
    """点群の凸包を計算して Mesh（頂点・面）として返す。"""
    hull = ConvexHull(list(points), incremental=False)
    new_indices = {i: ni for ni, i in enumerate(hull.vertices)}
    vertices = hull.points[hull.vertices, :]
    faces = np.vectorize(lambda i: new_indices[i])(hull.simplices)
    return Mesh(vertices.tolist(), faces.tolist())


def get_unit_vector(vec):
    """ベクトルを正規化して単位ベクトルを返す（ゼロベクトルはそのまま返す）。"""
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def get_normal(v1, v2, v3):
    """三角形の 3 頂点から単位法線ベクトルを返す。"""
    return get_unit_vector(np.cross(np.array(v3) - v1, np.array(v2) - v1))


def orient_face(vertices, face, point=None):
    """三角形面の法線が指定点から外向きになるよう頂点順序を調整する。"""
    if point is None:
        point = np.average(vertices, axis=0)
    v1, v2, v3 = vertices[face]
    normal = get_normal(v1, v2, v3)
    if normal.dot(point - v1) < 0:
        face = face[::-1]
    return tuple(face)


def read(filename):
    """ファイルの全内容を文字列として読み込んで返す。"""
    with open(filename, "r") as f:
        return f.read()


def write(filename, string):
    """文字列をファイルに書き込む。"""
    with open(filename, "w") as f:
        f.write(string)


def mesh_from_points(points, under=True):
    """点群の凸包から法線方向を整えた Mesh を返す。"""
    hull = convex_hull(points)
    vertices, faces = np.array(hull.vertices), np.array(hull.faces)
    centroid = np.average(vertices, axis=0)
    new_faces = [orient_face(vertices, face, point=centroid) for face in faces]
    if under:
        new_faces.extend(map(tuple, map(reversed, list(new_faces))))
    return Mesh(vertices.tolist(), new_faces)


def read_obj(path, decompose=True):
    """OBJ ファイルを読み込み、メッシュ辞書（または単一 Mesh）を返す。"""
    mesh = Mesh([], [])
    meshes = {}
    vertices = []
    faces = []
    for line in read(path).split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "o":
            name = tokens[1]
            mesh = Mesh([], [])
            meshes[name] = mesh
        elif tokens[0] == "v":
            vertex = tuple(map(float, tokens[1:4]))
            vertices.append(vertex)
        elif tokens[0] in ("vn", "s"):
            pass
        elif tokens[0] == "f":
            face = tuple(int(token.split("/")[0]) - 1 for token in tokens[1:])
            faces.append(face)
            mesh.faces.append(face)
    if not decompose:
        return Mesh(vertices, faces)

    for name, mesh in meshes.items():
        indices = sorted({i for face in mesh.faces for i in face})
        mesh.vertices[:] = [vertices[i] for i in indices]
        new_index_from_old = {i2: i1 for i1, i2 in enumerate(indices)}
        mesh.faces[:] = [
            tuple(new_index_from_old[i1] for i1 in face) for face in mesh.faces
        ]
    return meshes


def tform_points(affine, points):
    """アフィン姿勢を点群の各点に適用してワールド座標リストを返す。"""
    return [tform_point(affine, p) for p in points]


def sample_curve(positions_curve, time_step=1e-2):
    """スプライン曲線を time_step 刻みでサンプリングして (時刻, 位置) を yield する。"""
    start_time = positions_curve.x[0]
    end_time = positions_curve.x[-1]
    times = np.append(np.arange(start_time, end_time, step=time_step), [end_time])
    for t in times:
        q = positions_curve(t)
        yield t, q


def get_velocity(body, client=None):
    """物体のベースの線速度と角速度を (linear, angular) として返す。"""
    client = client or DEFAULT_CLIENT
    linear, angular = client.getBaseVelocity(int(body))
    return linear, angular  # [x,y,z], [wx,wy,wz]


def set_velocity(body, linear=None, angular=None, client=None, **kwargs):
    """物体のベースの線速度や角速度を設定する。"""
    client = client or DEFAULT_CLIENT
    if linear is not None:
        client.resetBaseVelocity(int(body), linearVelocity=linear)
    if angular is not None:
        client.resetBaseVelocity(int(body), angularVelocity=angular)


class PoseSaver(Saver):
    """物体の姿勢（位置・速度）を保存して後から復元するセーバー。"""

    def __init__(self, body, pose=None, client=None):
        self.client = client
        self.body = body
        if pose is None:
            pose = get_pose(self.body, client=client)
        self.pose = pose
        self.velocity = get_velocity(self.body, client=client)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_pose(self.body, self.pose, client=self.client)
        set_velocity(self.body, *self.velocity, client=self.client)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class ConfSaver(Saver):
    """物体の関節角度と速度を保存して後から復元するセーバー。"""

    def __init__(self, body, joints=None, positions=None, client=None, **kwargs):
        self.body = body
        self.client = client
        if joints is None:
            joints = get_movable_joints(self.body, client=self.client)
        self.joints = joints
        if positions is None:
            positions = get_joint_positions(self.body, self.joints, client=self.client)
        self.positions = positions
        self.velocities = get_joint_velocities(
            self.body, self.joints, client=self.client
        )

    @property
    def conf(self):
        return self.positions

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_joint_states(
            self.body, self.joints, self.positions, self.velocities, client=self.client
        )

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class BodySaver(Saver):
    """物体の姿勢と関節状態を一括して保存・復元するセーバー。"""

    def __init__(self, body, client=None, **kwargs):
        self.body = body
        self.client = client
        self.pose_saver = PoseSaver(body, client=client)
        self.conf_saver = ConfSaver(body, client=client, **kwargs)
        self.savers = [self.pose_saver, self.conf_saver]

    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


def plural(word):
    """英単語の複数形を返す（例外辞書付き）。"""
    exceptions = {"radius": "radii"}
    if word in exceptions:
        return exceptions[word]
    if word.endswith("s"):
        return word
    return word + "s"


def get_default_geometry():
    """デフォルト値を持つジオメトリパラメータ辞書を返す。"""
    return {
        "halfExtents": DEFAULT_EXTENTS,
        "radius": DEFAULT_RADIUS,
        "length": DEFAULT_HEIGHT,  # 'height'
        "fileName": DEFAULT_MESH,
        "meshScale": DEFAULT_SCALE,
        "planeNormal": DEFAULT_NORMAL,
    }


def create_shape_array(geoms, poses, colors=None, client=None, **kwargs):
    """複数のジオメトリを配列としてまとめて衝突・視覚形状を作成する。"""
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    # createCollisionShape: height
    # createVisualShape: length
    # createCollisionShapeArray: lengths
    # createVisualShapeArray: lengths
    client = client or DEFAULT_CLIENT
    mega_geom = defaultdict(list)
    for geom in geoms:
        extended_geom = get_default_geometry()
        extended_geom.update(geom)
        # extended_geom = geom.copy()
        for key, value in extended_geom.items():
            mega_geom[plural(key)].append(value)

    collision_args = mega_geom.copy()
    for point, quat in poses:
        collision_args["collisionFramePositions"].append(point)
        collision_args["collisionFrameOrientations"].append(quat)
    collision_id = client.createCollisionShapeArray(**collision_args)
    if colors is None:  # or not has_gui():
        return collision_id, NULL_ID

    visual_args = mega_geom.copy()
    for (point, quat), color in zip(poses, colors):
        # TODO: color doesn't seem to work correctly here
        visual_args["rgbaColors"].append(list(color))
        visual_args["visualFramePositions"].append(point)
        visual_args["visualFrameOrientations"].append(quat)
    visual_id = client.createVisualShapeArray(**visual_args)
    return collision_id, visual_id


def get_aabb_edges(aabb: AABB):
    """AABB の全辺を (始点, 終点) ペアのリストとして返す。"""
    aabb_elements = [aabb.lower, aabb.upper]
    d = len(aabb.lower)
    vertices = list(itertools.product(range(len(aabb_elements)), repeat=d))
    lines = []
    for i1, i2 in itertools.combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb_elements[i1[k]][k] for k in range(d)]
            p2 = [aabb_elements[i2[k]][k] for k in range(d)]
            lines.append((p1, p2))
    return lines


def draw_oobb(oobb: OOBB, origin=False, **kwargs):
    """OOBB の辺を PyBullet デバッグラインで描画してハンドルリストを返す。"""
    handles = []

    if origin:
        handles.extend(draw_pose(oobb.pose, **kwargs))
    for edge in get_aabb_edges(oobb.aabb):
        p1, p2 = tform_points(oobb.pose, edge)
        handles.append(add_line(p1, p2, **kwargs))
    return handles


def draw_pose(pose, length=0.1, d=3, **kwargs):
    """姿勢の座標軸を RGB の矢印で描画してハンドルリストを返す。"""
    origin_world = tform_point(pose, np.zeros(3))
    handles = []
    for k in range(d):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = tform_point(pose, length * axis)
        handles.append(add_line(origin_world, axis_world, color=axis, **kwargs))
    return handles


def aabb_from_points(points):
    """点群を包含する最小の AABB を返す。"""
    return AABB(np.min(points, axis=0), np.max(points, axis=0))


class WorldSaver(Saver):
    """シミュレーション内の全物体（または指定物体）の状態を保存して後から復元するセーバー。"""

    def __init__(self, bodies=None, client=None, **kwargs):
        if bodies is None:
            bodies = get_bodies(client=client, **kwargs)
        self.bodies = bodies
        self.client = client
        self.body_savers = [BodySaver(body, client=client) for body in self.bodies]

    def restore(self, **kwargs):
        for body_saver in self.body_savers:
            body_saver.restore(**kwargs)


def body_from_end_effector(end_effector_pose, grasp_pose):
    """world_from_parent * parent_from_child = world_from_child."""
    return multiply(end_effector_pose, grasp_pose)


class Attachment(object):
    """親リンクに固定されたグラスプ姿勢で子物体をアタッチする関係を表すクラス。"""

    def __init__(self, parent, parent_link, grasp_pose, child):
        self.parent = parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(
            self.parent, get_link_subtree(self.parent, self.parent_link)
        )

    def assign(self, **kwargs):
        parent_link_pose = get_link_pose(self.parent, self.parent_link, **kwargs)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose, **kwargs)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.parent, self.child)


def pairwise_collisions(body, obstacles, link=None, **kwargs):
    """物体が障害物リスト内のいずれかと衝突しているかを返す。"""
    return any(
        pairwise_collision(body1=body, body2=other, link1=link, **kwargs)
        for other in obstacles
        if body != other
    )


DEFAULT_RESOLUTION = math.radians(3)  # 0.05


def get_default_resolution(body, joint, **kwargs):
    """指定した関節タイプに応じたデフォルトの解像度（ステップサイズ）を返す。"""
    joint_type = get_joint_type(body, joint, **kwargs)
    if joint_type == p.JOINT_REVOLUTE:
        return math.radians(3)  # 0.05
    elif joint_type == p.JOINT_PRISMATIC:
        return 0.02
    return DEFAULT_RESOLUTION


def wrap_interval(value, interval: Interval = UNIT_LIMITS, **kwargs):
    """値を指定区間にラップ（剰余処理）して返す。"""
    if (interval.lower == -np.inf) and (np.inf == interval.upper):
        return value
    assert -np.inf < interval.lower <= interval.upper < np.inf
    return (value - interval.lower) % (interval.upper - interval.lower) + interval.lower


def interval_difference(value2, value1, interval: Interval = UNIT_LIMITS):
    """ラップされた区間内での value2 - value1 の最短差分を返す。"""
    value2 = wrap_interval(value2, interval)
    value1 = wrap_interval(value1, interval)
    straight_distance = value2 - value1
    if value2 >= value1:
        wrap_difference = (interval.lower - value1) + (value2 - interval.upper)
    else:
        wrap_difference = (interval.upper - value1) + (value2 - interval.lower)
    # return [straight_distance, wrap_difference]
    if abs(wrap_difference) < abs(straight_distance):
        return wrap_difference
    return straight_distance


def interval_distance(value1, value2, **kwargs):
    """ラップされた区間内での 2 値間の距離（絶対値）を返す。"""
    return abs(interval_difference(value2, value1, **kwargs))


def circular_interval(lower=-np.pi):  # [-np.pi, np.pi)
    """指定した下限から 2π 幅の円形区間を返す。"""
    return Interval(lower, lower + 2 * np.pi)


def wrap_angle(theta, **kwargs):
    """角度を [-π, π) にラップして返す。"""
    return wrap_interval(theta, interval=circular_interval())


def circular_difference(theta2, theta1, **kwargs):
    """円形区間内での theta2 - theta1 の最短差分を返す。"""
    interval = circular_interval()
    extent = get_aabb_extent(interval)
    diff_interval = Interval(-extent / 2, +extent / 2)
    difference = wrap_interval(theta2 - theta1, interval=diff_interval)
    return difference


def get_difference_fn(body, joints, **kwargs):
    """関節ごとの差分計算クロージャを返す（円形関節は最短差分を使用）。"""
    circular_joints = [is_circular(body, joint, **kwargs) for joint in joints]

    def fn(q2, q1):
        return tuple(
            circular_difference(value2, value1) if circular else (value2 - value1)
            for circular, value2, value1 in zip(circular_joints, q2, q1)
        )

    return fn


def wrap_position(body, joint, position, **kwargs):
    """円形関節の場合は角度を [-π, π) にラップして返す。"""
    if is_circular(body, joint, **kwargs):
        return wrap_angle(position, **kwargs)
    return position


def wrap_positions(body, joints, positions, **kwargs):
    """各関節の角度を可動域に合わせてラップしたリストを返す。"""
    assert len(joints) == len(positions)
    return [
        wrap_position(body, joint, position, **kwargs)
        for joint, position in zip(joints, positions)
    ]


def get_refine_fn(body, joints, num_steps=0, **kwargs):
    """2 コンフィギュレーション間を num_steps 段階で補間するクロージャを返す。"""
    difference_fn = get_difference_fn(body, joints, **kwargs)
    num_steps = num_steps + 1

    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1.0 / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(wrap_positions(body, joints, positions, **kwargs))
            yield q

    return fn


def get_default_resolutions(body, joints, resolutions=None, **kwargs):
    """resolutions が None の場合は各関節のデフォルト解像度配列を返す。"""
    if resolutions is not None:
        return resolutions
    return np.array([get_default_resolution(body, joint, **kwargs) for joint in joints])


def get_extend_fn(body, joints, resolutions=None, norm=2, **kwargs):
    """2 コンフィギュレーション間を解像度に基づいて補間するクロージャを返す。"""
    # norm = 1, 2, INF
    resolutions = get_default_resolutions(body, joints, resolutions, **kwargs)
    difference_fn = get_difference_fn(body, joints, **kwargs)

    def fn(q1, q2):
        # steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(
            np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm)
        )
        refine_fn = get_refine_fn(body, joints, num_steps=steps, **kwargs)
        return refine_fn(q1, q2)

    return fn


def interpolate_joint_waypoints(
    body,
    joints,
    waypoints,
    resolutions=None,
    collision_fn=lambda *args, **kwargs: False,
    **kwargs,
):
    """ウェイポイント間を補間してパスを生成し、衝突があれば None を返す。"""
    # TODO: unify with refine_path
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions, **kwargs)
    path = waypoints[:1]
    for waypoint in waypoints[1:]:
        assert len(joints) == len(waypoint)
        for q in list(extend_fn(path[-1], waypoint)):
            if collision_fn(q):
                return None
            path.append(q)  # TODO: could instead yield
    return path


def get_lifetime(lifetime):
    """lifetime が None の場合は 0（永続）を返す。"""
    if lifetime is None:
        return 0
    return lifetime


def add_text(
    text,
    position=unit_point(),
    color=BLACK,
    lifetime=None,
    parent=NULL_ID,
    parent_link=BASE_LINK,
    client=None,
    **kwargs,
):
    """指定位置にデバッグテキストを表示してハンドル ID を返す。"""
    client = client or DEFAULT_CLIENT
    return client.addUserDebugText(
        str(text),
        textPosition=position,
        textColorRGB=list(color)[:3],  # textSize=1,
        lifeTime=get_lifetime(lifetime),
        parentObjectUniqueId=parent,
        parentLinkIndex=parent_link,
    )


def add_line(
    start,
    end,
    color: RGBA = BLACK,
    width=1,
    lifetime=None,
    parent=NULL_ID,
    parent_link=BASE_LINK,
    client=None,
    **kwargs,
):
    """2 点間にデバッグラインを描画してハンドル ID を返す。"""
    client = client or DEFAULT_CLIENT
    assert (len(start) == 3) and (len(end) == 3)
    # time.sleep(1e-3) # When too many lines are added within a short period of time, the following error can occur
    return client.addUserDebugLine(
        start,
        end,
        lineColorRGB=list(color)[:3],
        lineWidth=width,
        lifeTime=get_lifetime(lifetime),
        parentObjectUniqueId=parent,
        parentLinkIndex=parent_link,
    )


def draw_point(point, size=0.01, **kwargs):
    """点を各軸方向の短い線セグメントで描画してハンドルリストを返す。"""
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size / 2 * axis
        p2 = np.array(point) + size / 2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines


def draw_collision_info(collision_info, **kwargs):
    """衝突情報の接触点を結ぶ線と点を描画してハンドルリストを返す。"""
    point1 = collision_info.positionOnA
    point2 = collision_info.positionOnB
    handles = [add_line(point1, point2, **kwargs)]
    for point in [point1, point2]:
        handles.extend(draw_point(point, **kwargs))
    return handles


class State(object):
    """アタッチメントの集合でシミュレーション状態を表すクラス。"""

    def __init__(self, attachments={}):
        self.attachments = dict(attachments)

    def propagate(self, **kwargs):
        for relative_pose in self.attachments.values():
            relative_pose.assign(**kwargs)

    def copy(self):  # update
        return self.__class__(attachments=self.attachments)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, sorted(self.attachments))


def get_top_and_bottom_grasps(
    body,
    body_aabb,
    body_pose,
    tool_pose=unit_pose(),
    under=False,
    max_width=MAX_GRASP_WIDTH,
    grasp_length=GRASP_LENGTH,
    **kwargs,
):
    """物体の上面・下面から把持するための候補姿勢リストを生成する。

    物体の姿勢から重力方向に最も近い軸を特定し、その軸に垂直な面から
    グリッパーが接近できる把持候補を生成する。幅（w）と奥行き（l）の
    それぞれから候補を生成するため、最大 2*(1+under) 個の候補を返す。

    Args:
        body: PyBullet の物体 ID（現在は使用していないが将来の拡張のため残している）。
        body_aabb: 物体のローカル AABB（寸法の取得に使用）。
        body_pose: 物体のワールド姿勢（どの軸が上を向いているかの判定に使用）。
        tool_pose: ツール先端フレームへの補正姿勢（デフォルト: 単位姿勢）。
        under: True の場合、下面からの把持候補も含める。
        max_width: グリッパーの最大開口幅。これを超える寸法の軸は除外される。
        grasp_length: グリッパーが物体に差し込む深さ。

    Returns:
        把持候補姿勢のリスト（ツール先端座標系）。優先度の高い候補が末尾になるよう reversed で返す。
    """
    # TODO: rename the box grasps
    # from IPython import embed; embed()
    rotation_matrix = R.from_quat(list(body_pose[1]))

    rotation_matrix = rotation_matrix.as_matrix()
    best_axis = np.argmax(np.abs(rotation_matrix[2, :3]))
    direction = np.sign(rotation_matrix[2, best_axis])

    dims = np.array([*get_aabb_extent(body_aabb)])
    w, l, h = dims
    mask = np.zeros(3)
    mask[best_axis] = 1.0

    distance = dims[best_axis] / 2.0 - grasp_length
    translate_z = Pose(point=np.array([0.0, 0.0, distance]))

    if best_axis == 2:
        if direction > 0.0:
            val = np.pi
        else:
            val = 0.0
        reflect_z = Pose(euler=[val, 0, 0])
    elif best_axis == 1:
        if direction > 0.0:
            val = -np.pi / 2.0
        else:
            val = np.pi / 2.0
        reflect_z = Pose(euler=[val, 0, 0])
    else:
        if direction > 0.0:
            val = np.pi / 2.0
        else:
            val = -np.pi / 2.0
        reflect_z = Pose(euler=[0, val, 0])

    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [
                multiply(
                    tool_pose,
                    translate_z,
                    rotate_z,
                    reflect_z,
                )
            ]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z, reflect_z)]

    return list(reversed(grasps))


def empty_sequence():
    """空のイテレータを返す。"""
    return iter([])


def get_mass(body, link=BASE_LINK, **kwargs):  # mass in kg
    """指定リンク（デフォルトはベース）の質量（kg）を返す。"""
    # TODO: get full mass
    return get_dynamics_info(body, link, **kwargs).mass


def convex_combination(x, y, w=0.5):
    """x と y の重み w による凸結合 (1-w)*x + w*y を返す。"""
    return (1 - w) * np.array(x) + w * np.array(y)


def uniform_generator(d):
    """d 次元の一様乱数ベクトルを無限に生成するジェネレータを返す。"""
    while True:
        yield np.random.uniform(size=d)


def unit_generator(d, **kwargs):
    """d 次元の単位超立方体 [0,1]^d から一様乱数を無限に生成するジェネレータを返す。"""
    return uniform_generator(d)


def interval_generator(lower, upper, **kwargs):
    """lower と upper の範囲から一様乱数サンプルを無限に生成するジェネレータを返す。"""
    assert len(lower) == len(upper)
    assert np.less_equal(lower, upper).all()
    if np.equal(lower, upper).all():
        return iter([lower])
    return (
        convex_combination(lower, upper, w=weights)
        for weights in unit_generator(d=len(lower), **kwargs)
    )


def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    """関節の可動域内からランダムコンフィギュレーションを返すクロージャを生成する。"""
    lower_limits, upper_limits = get_custom_limits(
        body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS, **kwargs
    )
    generator = interval_generator(lower_limits, upper_limits, **kwargs)

    def fn():
        return tuple(next(generator))

    return fn


def unit_vector(data, axis=None, out=None):
    """ベクトルまたは行列の指定軸に沿って正規化した単位ベクトルを返す。"""
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """2 つのクォータニオンを球面線形補間（SLERP）した結果を返す。"""
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quat_combination(quat1, quat2, fraction=0.5):
    """SLERP で 2 つのクォータニオンを補間した結果を返す。"""
    # return p.getQuaternionSlerp(quat1, quat2, interpolationFraction=fraction)
    return quaternion_slerp(quat1, quat2, fraction)


def clip(value, min_value=-np.inf, max_value=+np.inf):
    """値を [min_value, max_value] の範囲にクランプして返す。"""
    return min(max(min_value, value), max_value)


def quat_angle_between(quat0, quat1):
    """2 つのクォータニオン間の回転角（ラジアン）を返す。"""
    delta = p.getDifferenceQuaternion(quat0, quat1)
    d = clip(delta[-1], min_value=-1.0, max_value=1.0)
    angle = math.acos(d)
    return angle


def get_pose_distance(pose1, pose2):
    """2 つの姿勢間の位置距離と角度距離を (pos_distance, ori_distance) で返す。"""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    pos_distance = get_distance(pos1, pos2)
    ori_distance = quat_angle_between(quat1, quat2)
    return pos_distance, ori_distance


def interpolate_poses(pose1, pose2, pos_step_size=0.01, ori_step_size=np.pi / 16):
    """2 姿勢間を位置・姿勢ステップサイズに基づいて補間した姿勢を yield する。"""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    num_steps = max(
        2,
        int(
            math.ceil(
                max(
                    np.divide(
                        get_pose_distance(pose1, pose2), [pos_step_size, ori_step_size]
                    )
                )
            )
        ),
    )
    yield pose1
    for w in np.linspace(0, 1, num=num_steps, endpoint=True)[1:-1]:
        pos = convex_combination(pos1, pos2, w=w)
        quat = quat_combination(quat1, quat2, fraction=w)
        yield (pos, quat)
    yield pose2


def unit_from_theta(theta):
    """角度 theta から 2D 単位ベクトルを返す。"""
    return np.array([np.cos(theta), np.sin(theta)])


def sample_reachable_base(robot, point, reachable_range=(0.25, 1.0)):
    """指定点周辺の到達可能範囲内でロボットのベース姿勢をサンプリングする。"""
    radius = np.random.uniform(*reachable_range)
    x, y = radius * unit_from_theta(np.random.uniform(-np.pi, np.pi)) + point[:2]
    yaw = np.random.uniform(CIRCULAR_LIMITS.lower, CIRCULAR_LIMITS.upper)
    base_values = (x, y, yaw)
    return base_values


def uniform_pose_generator(robot, gripper_pose, **kwargs):
    """グリッパー姿勢に到達可能なベース姿勢を無限に生成するジェネレータを返す。"""
    point = point_from_pose(gripper_pose)
    while True:
        base_values = sample_reachable_base(robot, point, **kwargs)
        if base_values is None:
            break
        yield base_values


def custom_limits_from_base_limits(robot, base_limits, yaw_limit=None, **kwargs):
    """ベース可動域から x, y（, yaw）関節のカスタム制限辞書を生成する。"""
    x_limits, y_limits = zip(*base_limits)
    custom_limits = {
        joint_from_name(robot, "x", **kwargs): x_limits,
        joint_from_name(robot, "y", **kwargs): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update(
            {
                joint_from_name(robot, "theta", **kwargs): yaw_limit,
            }
        )
    return custom_limits


def remove_alpha(color: RGBA) -> RGB:
    """RGBA カラーからアルファ値を除いた RGB カラーを返す。"""
    return RGB(color.red, color.green, color.blue)


def tform_oobb(affine, oobb: OOBB) -> OOBB:
    """アフィン姿勢を OOBB に適用して変換後の OOBB を返す。"""
    return OOBB(oobb.aabb, multiply(affine, oobb.pose))


def aabb2d_from_aabb(aabb: AABB) -> AABB:
    """3D AABB から XY 平面上の 2D AABB を返す。"""
    return AABB(aabb.lower[:2], aabb.upper[:2])


def convex_centroid(vertices):
    """凸多角形頂点リストから 2D 重心座標を返す。"""
    vertices = [np.array(v[:2]) for v in vertices]
    segments = get_wrapped_pairs(vertices)
    return sum((v1 + v2) * np.cross(v1, v2) for v1, v2 in segments) / (
        6.0 * convex_signed_area(vertices)
    )


def aabb_empty(aabb: AABB) -> bool:
    """AABB が空（upper < lower の軸が 1 つ以上ある）かどうかを返す。"""
    return np.less(aabb.upper, aabb.lower).any()


def sample_aabb(aabb: AABB):
    """AABB 内から一様乱数で 1 点をサンプリングして返す。"""
    return np.random.uniform(aabb.lower, aabb.upper)


def convex_area(vertices):
    """凸多角形の面積（正値）を返す。"""
    return abs(convex_signed_area(vertices))


def get_wrapped_pairs(sequence):
    """シーケンスから循環する隣接ペアのリストを返す。"""
    sequence = list(sequence)
    return safe_zip(sequence, sequence[1:] + sequence[:1])


def convex_signed_area(vertices):
    """凸多角形の符号付き面積をショールース公式で返す。"""
    if len(vertices) < 3:
        return 0.0
    vertices = [np.array(v[:2]) for v in vertices]
    segments = get_wrapped_pairs(vertices)
    return sum(np.cross(v1, v2) for v1, v2 in segments) / 2.0


def sample_placement_on_aabb(
    top_body,
    bottom_aabb: AABB,
    top_pose=unit_pose(),
    percent=1.0,
    max_attempts=50,
    epsilon=1e-3,
    **kwargs,
):
    """物体を AABB 上の安定した場所にランダム配置する姿勢をサンプリングする。"""
    for _ in range(max_attempts):
        theta = np.random.uniform(CIRCULAR_LIMITS.lower, CIRCULAR_LIMITS.upper)
        rotation = Euler(yaw=theta)
        set_pose(top_body, multiply(Pose(euler=rotation), top_pose), **kwargs)
        center, extent = get_center_extent(top_body, **kwargs)
        lower = (np.array(bottom_aabb.lower) + percent * extent / 2)[:2]
        upper = (np.array(bottom_aabb.upper) - percent * extent / 2)[:2]
        aabb = AABB(lower, upper)
        if aabb_empty(aabb):
            continue
        x, y = sample_aabb(aabb)
        z = (bottom_aabb.upper + extent / 2.0)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(top_body, **kwargs) - center)
        pose = multiply(Pose(point, rotation), top_pose)
        set_pose(top_body, pose, **kwargs)
        return pose
    return None


def all_between(lower_limits, values, upper_limits):
    """すべての値が対応する下限と上限の範囲内にあるかを返す。"""
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return (
        np.less_equal(lower_limits, values).all()
        and np.less_equal(values, upper_limits).all()
    )


def inverse_kinematics_helper(
    robot,
    link,
    target_pose,
    client=None,
    lower_limits=None,
    upper_limits=None,
    **kwargs,
):
    """PyBullet の IK ソルバーを 1 回呼び出して全関節の解を返す（失敗時は None）。"""
    (target_point, target_quat) = target_pose
    assert target_point is not None

    if target_quat is None:
        kinematic_conf = client.calculateInverseKinematics(
            int(robot), link, target_point, maxNumIterations=1000
        )
    else:
        kinematic_conf = client.calculateInverseKinematics(
            int(robot), link, target_point, target_quat, maxNumIterations=1000
        )
    if (kinematic_conf is None) or any(map(math.isnan, kinematic_conf)):
        return None
    return kinematic_conf


def is_point_close(point1, point2, tolerance=1e-3):
    """2 点が許容誤差内で一致しているかどうかを返す。"""
    return all_close(point1, point2, atol=tolerance)


def all_close(a, b, atol=1e-6, rtol=0.0):
    """2 つの配列が許容誤差内で一致しているかどうかを返す。"""
    assert len(a) == len(b)  # TODO: shape
    return np.allclose(a, b, atol=atol, rtol=rtol)


def is_quat_close(quat1, quat2, tolerance=1e-3 * np.pi):
    """2 つのクォータニオンが符号の違いを考慮して許容誤差内で一致するかを返す。"""
    return any(
        all_close(quat1, sign * np.array(quat2), atol=tolerance) for sign in [-1.0, +1]
    )


def is_pose_close(pose, target_pose, pos_tolerance=1e-3, ori_tolerance=1e-3 * np.pi):
    """姿勢が目標姿勢に位置・向きともに許容誤差内で一致しているかを返す。"""
    (point, quat) = pose
    (target_point, target_quat) = target_pose
    if (target_point is not None) and not is_point_close(
        point, target_point, tolerance=pos_tolerance
    ):
        return False
    if (target_quat is not None) and not is_quat_close(
        quat, target_quat, tolerance=ori_tolerance
    ):
        return False
    return True


def inverse_kinematics(
    robot,
    link,
    target_pose,
    joints,
    max_iterations=20,
    max_time=np.inf,
    custom_limits={},
    **kwargs,
):
    """指定リンクを target_pose に収める逆運動学（IK）を解く。

    PyBullet の IK ソルバーを反復的に呼び出し、収束または max_iterations に
    達するまで試みる。

    Args:
        robot: PyBullet のロボット ID。
        link: IK のターゲットとなるリンクインデックス（ツール先端など）。
        target_pose: ターゲット姿勢 (position, quaternion)。
        joints: IK 解を返す対象の関節リスト（全可動関節のサブセット）。
        max_iterations: IK 反復の上限回数。
        max_time: IK 計算の制限時間（秒）。
        custom_limits: 関節ごとのカスタム可動範囲 {joint: (lower, upper)}。

    Returns:
        joints に対応する関節角度のリスト。収束しない・範囲外の場合は None。

    Note:
        IK 計算中にロボットの関節角度を一時的に変更する副作用がある。
        呼び出し元は必要に応じて元の状態を保存・復元すること。
    """
    start_time = time.time()
    movable_joints = get_movable_joints(robot, **kwargs)

    for _ in range(max_iterations):
        if elapsed_time(start_time) >= max_time:
            return None
        kinematic_conf = inverse_kinematics_helper(robot, link, target_pose, **kwargs)
        if kinematic_conf is None:
            return None
        set_joint_positions(robot, movable_joints, kinematic_conf, **kwargs)
        if is_pose_close(get_link_pose(robot, link, **kwargs), target_pose):
            break
    else:
        return None

    conf = [
        q
        for q, j in zip(kinematic_conf, get_movable_joints(robot, **kwargs))
        if j in joints
    ]

    lower_limits, upper_limits = get_custom_limits(
        robot, joints, custom_limits, **kwargs
    )
    if not all_between(lower_limits, conf, upper_limits):
        return None

    return conf


def recenter_oobb(oobb: OOBB):
    """OOBB を AABB の中心が原点に来るように再中心化した OOBB を返す。"""
    extent = get_aabb_extent(oobb.aabb)
    new_aabb = AABB(-extent / 2.0, +extent / 2.0)
    return OOBB(new_aabb, multiply(oobb.pose, Pose(point=get_aabb_center(oobb.aabb))))


def scale_aabb(aabb, scale):
    """AABB を中心を保ちながら指定スケールで拡縮した新しい AABB を返す。"""
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    if np.isscalar(scale):
        scale = scale * np.ones(len(extent))
    new_extent = np.multiply(scale, extent)
    return aabb_from_extent_center(new_extent, center)


def get_limits_fn(body, joints, custom_limits={}, verbose=False, **kwargs):
    """コンフィギュレーションが関節限界を超えているかを返すクロージャを生成する。"""
    lower_limits, upper_limits = get_custom_limits(
        body, joints, custom_limits, **kwargs
    )

    def limits_fn(q):
        if not all_between(lower_limits, q, upper_limits):
            return True
        return False

    return limits_fn


def get_link_ancestors(body, link, **kwargs):
    """指定リンクのすべての祖先リンクインデックスをルートから順に返す。"""
    parent = get_link_parent(body, link, **kwargs)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, **kwargs) + [parent]


def get_joint_ancestors(body, joint, **kwargs):
    """指定関節の祖先リンク（自身含む）のインデックスリストを返す。"""
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link, **kwargs) + [link]


def get_moving_pairs(body, moving_joints, **kwargs):
    """Check all fixed and moving pairs Do not check all fixed and fixed pairs
    Check all moving pairs with a common."""
    moving_links = list(
        filter(
            lambda link: can_collide(body, link, **kwargs),
            get_moving_links(body, moving_joints, **kwargs),
        )
    )
    for link1, link2 in itertools.combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1, **kwargs)) & set(
            moving_joints
        )
        ancestors2 = set(get_joint_ancestors(body, link2, **kwargs)) & set(
            moving_joints
        )
        if ancestors1 != ancestors2:
            yield link1, link2


def are_links_adjacent(body, link1, link2, **kwargs):
    """2 つのリンクが親子関係にあるかどうかを返す。"""
    return (get_link_parent(body, link1, **kwargs) == link2) or (
        get_link_parent(body, link2, **kwargs) == link1
    )


def get_self_link_pairs(
    body, joints, disabled_collisions=set(), only_moving=True, **kwargs
):
    """自己衝突チェック対象のリンクペアリストを返す。"""
    moving_links = list(
        filter(
            lambda link: can_collide(body, link, **kwargs),
            get_moving_links(body, joints, **kwargs),
        )
    )
    fixed_links = list(
        filter(
            lambda link: can_collide(body, link, **kwargs),
            set(get_links(body, **kwargs)) - set(moving_links),
        )
    )
    check_link_pairs = list(itertools.product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints, **kwargs))
    else:
        check_link_pairs.extend(itertools.combinations(moving_links, 2))
    check_link_pairs = list(
        filter(
            lambda pair: not are_links_adjacent(body, *pair, **kwargs), check_link_pairs
        )
    )
    check_link_pairs = list(
        filter(
            lambda pair: (pair not in disabled_collisions)
            and (pair[::-1] not in disabled_collisions),
            check_link_pairs,
        )
    )
    return check_link_pairs


def cached_fn(fn, cache=True, **global_kargs):
    """関数をキャッシュ付きでラップした高速クロージャを返す。"""
    def normal(*args, **local_kwargs):
        kwargs = dict(global_kargs)
        kwargs.update(local_kwargs)
        return fn(*args, **kwargs)

    if not cache:
        return normal

    try:
        from functools import lru_cache as cache

        @cache(maxsize=None, typed=False)
        def wrapped(*args, **local_kwargs):
            return normal(*args, **local_kwargs)

        return wrapped
    except ImportError:
        pass

    key_fn = id
    cache = {}

    def wrapped(*args, **local_kwargs):
        args_key = tuple(map(key_fn, args))
        local_kwargs_key = frozenset(
            {key: key_fn(value) for key, value in local_kwargs.items()}.items()
        )
        key = (args_key, local_kwargs_key)
        if key not in cache:
            cache[key] = normal(*args, **local_kwargs)
        return cache[key]

    return wrapped


def get_collision_fn(
    body,
    joints,
    obstacles=[],
    attachments=[],
    self_collisions=True,
    disabled_collisions=set(),
    custom_limits={},
    use_aabb=False,
    cache=False,
    max_distance=MAX_DISTANCE,
    extra_collisions=None,
    **kwargs,
):
    """関節角度を受け取り衝突の有無を返すクロージャを生成する。

    生成されるクロージャ collision_fn(q) は、関節角度 q に対して
    衝突（自己衝突・障害物衝突・関節限界超過）が発生するかどうかを bool で返す。

    Args:
        body: 衝突判定対象のロボット ID。
        joints: 衝突判定に使用する関節リスト。
        obstacles: 障害物の物体 ID リスト。
        attachments: グリッパーで把持中の物体などのアタッチメントリスト。
        self_collisions: 自己衝突を判定に含めるかどうか。
        disabled_collisions: 無視する自己衝突ペアの集合 {(link1, link2), ...}。
        custom_limits: 関節ごとのカスタム可動範囲 {joint: (lower, upper)}。
        use_aabb: AABB で事前フィルタリングを行うかどうか。
        max_distance: この距離以内に接近した場合を衝突とみなす閾値（メートル）。

    Returns:
        collision_fn(q, verbose=False) -> bool：
        q が衝突状態なら True を返すクロージャ。
        内部で set_joint_positions を呼ぶためロボット状態を変更する副作用がある。
    """
    check_link_pairs = (
        get_self_link_pairs(body, joints, disabled_collisions, **kwargs)
        if self_collisions
        else []
    )
    moving_links = frozenset(
        link
        for link in get_moving_links(body, joints, **kwargs)
        if can_collide(body, link, **kwargs)
    )
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(
        map(parse_body, attached_bodies)
    )
    get_obstacle_aabb = cached_fn(
        get_buffered_aabb, cache=cache, max_distance=max_distance / 2.0, **kwargs
    )
    limits_fn = get_limits_fn(body, joints, custom_limits=custom_limits, **kwargs)

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True

        set_joint_positions(body, joints, q, **kwargs)

        for attachment in attachments:
            attachment.assign(**kwargs)

        if extra_collisions is not None and extra_collisions(**kwargs):
            return True

        get_moving_aabb = cached_fn(
            get_buffered_aabb, cache=True, max_distance=max_distance / 2.0, **kwargs
        )

        for link1, link2 in check_link_pairs:
            if (
                not use_aabb
                or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))
            ) and pairwise_link_collision(body, link1, body, link2, **kwargs):
                return True

        for body1, body2 in itertools.product(moving_bodies, obstacles):
            if (
                not use_aabb
                or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pairwise_collision(body1, body2, **kwargs):
                return True
        return False

    return collision_fn


def get_default_weights(body, joints, weights=None):
    """weights が None の場合は等重みの配列を返す。"""
    if weights is not None:
        return weights
    return 1 * np.ones(len(joints))


def get_distance_fn(body, joints, weights=None, norm=2, **kwargs):
    """重み付きコンフィギュレーション距離を計算するクロージャを返す。"""
    weights = get_default_weights(body, joints, weights)
    difference_fn = get_difference_fn(body, joints, **kwargs)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        if norm == 2:
            return np.sqrt(np.dot(weights, diff * diff))
        return np.linalg.norm(np.multiply(weights, diff), ord=norm)

    return fn


def check_initial_end(
    body, joints, start_conf, end_conf, collision_fn, verbose=True, **kwargs
):
    """開始・終了コンフィギュレーションが衝突していないかを確認する。"""
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        set_joint_positions(body, joints, start_conf, **kwargs)
        print("Warning: initial configuration is in collision")
        wait_if_gui(**kwargs)
        return False
    if collision_fn(end_conf, verbose=verbose):
        set_joint_positions(body, joints, end_conf, **kwargs)
        print("Warning: end configuration is in collision")
        wait_if_gui(**kwargs)
        return False
    return True


def euler_from_quat(quat):
    """クォータニオンからオイラー角（ロール・ピッチ・ヨー）を返す。"""
    return p.getEulerFromQuaternion(quat)  # rotation around fixed axis


def single_collision(body, **kwargs):
    """物体がシミュレーション内の他の物体と衝突しているかを返す。"""
    return pairwise_collisions(body, get_bodies(), **kwargs)


def remove_handles(handles, **kwargs):
    """デバッグハンドルのリストをすべて削除する。"""
    for handle in handles:
        remove_debug(handle, **kwargs)
    handles[:] = []


def multiply_quats(*quats):
    """複数のクォータニオンを順に合成した結果を返す。"""
    return quat_from_pose(multiply(*[(unit_point(), quat) for quat in quats]))


def get_time_step(client=None, **kwargs):
    """シミュレーションの固定タイムステップ（秒）を返す。"""
    client = client or DEFAULT_CLIENT
    return client.getPhysicsEngineParameters()["fixedTimeStep"]


def get_ordered_ancestors(robot, link, **kwargs):
    """指定リンクの祖先から自身までを順に並べたリンクインデックスリストを返す。"""
    return get_link_ancestors(robot, link, **kwargs)[1:] + [link]


def get_configuration(body, **kwargs):
    """物体の可動関節すべての現在角度をタプルで返す。"""
    return get_joint_positions(body, get_movable_joints(body, **kwargs), **kwargs)


def set_configuration(body, values, **kwargs):
    """物体の可動関節すべての角度を一括設定する。"""
    set_joint_positions(body, get_movable_joints(body, **kwargs), values, **kwargs)


def create_sub_robot(robot, first_joint, target_link):
    """IK 計算用に first_joint 以降のリンクを複製したサブロボットを作成する。"""
    # TODO: create a class or generator for repeated use
    selected_links = get_link_subtree(
        robot, first_joint
    )  # TODO: child_link_from_joint?
    selected_joints = prune_fixed_joints(robot, selected_links)
    assert target_link in selected_links
    sub_target_link = selected_links.index(target_link)
    sub_robot = clone_body(
        robot, links=selected_links, visual=False, collision=False
    )  # TODO: joint limits
    assert len(selected_joints) == len(get_movable_joints(sub_robot))
    return sub_robot, selected_joints, sub_target_link


def multiple_sub_inverse_kinematics(
    robot,
    first_joint,
    target_link,
    target_pose,
    max_attempts=1,
    max_solutions=np.inf,
    max_time=np.inf,
    custom_limits={},
    first_close=True,
    **kwargs,
):
    """サブロボットを用いて複数の IK 解候補を探索して返す。"""
    start_time = time.time()
    ancestor_joints = prune_fixed_joints(
        robot, get_ordered_ancestors(robot, target_link)
    )
    affected_joints = ancestor_joints[ancestor_joints.index(first_joint) :]
    sub_robot, selected_joints, sub_target_link = create_sub_robot(
        robot, first_joint, target_link
    )
    sub_joints = prune_fixed_joints(
        sub_robot, get_ordered_ancestors(sub_robot, sub_target_link)
    )
    selected_joints = affected_joints

    sample_fn = get_sample_fn(robot, selected_joints, custom_limits=custom_limits)
    solutions = []
    for attempt in range(max_attempts):
        if (len(solutions) >= max_solutions) or (elapsed_time(start_time) >= max_time):
            break
        if not first_close or (attempt >= 1):  # TODO: multiple seed confs
            sub_conf = sample_fn()
            set_joint_positions(sub_robot, sub_joints, sub_conf, **kwargs)
        sub_kinematic_conf = inverse_kinematics(
            sub_robot,
            sub_target_link,
            target_pose,
            max_time=max_time - elapsed_time(start_time),
            **kwargs,
        )
        if sub_kinematic_conf is not None:
            sub_kinematic_conf = get_joint_positions(sub_robot, sub_joints, **kwargs)
            set_joint_positions(robot, selected_joints, sub_kinematic_conf, **kwargs)
            kinematic_conf = get_configuration(robot, **kwargs)
            solutions.append(kinematic_conf)
    if solutions:
        set_configuration(robot, solutions[-1])

    remove_body(sub_robot)
    return solutions


def matrix_from_quat(quat):
    """クォータニオンから 3x3 回転行列を返す。"""
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def get_pairs(sequence):
    """シーケンスから隣接要素のペアリストを返す。"""
    sequence = list(sequence)
    return safe_zip(sequence[:-1], sequence[1:])


def adjust_path(robot, joints, path, initial_conf=None, **kwargs):
    """パスを初期コンフィギュレーションから差分で再構築して返す。"""
    if path is None:
        return path
    if initial_conf is None:
        initial_conf = path[0]
    difference_fn = get_difference_fn(robot, joints, **kwargs)
    differences = [difference_fn(q2, q1) for q1, q2 in get_pairs(path)]
    adjusted_path = [np.array(initial_conf)]  # Assumed the same as path[0] mod rotation
    for difference in differences:
        if not np.array_equal(difference, np.zeros(len(joints))):
            adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path


def step_curve(robot, joints, curve, time_step=2e-2, print_freq=None, **kwargs):
    """スプライン曲線を time_step 刻みでロボット関節に追従させ位置を yield する。"""
    start_time = time.time()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    for num_steps, (time_elapsed, positions) in enumerate(
        sample_curve(curve, time_step=time_step)
    ):
        set_joint_positions(robot, joints, positions, **kwargs)

        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    elapsed_time(start_time),
                    num_steps / elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        yield positions
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, elapsed_time(start_time)
            )
        )


class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    """A context manager that block stdout for its scope, usage:

    with HideOutput():     os.system('ls -l')
    """
    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno)  # Added
