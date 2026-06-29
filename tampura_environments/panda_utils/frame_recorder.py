from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import imageio.v3 as iio

import tampura_environments.panda_utils.pb_utils as pbu


class FrameRecorder:
    """カメラフレームをメモリ上に蓄積し、必要に応じて GIF を書き出すクラス。

    Usage::

        recorder = FrameRecorder(capture_fn, save_dir, interval=0.1)

        # env.step() の前後で:
        state.frame_callback = recorder.make_step_callback()
        result = super().step(action, belief, store)
        state.frame_callback = None

        recorder.make_gif()   # 全ステップ終了後に GIF を書き出す
    """

    def __init__(
        self,
        capture_fn: Callable[[], Optional[Any]],
        save_dir: str,
        interval: float = 0.1,
        enabled: bool = True,
    ):
        self._capture_fn = capture_fn
        self._save_dir = Path(save_dir)
        self._interval = interval
        self._enabled = enabled
        self._frames: List[Any] = []

    def _capture(self) -> None:
        """``capture_fn`` を呼び出して1フレームを取得し、メモリ上のリストに追加する。``capture_fn`` が ``None`` を返した場合は何もしない。"""
        rgb = self._capture_fn()
        if rgb is not None:
            self._frames.append(rgb)

    def make_step_callback(self, time_step: float = 5e-3) -> Optional[Callable[[], None]]:
        """``interval`` シミュレーション秒ごとに1フレームを取得するコールバックを返す。

        返り値を実行前に ``SceneState.frame_callback`` へ代入して使用する。
        ``enabled=False`` のときは ``None`` を返すため、呼び出し元は代入をスキップできる。
        """
        if not self._enabled:
            return None
        steps_per_frame = max(1, round(self._interval / time_step))
        counter = [0]

        def callback() -> None:
            counter[0] += 1
            if counter[0] % steps_per_frame == 0:
                self._capture()

        return callback

    def capture_frame(self) -> None:
        """モーションを伴わないイベント向けの単発フレーム取得（同期）。"""
        if self._enabled:
            self._capture()

    def make_gif(self) -> None:
        """メモリ上のフレームをまとめて ``generated.gif`` に書き出す。``enabled=False`` またはフレームが1枚もない場合は何もしない。"""
        if not self._enabled or not self._frames:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            self._save_dir / "generated.gif",
            self._frames,
            duration=0.05,  # seconds per frame → 20 fps
            loop=0,
        )


def make_robot_capture_fn(world_getter: Callable) -> Callable[[], Optional[Any]]:
    """ロボットの搭載カメラからフレームを取得する ``capture_fn`` を返す。"""
    def capture() -> Optional[Any]:
        world = world_getter()
        if world is None:
            return None
        camera_pose = world.robot.camera.get_pose(client=world.client)
        camera_matrix = world.robot.camera.camera_matrix
        img = pbu.get_image_at_pose(
            camera_pose, camera_matrix, tiny=True, client=world.client
        )
        # Flip: PyBullet uses OpenGL origin (bottom-left); NumPy/imageio expects top-left.
        return img.rgbPixels[::-1, :, :3]
    return capture


def make_external_capture_fn(
    world_getter: Callable,
    camera_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    width: int = 640,
    height: int = 480,
    vertical_fov: float = 60.0,
) -> Callable[[], Optional[Any]]:
    """ワールド座標系に固定された外部カメラからフレームを取得する ``capture_fn`` を返す。

    Args:
        world_getter: ``World`` インスタンスを返す callable。``None`` を返した場合はフレームをスキップする。
        camera_pos: カメラ位置 (x, y, z)。
        target_pos: カメラが向く注視点 (x, y, z)。
        width: 画像幅（ピクセル）。
        height: 画像高さ（ピクセル）。
        vertical_fov: 垂直画角（度）。
    """
    def capture() -> Optional[Any]:
        world = world_getter()
        if world is None:
            return None
        img = pbu.get_image(
            camera_pos=camera_pos,
            target_pos=target_pos,
            width=width,
            height=height,
            vertical_fov=vertical_fov,
            tiny=True,
            client=world.client,
        )
        # Flip: PyBullet uses OpenGL origin (bottom-left); NumPy/imageio expects top-left.
        return img.rgbPixels[::-1, :, :3]
    return capture
