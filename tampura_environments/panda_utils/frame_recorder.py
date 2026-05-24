from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import imageio.v3 as iio

import tampura_environments.panda_utils.pb_utils as pbu


class FrameRecorder:
    """Context manager that captures camera frames at a fixed interval in a background thread.

    Usage::

        capture_fn = make_external_capture_fn(lambda: self.world, camera_pos, target_pos)
        recorder = FrameRecorder(capture_fn, save_dir, interval=0.1)

        with recorder:
            env.step(...)   # frames captured throughout action execution

        recorder.make_gif()
    """

    def __init__(
        self,
        capture_fn: Callable[[], Optional[Any]],
        save_dir: str,
        interval: float = 0.1,
        enabled: bool = True,
    ):
        self._capture_fn = capture_fn
        self._frame_dir = Path(save_dir) / "frames"
        self._interval = interval
        self._enabled = enabled
        self._count = 0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> FrameRecorder:
        """``with recorder:`` の開始時に呼ばれる。``enabled=True`` のときのみバックグラウンドスレッドを起動する。"""
        if self._enabled:
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        """``with`` ブロックを抜けるときに呼ばれる（例外発生時も含む）。``enabled=True`` のときのみスレッドを停止して合流する。"""
        if self._enabled:
            self._stop_event.set()
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._capture()
            self._stop_event.wait(self._interval)

    def _capture(self) -> None:
        """``capture_fn`` を呼び出して1フレームを取得し、連番 PNG として保存する。``capture_fn`` が ``None`` を返した場合は何もしない。"""
        rgb = self._capture_fn()
        if rgb is None:
            return
        self._frame_dir.mkdir(exist_ok=True)
        iio.imwrite(
            self._frame_dir / f"frame_{self._count:05d}.png",
            rgb,
        )
        self._count += 1

    def make_gif(self) -> None:
        """保存済みの PNG フレームをまとめて ``generated.gif`` に変換する。``enabled=False`` またはフレームが1枚もない場合は何もしない。"""
        if not self._enabled:
            return
        png_paths = sorted(self._frame_dir.glob("*.png"))
        if not png_paths:
            return
        images = [iio.imread(p) for p in png_paths]
        iio.imwrite(
            self._frame_dir.parent / "generated.gif",
            images,
            duration=0.05,
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
        # Flip vertically: PyBullet returns images in OpenGL convention (origin=bottom-left),
        # but NumPy/imageio expects origin=top-left.
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
        # Flip vertically: PyBullet returns images in OpenGL convention (origin=bottom-left),
        # but NumPy/imageio expects origin=top-left.
        return img.rgbPixels[::-1, :, :3]
    return capture
