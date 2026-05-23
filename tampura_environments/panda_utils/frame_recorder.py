from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import imageio.v3 as iio

import tampura_environments.panda_utils.pb_utils as pbu


class FrameRecorder:
    """Context manager that captures camera frames at a fixed interval in a background thread.

    Usage::

        recorder = FrameRecorder(lambda: self.world, save_dir, interval=0.1)

        with recorder:
            env.step(...)   # frames captured throughout action execution

        recorder.make_gif()
    """

    def __init__(self, world_getter: Callable, save_dir: str, interval: float = 0.1):
        self._world_getter = world_getter
        self._frame_dir = Path(save_dir) / "frames"
        self._interval = interval
        self._count = 0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> FrameRecorder:
        """``with recorder:`` の開始時に呼ばれる。バックグラウンドスレッドを起動する。"""
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        """``with`` ブロックを抜けるときに呼ばれる（例外発生時も含む）。スレッドを停止して合流する。"""
        self._stop_event.set()
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._capture()
            self._stop_event.wait(self._interval)

    def _capture(self) -> None:
        """ロボットのカメラ位置から1フレームを撮影し、連番 PNG として保存する。``world`` が未初期化の場合は何もしない。"""
        world = self._world_getter()
        if world is None:
            return
        self._frame_dir.mkdir(exist_ok=True)
        camera_pose = world.robot.camera.get_pose(client=world.client)
        camera_matrix = world.robot.camera.camera_matrix
        camera_image = pbu.get_image_at_pose(
            camera_pose, camera_matrix, tiny=True, client=world.client
        )
        iio.imwrite(
            self._frame_dir / f"frame_{self._count:05d}.png",
            camera_image.rgbPixels[:, :, :3],
        )
        self._count += 1

    def make_gif(self) -> None:
        """保存済みの PNG フレームをまとめて ``generated.gif`` に変換する。フレームが1枚もない場合は何もしない。"""
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
