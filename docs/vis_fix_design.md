# DESIGN: Visualization Fix (FrameRecorder threading bug)

## 1. SCOPE

This fix applies to:
- `panda_utils/frame_recorder.py` — add synchronous capture method
- `find_dice/env.py` — replace background-thread recording with synchronous capture

After verifying in find_dice, apply the same pattern to any new environment
(e.g. `find_block_stack_dice/env.py`).

---

## 2. ROOT CAUSE

`FrameRecorder` spawns a `threading.Thread` (daemon=True) that calls `capture_fn()`
every 0.1s while `step()` executes:

```python
# find_dice/env.py — current buggy pattern
def step(self, action, belief, store):
    with self._recorder:        # ← starts background thread here
        return super().step(action, belief, store)
```

PyBullet is **NOT thread-safe**. The background thread reads camera data concurrently
with the main thread running physics steps, causing:
- Race conditions on the PyBullet client
- Dropped / corrupted frames
- Occasional hangs on `_thread.join()`

---

## 3. FIX

Replace concurrent background capture with synchronous per-step capture.

### 3.1 Change 1: panda_utils/frame_recorder.py

Add `capture_frame()` method to `FrameRecorder`:

```python
def capture_frame(self) -> None:
    """Synchronous single-frame capture. Call after each physics step. No threading needed."""
    if self._enabled:
        self._capture()
```

Existing `__enter__`, `__exit__`, `_loop` methods are kept as-is (may still be used
by other callers), but are no longer called from env.py.

### 3.2 Change 2: find_dice/env.py

Replace:
```python
def step(self, action, belief, store):
    with self._recorder:
        return super().step(action, belief, store)
```

With:
```python
def step(self, action, belief, store):
    result = super().step(action, belief, store)
    self._recorder.capture_frame()   # synchronous; main thread only; after physics completes
    return result
```

No other changes to `__init__`, `wrapup()`, or the FrameRecorder constructor call.

---

## 4. VERIFICATION

1. Set `record: true` in `env_configs/find_dice.yml` (or a test config)
2. Run find_dice for several steps
3. Confirm `generated.gif` is produced in `save_dir/frames/` without errors
4. Confirm no race condition warnings or hangs during execution

---

## 5. PROPAGATION

Once verified in find_dice, apply Change 2 (only `step()` pattern) to any new environment
that inherits from `TampuraEnv` and uses `FrameRecorder`.
Change 1 (`frame_recorder.py`) is shared and only needs to be done once.
