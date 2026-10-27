"""GPU Acceleration Support.

Provides GPU-accelerated computation for APGI framework operations.
Leverages JAX/TensorFlow for automatic GPU utilization.
Includes hardware detection, memory management, and fallback to CPU.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""

    available: bool
    device_name: Optional[str] = None
    memory_total: Optional[int] = None  # MB
    memory_free: Optional[int] = None  # MB
    compute_capability: Optional[str] = None
    backend: Optional[str] = None


class GPUAccelerationManager:
    """Manager for GPU acceleration in APGI framework.

    Handles GPU detection, memory management, and provides
    hardware-accelerated computation functions.
    """

    def __init__(self) -> None:
        self._jax_available = False
        self._tensorflow_available = False
        self._torch_available = False
        self._gpu_info: Optional[GPUInfo] = None
        self._preferred_backend: Optional[str] = None

        self._detect_backends()
        self._detect_gpu()

    def _detect_backends(self) -> None:
        """Detect available acceleration backends."""
        # Check JAX
        try:
            import jax

            self._jax_available = True
            logger.info(f"JAX {jax.__version__} detected")
        except ImportError:
            logger.debug("JAX not available")

        # Check TensorFlow
        try:
            import tensorflow as tf

            self._tensorflow_available = True
            logger.info(f"TensorFlow {tf.__version__} detected")
        except ImportError:
            logger.debug("TensorFlow not available")

        # Check PyTorch
        try:
            import torch

            self._torch_available = True
            logger.info(f"PyTorch {torch.__version__} detected")
        except ImportError:
            logger.debug("PyTorch not available")

    def _detect_gpu(self) -> None:
        """Detect GPU hardware and capabilities."""
        gpu_available = False
        device_name = None
        backend = None

        # Check JAX GPU
        if self._jax_available:
            try:
                import jax

                devices = jax.devices()
                gpu_devices = [d for d in devices if d.platform == "gpu"]
                if gpu_devices:
                    gpu_available = True
                    device_name = str(gpu_devices[0])
                    backend = "jax"
                    logger.info(f"JAX GPU detected: {device_name}")
            except Exception as e:
                logger.debug(f"JAX GPU detection failed: {e}")

        # Check TensorFlow GPU
        if not gpu_available and self._tensorflow_available:
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    gpu_available = True
                    device_name = gpus[0].name
                    backend = "tensorflow"
                    logger.info(f"TensorFlow GPU detected: {device_name}")
            except Exception as e:
                logger.debug(f"TensorFlow GPU detection failed: {e}")

        # Check PyTorch GPU
        if not gpu_available and self._torch_available:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_available = True
                    device_name = torch.cuda.get_device_name(0)
                    backend = "pytorch"
                    logger.info(f"PyTorch CUDA GPU detected: {device_name}")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    gpu_available = True
                    device_name = "Apple Metal Performance Shaders"
                    backend = "pytorch_mps"
                    logger.info(f"PyTorch MPS detected: {device_name}")
            except Exception as e:
                logger.debug(f"PyTorch GPU detection failed: {e}")

        self._gpu_info = GPUInfo(
            available=gpu_available,
            device_name=device_name,
            backend=backend,
        )

        if gpu_available:
            logger.info(f"GPU acceleration enabled via {backend}")
        else:
            logger.info("GPU not available, using CPU")

    @property
    def gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_info.available if self._gpu_info else False

    @property
    def gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU device information."""
        return self._gpu_info

    @property
    def preferred_backend(self) -> Optional[str]:
        """Get preferred acceleration backend."""
        if self._preferred_backend:
            return self._preferred_backend
        return self._gpu_info.backend if self._gpu_info else None

    def set_preferred_backend(self, backend: str) -> bool:
        """Set preferred backend for GPU acceleration.

        Args:
            backend: Backend name (jax, tensorflow, pytorch)

        Returns:
            True if backend is available
        """
        backend_map = {
            "jax": self._jax_available,
            "tensorflow": self._tensorflow_available,
            "pytorch": self._torch_available,
        }

        if backend_map.get(backend, False):
            self._preferred_backend = backend
            logger.info(f"Preferred backend set to: {backend}")
            return True
        else:
            logger.warning(f"Backend {backend} not available")
            return False

    def get_array_module(self) -> Any:
        """Get array module with GPU support.

        Returns:
            numpy-like module (jax.numpy, tensorflow, or numpy)
        """
        backend = self.preferred_backend

        if backend == "jax" and self._jax_available:
            import jax.numpy as jnp

            return jnp
        elif backend == "tensorflow" and self._tensorflow_available:
            import tensorflow as tf

            return tf
        elif backend == "pytorch" and self._torch_available:
            import torch

            return torch

        return np

    def to_device(self, array: np.ndarray) -> Any:
        """Move numpy array to GPU device.

        Args:
            array: Input numpy array

        Returns:
            Array on GPU device
        """
        backend = self.preferred_backend

        if backend == "jax" and self._jax_available:
            import jax

            return jax.device_put(array)
        elif backend == "tensorflow" and self._tensorflow_available:
            import tensorflow as tf

            return tf.convert_to_tensor(array)
        elif backend in ("pytorch", "pytorch_mps") and self._torch_available:
            import torch

            device = "cuda" if torch.cuda.is_available() else "mps"
            tensor = torch.from_numpy(array)
            return tensor.to(device)

        return array

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert device array back to numpy.

        Args:
            array: Device array

        Returns:
            Numpy array
        """
        backend = self.preferred_backend

        if backend == "jax" and self._jax_available:
            return np.array(array)
        elif backend == "tensorflow" and self._tensorflow_available:
            return array.numpy()
        elif backend in ("pytorch", "pytorch_mps") and self._torch_available:
            import torch

            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()

        return np.array(array)

    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations.

        Ensures proper device placement and cleanup.
        """
        if not self.gpu_available:
            yield self
            return

        try:
            logger.debug("Entering GPU context")
            yield self
        finally:
            logger.debug("Exiting GPU context")
            # Force synchronization if needed
            if self.preferred_backend == "jax" and self._jax_available:
                import jax

                jax.block_until_ready(jax.numpy.array([0.0]))

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Matrix product
        """
        xp = self.get_array_module()
        a_device = self.to_device(a)
        b_device = self.to_device(b)
        result = xp.matmul(a_device, b_device)
        return self.to_numpy(result)

    def fft(self, a: np.ndarray) -> np.ndarray:
        """GPU-accelerated Fast Fourier Transform.

        Args:
            a: Input array

        Returns:
            FFT result
        """
        xp = self.get_array_module()
        a_device = self.to_device(a)
        result = xp.fft.fft(a_device)
        return self.to_numpy(result)

    def stats(self) -> dict:
        """Get GPU acceleration statistics."""
        return {
            "gpu_available": self.gpu_available,
            "gpu_info": {
                "available": self._gpu_info.available if self._gpu_info else False,
                "device_name": self._gpu_info.device_name if self._gpu_info else None,
                "backend": self._gpu_info.backend if self._gpu_info else None,
            },
            "backends": {
                "jax": self._jax_available,
                "tensorflow": self._tensorflow_available,
                "pytorch": self._torch_available,
            },
            "preferred_backend": self.preferred_backend,
        }


def enable_gpu_acceleration() -> GPUAccelerationManager:
    """Enable and return GPU acceleration manager.

    Returns:
        Configured GPUAccelerationManager instance
    """
    manager = GPUAccelerationManager()
    if manager.gpu_available:
        logger.info("GPU acceleration enabled successfully")
    else:
        logger.info("Running in CPU mode (GPU not available)")
    return manager


# Global GPU acceleration manager instance
gpu_manager: Optional[GPUAccelerationManager] = None


def get_gpu_manager() -> GPUAccelerationManager:
    """Get or create global GPU acceleration manager."""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = GPUAccelerationManager()
    return gpu_manager
