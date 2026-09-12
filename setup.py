import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "src" / "csrc"
EXTENSIONS_DIR = ROOT / "build" / "extensions"
CPP_ENGINE_DIR = ROOT / "cpp_engine"


def _build_native_requested() -> bool:
    """Whether to build the optional pybind11 module for the C++ engine.

    Off by default: the native build needs pybind11, a CUDA toolkit and NCCL, and
    takes minutes. A host that only wants the Torch runtimes must still install.
    """
    return os.environ.get("POCKETLLM_BUILD_CPP", "").strip().lower() in {"1", "true", "yes", "on"}


def _python_cmake_hints() -> list[str]:
    """Interpreter hints for CMake's FindPython3.

    ``-DPython3_EXECUTABLE`` alone is not enough against a conda environment:
    FindPython3 reports "missing: Development.Module" even though the headers are
    installed. Passing the root, include dir and library explicitly resolves it.
    """
    hints = [f"-DPython3_EXECUTABLE={sys.executable}"]

    prefix = sysconfig.get_config_var("prefix")
    if prefix:
        hints.append(f"-DPython3_ROOT_DIR={prefix}")

    include_dir = sysconfig.get_path("include")
    if include_dir and Path(include_dir).is_dir():
        hints.append(f"-DPython3_INCLUDE_DIR={include_dir}")

    # A shared libpython is not guaranteed to exist (static builds are valid), so
    # only pass the library when it is actually there.
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if libdir and ldlibrary:
        candidate = Path(libdir) / ldlibrary
        if candidate.is_file():
            hints.append(f"-DPython3_LIBRARY={candidate}")

    return hints


def _nccl_cmake_hints() -> list[str]:
    """Forward NCCL locations when the environment names them.

    A native module built without NCCL loads fine and then fails every TP>1 run
    with "Qwen TP requires an NCCL-enabled build", so an explicit location is
    worth forwarding rather than leaving to discovery.
    """
    hints = []
    for variable in ("NCCL_INCLUDE_DIR", "NCCL_LIBRARY", "NCCL_ROOT"):
        value = os.environ.get(variable)
        if value:
            hints.append(f"-D{variable}={value}")
    return hints


class BuildExtensions(BuildExtension):
    def run(self):
        super().run()
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for ext in self.extensions:
            built_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            if built_path.exists():
                shutil.copy2(built_path, EXTENSIONS_DIR / built_path.name)
        if _build_native_requested():
            self.build_native_module()

    def build_native_module(self):
        """Configure and build ``pocketllm_cpp`` through the engine's CMake project.

        Any failure here is fatal. A silently missing native module is worse than a
        failed install: ``backend="auto"`` would quietly fall back to Torch, and the
        caller who asked for the C++ engine would get different kernels than
        requested with no error to explain it.
        """
        try:
            import pybind11
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "POCKETLLM_BUILD_CPP=1 requires pybind11 (pip install pybind11)"
            ) from exc

        cmake = shutil.which("cmake")
        if cmake is None:  # pragma: no cover - environment dependent
            raise RuntimeError("POCKETLLM_BUILD_CPP=1 requires cmake on PATH")

        backend = os.environ.get("POCKETLLM_BACKEND", "cuda")
        build_dir = Path(self.build_temp).resolve() / "cpp_engine"
        build_dir.mkdir(parents=True, exist_ok=True)

        configure = [
            cmake,
            "-S", str(CPP_ENGINE_DIR),
            "-B", str(build_dir),
            f"-DPOCKET_BACKEND={backend}",
            "-DPOCKET_BUILD_PYTHON=ON",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]
        configure.extend(_python_cmake_hints())
        configure.extend(_nccl_cmake_hints())

        subprocess.check_call(configure)
        subprocess.check_call([
            cmake, "--build", str(build_dir),
            "--target", "pocketllm_cpp",
            "-j", str(os.cpu_count() or 1),
        ])

        built = sorted((build_dir / "python").glob("pocketllm_cpp*.so"))
        if not built:
            raise RuntimeError(
                f"pocketllm_cpp built but no shared object was found under {build_dir / 'python'}"
            )

        # Install top-level, next to the Torch extensions, so `import pocketllm_cpp`
        # resolves without the manual copy onto sys.path the old flow required.
        destination_dir = Path(self.build_lib).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        for module in built:
            shutil.copy2(module, destination_dir / module.name)
            shutil.copy2(module, EXTENSIONS_DIR / module.name)


setup(
    # Resolved from the tree rather than hand-listed; the previous literal had
    # drifted, omitting src.components.gguf, src.models.glm_dsa and
    # src.models.qwen4_exp while naming two directories that are not packages.
    # find_namespace_packages picks up src.components and src.loader.mappings,
    # which are real namespace packages actively imported but have no __init__.py.
    packages=find_namespace_packages(
        include=["pocketllm", "pocketllm.*", "src", "src.*"],
        # src.csrc holds only C++/CUDA sources; src.gguf and src.moe are stale empty dirs.
        exclude=["src.csrc", "src.csrc.*", "src.gguf", "src.moe"],
    ),
    ext_modules=[
        CUDAExtension(
            name="cuda_kernel",
            sources=[
                "src/csrc/cuda_kernel.cpp",
                "src/csrc/cuda_kernel_impl.cu",
                "src/csrc/minimax_rope_kernel.cu",
                "src/csrc/llama_mmq/gguf_mma_wrapper.cu",
                "src/csrc/qwen4_exp_moe.cu",
                "src/csrc/qwen4_exp_gated_delta.cu",
                "src/csrc/qwen4_exp_qsa.cu",
                "src/csrc/qwen4_exp_hyper_connection.cu",
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
        Extension(
            name="deepseek_cpu_moe_ext",
            sources=["src/csrc/deepseek_cpu_moe_ext.cpp"],
            extra_compile_args=["-O3", "-mavx2", "-mfma", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            name="moe_dispatch_cuda_ext",
            sources=[
                "src/csrc/moe_dispatch_cuda_ext.cpp",
                "src/csrc/moe_dispatch_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtensions},
)
