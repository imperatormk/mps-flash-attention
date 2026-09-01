"""
Setup script for MPS Flash Attention
"""

import os
import sys
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class ObjCppBuildExt(build_ext):
    """Build extension for Objective-C++ with PyTorch."""
    def build_extensions(self):
        # Import torch only when actually building
        import torch
        from torch.utils import cpp_extension

        # Register .mm as a valid source extension
        self.compiler.src_extensions.append('.mm')

        # Get original compile function
        original_compile = self.compiler._compile

        def objcpp_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.mm'):
                # Force Objective-C++ mode for .mm files
                extra_postargs = ['-x', 'objective-c++'] + list(extra_postargs or [])
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = objcpp_compile

        for ext in self.extensions:
            ext.include_dirs.extend(cpp_extension.include_paths())

        super().build_extensions()

        # Copy libMFABridge.dylib to lib/ after building
        self._copy_swift_bridge()

    def _copy_swift_bridge(self):
        """Copy Swift bridge dylib to package lib/ directory."""
        src_path = os.path.join(
            os.path.dirname(__file__),
            "swift-bridge", ".build", "release", "libMFABridge.dylib"
        )
        dst_dir = os.path.join(os.path.dirname(__file__), "mps_flash_attn", "lib")
        dst_path = os.path.join(dst_dir, "libMFABridge.dylib")

        if os.path.exists(src_path):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"Copied libMFABridge.dylib to {dst_path}")
        else:
            print(f"Warning: {src_path} not found. Build swift-bridge first with:")
            print("  cd swift-bridge && swift build -c release")


def get_extensions():
    if sys.platform != "darwin":
        return []

    # When packaging a final wheel we use prebuilt .so files dropped in-tree
    # by scripts/build_dual_wheel.sh, so skip ext_modules entirely.
    if os.environ.get("MFA_SKIP_EXT") == "1":
        return []

    # We build the same source twice with different names so a single wheel
    # can support torch ABIs on either side of the 2.10 vtable break.
    # Name is set via MFA_EXT_NAME env var (defaults to _C_legacy).
    ext_name = os.environ.get("MFA_EXT_NAME", "_C_legacy")
    # torch >= 2.12 headers need C++20; torch 2.5's headers specialize
    # std::is_arithmetic, which libc++ rejects in C++20 mode.
    cxx_std = "-std=c++17" if ext_name == "_C_legacy" else "-std=c++20"
    return [Extension(
        name=f"mps_flash_attn.{ext_name}",
        sources=["mps_flash_attn/csrc/mps_flash_attn.mm"],
        extra_compile_args=[
            cxx_std, "-O3",
            # Newer SDK libc++ marks std traits no_specializations; torch 2.5
            # headers specialize std::is_arithmetic.
            "-Wno-invalid-specialization",
            f"-DTORCH_EXTENSION_NAME={ext_name}",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
            "-Wl,-undefined,dynamic_lookup",
        ],
    )]


setup(
    name="mps-flash-attn",
    version="0.6.3",
    packages=find_packages(),
    package_data={
        "mps_flash_attn": [
            "lib/*.dylib",
            "kernels/*.metallib",
            "kernels/*.bin",
            "kernels/*.json",
            "_C_legacy*.so",
            "_C_modern*.so",
            "_C_next*.so",
        ],
    },
    include_package_data=True,
    install_requires=["torch>=2.5,<2.14"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": ObjCppBuildExt},
)
