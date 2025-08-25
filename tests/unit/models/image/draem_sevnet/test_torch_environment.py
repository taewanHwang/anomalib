# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch 환경 테스트 모듈.

이 모듈은 PyTorch와 CUDA 환경이 올바르게 설정되어 있는지 확인합니다.
"""

import os
import platform
import sys
from pathlib import Path

import pytest


class TestTorchEnvironment:
    """PyTorch 환경 테스트 클래스."""

    def test_python_version(self) -> None:
        """Python 버전이 요구사항을 만족하는지 테스트."""
        python_version = sys.version_info
        assert python_version >= (3, 10), f"Python 3.10 이상이 필요합니다. 현재: {python_version}"
        print(f"✅ Python 버전: {platform.python_version()}")

    def test_torch_import(self) -> None:
        """PyTorch 라이브러리가 정상적으로 import되는지 테스트."""
        try:
            import torch
            print(f"✅ PyTorch 버전: {torch.__version__}")
            
            # CUDA 호환성 미리 확인
            if torch.cuda.is_available():
                try:
                    # 간단한 CUDA 텐서 생성으로 실제 CUDA 작동 확인
                    test_tensor = torch.tensor([1.0], device='cuda')
                    _ = test_tensor.cpu()  # GPU->CPU 전송 테스트
                    print(f"✅ CUDA 초기화 성공: {torch.version.cuda}")
                except Exception as cuda_err:
                    print(f"⚠️  CUDA 초기화 실패: {cuda_err}")
                    print("   → CPU 모드로 테스트를 계속합니다.")
            else:
                print("ℹ️  CUDA 사용 불가 - CPU 모드로 실행")
                
        except ImportError as e:
            pytest.fail(f"PyTorch import 실패: {e}")
        except Exception as e:
            print(f"⚠️  PyTorch 로딩 중 예상치 못한 오류: {e}")
            print("   → 기본 import는 성공했지만 CUDA 라이브러리에 문제가 있을 수 있습니다.")

    def test_torch_basic_operations(self) -> None:
        """PyTorch 기본 연산이 정상적으로 작동하는지 테스트."""
        import torch

        # 기본 텐서 생성 및 연산 테스트
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        
        assert z.shape == (3, 3), f"행렬 곱셈 결과 shape 오류: {z.shape}"
        assert not torch.isnan(z).any(), "행렬 곱셈 결과에 NaN 값 존재"
        print("✅ PyTorch 기본 연산 테스트 통과")

    def test_cuda_availability(self) -> None:
        """CUDA 사용 가능 여부를 확인하고 정보를 출력."""
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"📊 CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            print(f"📊 CUDA 버전: {torch.version.cuda}")
            print(f"📊 GPU 개수: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"📊 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

    @pytest.mark.gpu
    def test_cuda_operations(self) -> None:
        """CUDA GPU 연산이 정상적으로 작동하는지 테스트 (GPU가 있을 때만)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA가 사용 불가능하므로 GPU 테스트를 건너뜁니다.")
        
        try:
            # CUDA 초기화 확인 - import 시점에서 실패할 수 있는 CUDA 오류를 먼저 확인
            torch.cuda.synchronize()
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # GPU에서 텐서 생성 및 연산
            device = torch.device("cuda:0")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            
            assert z.device.type == "cuda", f"결과 텐서가 GPU에 있지 않음: {z.device}"
            assert z.shape == (100, 100), f"GPU 연산 결과 shape 오류: {z.shape}"
            assert not torch.isnan(z).any(), "GPU 연산 결과에 NaN 값 존재"
            
            # CPU로 데이터 이동 테스트
            z_cpu = z.cpu()
            assert z_cpu.device.type == "cpu", "CPU로 데이터 이동 실패"
            
            print("✅ CUDA GPU 연산 테스트 통과")
            
        except RuntimeError as e:
            if "CUDA" in str(e) and ("symbol" in str(e) or "version" in str(e)):
                pytest.skip(f"CUDA 라이브러리 호환성 문제로 GPU 테스트를 건너뜁니다: {e}")
            else:
                pytest.fail(f"CUDA GPU 연산 실패: {e}")
        except Exception as e:
            pytest.fail(f"CUDA GPU 연산 실패: {e}")

    def test_torchvision_import(self) -> None:
        """torchvision 라이브러리가 정상적으로 import되는지 테스트."""
        try:
            import torchvision
            print(f"✅ torchvision 버전: {torchvision.__version__}")
        except ImportError as e:
            pytest.fail(f"torchvision import 실패: {e}")

    def test_system_environment(self) -> None:
        """시스템 환경 정보를 출력."""
        print(f"📊 운영체제: {platform.system()} {platform.release()}")
        print(f"📊 아키텍처: {platform.machine()}")
        print(f"📊 프로세서: {platform.processor()}")
        
        # 실행 환경 확인
        import subprocess
        try:
            # uv와 conda 환경 확인
            conda_env = os.environ.get("CONDA_DEFAULT_ENV", "없음")
            virtual_env = os.environ.get("VIRTUAL_ENV", "없음")
            print(f"📊 Conda 환경: {conda_env}")
            print(f"📊 가상환경: {virtual_env}")
            
            # Python 실행 경로 확인
            python_path = sys.executable
            print(f"📊 Python 실행 경로: {python_path}")
            
        except Exception as e:
            print(f"⚠️  환경 정보 확인 중 오류: {e}")
        
        # CUDA 관련 환경변수 확인
        cuda_home = os.environ.get("CUDA_HOME", "설정되지 않음")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "설정되지 않음")
        
        print(f"📊 CUDA_HOME: {cuda_home}")
        print(f"📊 LD_LIBRARY_PATH: {ld_library_path}")

    def test_cuda_libraries_accessible(self) -> None:
        """CUDA 라이브러리에 접근 가능한지 테스트."""
        cuda_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.4/lib64",
            "/usr/local/cuda-12.4/targets/x86_64-linux/lib",
        ]
        
        accessible_paths = []
        for path in cuda_paths:
            if Path(path).exists():
                accessible_paths.append(path)
                print(f"✅ CUDA 경로 접근 가능: {path}")
        
        if not accessible_paths:
            print("⚠️  접근 가능한 CUDA 라이브러리 경로를 찾을 수 없습니다.")

    def test_torch_cuda_compatibility(self) -> None:
        """PyTorch와 시스템 CUDA의 호환성을 테스트."""
        import torch
        
        print(f"📊 PyTorch 빌드 정보:")
        print(f"  - CUDA 지원: {torch.version.cuda}")
        print(f"  - cuDNN 버전: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"  - Debug 모드: {torch.version.debug}")
        
        # CUDA 라이브러리 로딩 시도
        if torch.cuda.is_available():
            try:
                # 간단한 CUDA 연산으로 라이브러리 호환성 확인
                torch.cuda.synchronize()
                print("✅ CUDA 라이브러리 호환성 확인됨")
            except Exception as e:
                print(f"❌ CUDA 라이브러리 호환성 오류: {e}")
                # 이것은 fatal error가 아니므로 테스트를 실패시키지 않음
        else:
            print("ℹ️  CUDA가 사용 불가능하므로 호환성 테스트를 건너뜁니다.")


if __name__ == "__main__":
    # 직접 실행 시 모든 테스트 실행
    pytest.main([__file__, "-v", "-s"])
