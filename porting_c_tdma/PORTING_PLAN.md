# PaScaL_TDMAcuda CUDA C++ Porting Plan

date: 2026-06-30
source: `../src/PaScaL_TDMA_cuda.f90`
reference report: `../../brain/4_260630_PaScaL_TDMAcuda_analysis.md`

## 1. Porting 기준

- NVIDIA가 공식 지원하는 CUDA C++ runtime API와 `nvcc`를 기준으로 한다.
- 원본 CUDA Fortran의 알고리즘, MPI all-to-all flow, device buffer layout, column-major 2D indexing을 1차 포팅에서 보존한다.
- CUDA 지원 하드웨어가 현재 없으므로 이번 단계는 소스 작성과 정적 구조 검토까지 수행한다. 실제 GPU 실행 검증은 NVCC + CUDA-aware MPI + GPU 환경에서 따로 해야 한다.

## 2. 원본에서 보존할 핵심 계약

### 2.1 Solver flow

원본 `pascal_solver`의 흐름을 그대로 보존한다.

단일 MPI rank:

```text
tdma_many_cuda
```

복수 MPI rank:

```text
tdma_modified_cuda
  -> pack Ard/Crd/Drd
  -> MPI_Alltoallv
  -> unpack Atr/Ctr/Dtr
  -> initialize Btr
  -> tdma_many_cuda on transformed system
  -> pack solved Dtr
  -> MPI_Alltoallv
  -> unpack Drd
  -> pascal_update
```

### 2.2 Layout

Fortran device array `A(0:Nsys-1,0:Nrow-1)`는 첫 번째 index가 contiguous인 column-major 2D layout이다. CUDA C++에서는 다음 flat indexing을 표준으로 한다.

```cpp
index2(sys, row, nsys) = sys + row * nsys;
```

예제의 3D 격자 `(i,j,k)`는 solver view에서 다음으로 해석한다.

```cpp
sys = i + j * n1;
row = k;
```

### 2.3 Plan ownership

원본 `ptdma_plan_cuda`에 해당하는 C++ type은 다음을 소유한다.

- MPI communicator/rank metadata.
- reduced arrays: `Ard/Brd/Crd/Drd`, shape `(Nsys,2)`.
- transformed arrays: `Atr/Btr/Ctr/Dtr`, shape `(local_Nsys,2*nprocs)`.
- all-to-all count/displacement descriptors.
- device communication buffers.

C++에서는 RAII를 적용해 `PascalTdmaPlan` destructor가 device allocations를 해제한다.

## 3. MPI 전략

기본 경로는 CUDA-aware MPI다.

```text
device pointer -> MPI_Alltoallv -> device pointer
```

현재 개발 머신은 CUDA 하드웨어가 없고, target MPI의 CUDA-aware 지원도 여기서 확인할 수 없다. 따라서 host staging fallback을 함께 둔다.

```text
device -> host staging -> MPI_Alltoallv -> host staging -> device
```

선택 방법:

- 기본값: `device-direct`
- 환경변수 `PASCAL_TDMA_MPI_MODE=host`: host staging fallback

## 4. Build 전략

기본 build는 `nvcc` + MPI C++ wrapper를 사용한다.

- `NVCC ?= nvcc`
- `MPICXX ?= mpicxx`
- `CUDA_ARCH ?= 80`
- `nvcc -ccbin $(MPICXX) ...`

현 로컬에는 `mpicxx`는 있으나 `nvcc`가 확인되지 않았다. 따라서 Makefile은 작성하되, 실제 build는 CUDA toolkit이 있는 환경에서 수행한다.

## 5. 자체 검토와 수정 사항

### 검토 1

- 문제: 원본은 device gather descriptor 배열을 kernel에 넘기지만 C++에서는 host loop가 이미 rank별 subsize/start를 알고 있다.
- 판단: C++ kernel argument를 scalar `sub0/sub1/start0/start1`로 단순화해도 pack/unpack 결과는 동일하다.
- 수정: device gather descriptor 배열은 만들지 않는다. 대신 `launch_pack`/`launch_unpack` wrapper가 rank별 scalar descriptor를 전달한다.

### 검토 2

- 문제: 원본의 `BIGbuf_A/B`는 coefficient 3개를 담는 큰 buffer이지만, 두 번째 all-to-all은 solved `Dtr`만 사용한다.
- 판단: 원본처럼 같은 buffer를 재사용하되, 두 번째 exchange에서는 minimal count/displacement를 사용한다.
- 수정: `big_counts_*`와 `counts_*`를 모두 plan에 유지한다.

### 검토 3

- 문제: 현재 CUDA hardware가 없어서 correctness test를 돌릴 수 없다.
- 판단: CUDA runtime error check와 명확한 example을 제공하고, 실제 검증은 target 환경으로 넘긴다.
- 수정: 모든 CUDA API call은 `PASCAL_TDMA_CUDA_CHECK`로 감싼다. MPI all-to-all 전 stream synchronize를 강제한다.

### 결론

사소한 naming 차이를 제외하면, 이번 1차 계획은 원본 알고리즘과 layout을 보존한다. 성능 최적화, line 내부 병렬화, PCR/CR 계열 변경은 이번 포팅 범위 밖으로 둔다.
