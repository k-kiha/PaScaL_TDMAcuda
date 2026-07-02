# PaScaL_TDMAcuda Study Case Matrix

작성일: 2026-07-02

목적: `PaScaL_TDMAcuda/Study`에서 수행할 1~10번 Study별로 어떤 격자, 어떤 GPU/MPI rank 수, 어떤 실행 옵션의 case를 계산해야 하는지 전수 목록을 고정한다.

## 공통 실행 조건

기본 실행 단위는 MPI rank 수와 GPU 수를 1:1로 대응시키는 것이다.

```text
GPU count = MPI nranks = np
CUDA_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7
ITERATIONS = 10
TDMA_THREADS = 128
REDUCED_THREADS = 128
BASELINE_NP = 2
SCALING_NP_LIST = 2 4 8
RUN_FORTRAN = 1
RUN_CXX = 1
CXX_DEFAULT_MPI_MODES = device
MPI_MODE_LIST = device host
```

빌드 기준:

```text
CUDA_ARCH = 90
Fortran original = CUDA Fortran + MPI
CUDA C++ port = CUDA C++ + MPI
```

해석 기준:

```text
iter = 0      : first-run correctness 및 warm-up 관찰
iter = 1..9   : 안정화된 timing 분석
total 기준    : total_s_max
rank 평균 보조: total_s_avg
```

TDMA 문제 구조:

```text
nsys = n1 * n2
nrow = n3 / nranks
global problem size = n1 * n2 * n3
```

## 실제 unique 실행 case 목록

아래 22개 `(np, n1, n2, n3)` 조합이 full Study에서 실제로 필요한 unique base case이다.

| np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 주요 용도 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 64 | 2048 | 4096 | 2048 | single GPU reference, correctness |
| 1 | 128 | 128 | 4096 | 16384 | 4096 | single GPU reference, correctness |
| 2 | 64 | 64 | 4096 | 4096 | 2048 | nsys sensitivity |
| 2 | 128 | 128 | 2048 | 16384 | 1024 | weak nrow, weak nsys, nrow sensitivity |
| 2 | 128 | 128 | 4096 | 16384 | 2048 | strong baseline, nsys/nrow sensitivity, MPI mode |
| 2 | 128 | 128 | 8192 | 16384 | 4096 | nrow sensitivity |
| 2 | 128 | 256 | 4096 | 32768 | 2048 | nsys sensitivity |
| 2 | 256 | 256 | 4096 | 65536 | 2048 | strong baseline, large nsys |
| 4 | 64 | 64 | 4096 | 4096 | 1024 | nsys sensitivity |
| 4 | 128 | 128 | 2048 | 16384 | 512 | nrow sensitivity |
| 4 | 128 | 128 | 4096 | 16384 | 1024 | strong scaling, weak nrow, MPI mode |
| 4 | 128 | 128 | 8192 | 16384 | 2048 | nrow sensitivity |
| 4 | 128 | 256 | 2048 | 32768 | 512 | weak nsys |
| 4 | 128 | 256 | 4096 | 32768 | 1024 | nsys sensitivity |
| 4 | 256 | 256 | 4096 | 65536 | 1024 | strong scaling, large nsys |
| 8 | 64 | 64 | 4096 | 4096 | 512 | nsys sensitivity |
| 8 | 128 | 128 | 2048 | 16384 | 256 | nrow sensitivity |
| 8 | 128 | 128 | 4096 | 16384 | 512 | strong scaling, nsys/nrow sensitivity, MPI mode |
| 8 | 128 | 128 | 8192 | 16384 | 1024 | weak nrow, nrow sensitivity |
| 8 | 128 | 256 | 4096 | 32768 | 512 | nsys sensitivity |
| 8 | 128 | 512 | 2048 | 65536 | 256 | weak nsys |
| 8 | 256 | 256 | 4096 | 65536 | 512 | strong scaling, large nsys |

실제 실행 구현 조합:

```text
Fortran original:
  위 22개 base case 전체 실행
  mpi_mode 표기는 device로 기록

CUDA C++ port default:
  위 22개 base case 전체 실행
  mpi_mode = device

CUDA C++ MPI mode comparison:
  128,128,4096에서 np=2,4,8만 추가 host mode 실행
  mpi_mode = host
```

따라서 기본 full Study의 실제 실행 수는 다음과 같다.

```text
Fortran original device : 22 runs
CUDA C++ device         : 22 runs
CUDA C++ host           : 3 runs
total                   : 47 runs
```

각 run은 `ITERATIONS=10`을 수행하므로 timing raw row는 최소 470개가 된다.

## 1. Correctness

목적:

Fortran original과 CUDA C++ port가 같은 해를 만드는지 확인한다. 현재 test problem의 기대 해는 모든 위치에서 `1.0`이다.

필요 case:

Correctness는 특정 subset만 보지 않고, 실제 unique 실행 case 전체에서 확인한다.

| 구현 | MPI mode | 대상 case |
|---|---|---|
| Fortran original | device 표기 | 22개 base case 전체 |
| CUDA C++ port | device | 22개 base case 전체 |
| CUDA C++ port | host | MPI mode 비교용 3개 case |

확인할 격자/GPU 수:

```text
np=1:
  64,64,2048
  128,128,4096

np=2:
  64,64,4096
  128,128,2048
  128,128,4096
  128,128,8192
  128,256,4096
  256,256,4096

np=4:
  64,64,4096
  128,128,2048
  128,128,4096
  128,128,8192
  128,256,2048
  128,256,4096
  256,256,4096

np=8:
  64,64,4096
  128,128,2048
  128,128,4096
  128,128,8192
  128,256,4096
  128,512,2048
  256,256,4096
```

추가 host mode correctness:

```text
CUDA C++ host mode:
  np=2, 128,128,4096
  np=4, 128,128,4096
  np=8, 128,128,4096
```

필수 출력:

```text
tdma_correctness_*.csv
```

확인 columns:

```text
solution_sum
solution_l2
solution_linf
sample_z0
sample_zmid
sample_zlast
expected_value
max_abs_error_to_expected
```

## 2. Total Performance

목적:

Fortran original과 CUDA C++ port의 전체 solve 시간을 같은 case에서 비교한다.

필요 case:

Total performance는 22개 base case 전체에서 수집한다.

| 구현 | MPI mode | 대상 case |
|---|---|---|
| Fortran original | device 표기 | 22개 base case 전체 |
| CUDA C++ port | device | 22개 base case 전체 |
| CUDA C++ port | host | MPI mode 비교용 3개 case |

해석 기준:

```text
iter=0    : warm-up 포함 first solve
iter=1..9 : 안정화 timing
주요 지표 : total_s_max
보조 지표 : total_s_avg
```

필수 출력:

```text
tdma_total_profile_*.csv
```

## 3. Compute vs Communication Breakdown

목적:

전체 solve 시간이 계산, MPI 통신, packing/unpacking 중 어디에서 소비되는지 분해한다.

필요 case:

Breakdown도 22개 base case 전체에서 수집한다. MPI mode 비교용 3개 case는 C++ host mode도 추가로 breakdown을 본다.

| 구현 | MPI mode | 대상 case |
|---|---|---|
| Fortran original | device 표기 | 22개 base case 전체 |
| CUDA C++ port | device | 22개 base case 전체 |
| CUDA C++ port | host | `np=2,4,8`, `128,128,4096` |

확인 columns:

```text
local_compute_s_max
pack_forward_s_max
mpi_forward_s_max
unpack_forward_s_max
reduced_compute_s_max
pack_backward_s_max
mpi_backward_s_max
unpack_backward_s_max
update_compute_s_max
compute_s_max
communication_s_max
packing_s_max
```

중요 비교:

```text
compute fraction       = compute_s_max / total_s_max
communication fraction = communication_s_max / total_s_max
packing fraction       = packing_s_max / total_s_max
```

## 4. Fortran vs CUDA C++ Performance Comparison

목적:

원본 CUDA Fortran 구현 대비 CUDA C++ port가 어느 정도의 성능을 보존하는지 확인한다.

필요 case:

같은 조건 비교를 위해 Fortran original과 CUDA C++ device mode가 모두 있는 22개 base case 전체를 사용한다.

| 비교 | MPI mode | 대상 case |
|---|---|---|
| Fortran original vs CUDA C++ port | device 기준 | 22개 base case 전체 |

비교에서 제외하거나 별도로 다룰 case:

```text
CUDA C++ host mode 3개 case는 Fortran vs C++ 기본 성능 비교가 아니라 MPI mode 비교용이다.
```

계산 지표:

```text
cxx_over_fortran_total = total_s_max_cxx_device / total_s_max_fortran
cxx_speed_ratio        = total_s_max_fortran / total_s_max_cxx_device
```

## 5. Strong Scaling

목적:

Global problem size를 고정하고 GPU/rank 수를 늘릴 때 성능이 얼마나 좋아지는지 확인한다.

Baseline:

```text
baseline np = 2
rank/GPU list = 2, 4, 8
mpi mode = device
```

전수 case:

| study_suite | np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 옵션 |
|---|---:|---:|---:|---:|---:|---:|---|
| strong_scaling | 2 | 128 | 128 | 4096 | 16384 | 2048 | Fortran, C++ device |
| strong_scaling | 4 | 128 | 128 | 4096 | 16384 | 1024 | Fortran, C++ device |
| strong_scaling | 8 | 128 | 128 | 4096 | 16384 | 512 | Fortran, C++ device |
| strong_scaling | 2 | 256 | 256 | 4096 | 65536 | 2048 | Fortran, C++ device |
| strong_scaling | 4 | 256 | 256 | 4096 | 65536 | 1024 | Fortran, C++ device |
| strong_scaling | 8 | 256 | 256 | 4096 | 65536 | 512 | Fortran, C++ device |

계산 지표:

```text
T_base = T_2
p_rel = p / 2
speedup_2base(p)    = T_2 / T_p
efficiency_2base(p) = T_2 / ((p / 2) * T_p)
throughput(p)       = (n1 * n2 * n3) / total_s_max
```

## 6. Weak Scaling - nrow 방향

목적:

`nsys=n1*n2`를 고정하고 `n3`를 rank 수에 비례시켜 local `nrow=n3/np`를 유지한다. 이 경로는 global z-line length와 distributed reduced-system 통신 구조의 영향을 보기 위한 것이다.

Baseline:

```text
baseline np = 2
rank/GPU list = 2, 4, 8
mpi mode = device
fixed nsys = 128 * 128 = 16384
local nrow = 1024
```

전수 case:

| study_suite | np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 옵션 |
|---|---:|---:|---:|---:|---:|---:|---|
| weak_nrow_scaling | 2 | 128 | 128 | 2048 | 16384 | 1024 | Fortran, C++ device |
| weak_nrow_scaling | 4 | 128 | 128 | 4096 | 16384 | 1024 | Fortran, C++ device |
| weak_nrow_scaling | 8 | 128 | 128 | 8192 | 16384 | 1024 | Fortran, C++ device |

관찰 포인트:

```text
communication_s_max
packing_s_max
reduced_compute_s_max
total_s_max 유지/증가 여부
```

## 7. Weak Scaling - nsys 방향

목적:

`n3`를 고정하고 `nsys=n1*n2`를 rank 수에 맞춰 키운다. 이 경로는 independent TDMA system 수가 늘 때 GPU 병렬성이 어떻게 변하는지 보기 위한 것이다.

Baseline:

```text
baseline np = 2
rank/GPU list = 2, 4, 8
mpi mode = device
fixed n3 = 2048
```

전수 case:

| study_suite | np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 옵션 |
|---|---:|---:|---:|---:|---:|---:|---|
| weak_nsys_scaling | 2 | 128 | 128 | 2048 | 16384 | 1024 | Fortran, C++ device |
| weak_nsys_scaling | 4 | 128 | 256 | 2048 | 32768 | 512 | Fortran, C++ device |
| weak_nsys_scaling | 8 | 128 | 512 | 2048 | 65536 | 256 | Fortran, C++ device |

관찰 포인트:

```text
local_compute_s_max
update_compute_s_max
throughput
compute_s_max / communication_s_max 비율
```

## 8. nsys Sensitivity

목적:

`n3`를 고정하고 `nsys=n1*n2`만 변화시켜 independent TDMA system 수의 영향을 본다.

조건:

```text
np/GPU = 2, 4, 8
n3 = 4096
mpi mode = device
```

전수 case:

| study_suite | np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 옵션 |
|---|---:|---:|---:|---:|---:|---:|---|
| nsys_sensitivity | 2 | 64 | 64 | 4096 | 4096 | 2048 | Fortran, C++ device |
| nsys_sensitivity | 2 | 128 | 128 | 4096 | 16384 | 2048 | Fortran, C++ device |
| nsys_sensitivity | 2 | 128 | 256 | 4096 | 32768 | 2048 | Fortran, C++ device |
| nsys_sensitivity | 4 | 64 | 64 | 4096 | 4096 | 1024 | Fortran, C++ device |
| nsys_sensitivity | 4 | 128 | 128 | 4096 | 16384 | 1024 | Fortran, C++ device |
| nsys_sensitivity | 4 | 128 | 256 | 4096 | 32768 | 1024 | Fortran, C++ device |
| nsys_sensitivity | 8 | 64 | 64 | 4096 | 4096 | 512 | Fortran, C++ device |
| nsys_sensitivity | 8 | 128 | 128 | 4096 | 16384 | 512 | Fortran, C++ device |
| nsys_sensitivity | 8 | 128 | 256 | 4096 | 32768 | 512 | Fortran, C++ device |

관찰 포인트:

```text
nsys 증가에 따른 throughput 변화
compute_s_max 증가율
communication_s_max 증가율
GPU 병렬 작업량 부족 여부
```

## 9. nrow Sensitivity

목적:

`nsys=n1*n2`를 고정하고 `n3`만 변화시켜 local row length와 global line length의 영향을 본다.

조건:

```text
np/GPU = 2, 4, 8
n1,n2 = 128,128
nsys = 16384
mpi mode = device
```

전수 case:

| study_suite | np/GPU | n1 | n2 | n3 | nsys | nrow per rank | 옵션 |
|---|---:|---:|---:|---:|---:|---:|---|
| nrow_sensitivity | 2 | 128 | 128 | 2048 | 16384 | 1024 | Fortran, C++ device |
| nrow_sensitivity | 2 | 128 | 128 | 4096 | 16384 | 2048 | Fortran, C++ device |
| nrow_sensitivity | 2 | 128 | 128 | 8192 | 16384 | 4096 | Fortran, C++ device |
| nrow_sensitivity | 4 | 128 | 128 | 2048 | 16384 | 512 | Fortran, C++ device |
| nrow_sensitivity | 4 | 128 | 128 | 4096 | 16384 | 1024 | Fortran, C++ device |
| nrow_sensitivity | 4 | 128 | 128 | 8192 | 16384 | 2048 | Fortran, C++ device |
| nrow_sensitivity | 8 | 128 | 128 | 2048 | 16384 | 256 | Fortran, C++ device |
| nrow_sensitivity | 8 | 128 | 128 | 4096 | 16384 | 512 | Fortran, C++ device |
| nrow_sensitivity | 8 | 128 | 128 | 8192 | 16384 | 1024 | Fortran, C++ device |

관찰 포인트:

```text
nrow 증가에 따른 local_compute_s_max
nrow 증가에 따른 update_compute_s_max
nrow 증가에 따른 reduced_compute_s_max
nrow 증가에 따른 communication_s_max
```

## 10. MPI Mode Comparison

목적:

CUDA C++ port에서 CUDA-aware MPI device-buffer path와 host-staging fallback의 차이를 확인한다.

조건:

```text
implementation = CUDA C++ port
mpi_mode = device, host
rank/GPU list = 2, 4, 8
grid = 128,128,4096
```

전수 case:

| study_suite | 구현 | MPI mode | np/GPU | n1 | n2 | n3 | nsys | nrow per rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mpi_mode_compare | CUDA C++ port | device | 2 | 128 | 128 | 4096 | 16384 | 2048 |
| mpi_mode_compare | CUDA C++ port | host | 2 | 128 | 128 | 4096 | 16384 | 2048 |
| mpi_mode_compare | CUDA C++ port | device | 4 | 128 | 128 | 4096 | 16384 | 1024 |
| mpi_mode_compare | CUDA C++ port | host | 4 | 128 | 128 | 4096 | 16384 | 1024 |
| mpi_mode_compare | CUDA C++ port | device | 8 | 128 | 128 | 4096 | 16384 | 512 |
| mpi_mode_compare | CUDA C++ port | host | 8 | 128 | 128 | 4096 | 16384 | 512 |

Fortran original reference:

```text
Fortran original도 같은 np=2,4,8 / 128,128,4096 case가 실행된다.
하지만 MPI mode 비교 자체는 CUDA C++ port의 device vs host 비교로 해석한다.
```

관찰 포인트:

```text
total_s_max_device vs total_s_max_host
mpi_forward_s_max_device vs mpi_forward_s_max_host
mpi_backward_s_max_device vs mpi_backward_s_max_host
packing_s_max_device vs packing_s_max_host
correctness device/host 일치 여부
```

## Study suite manifest 전수 목록

Full Study 기준 suite row는 35개이다. 이 목록은 “왜 이 case가 필요한지”를 보존하기 위한 것이며, 실제 실행은 unique base case 22개와 host mode 비교 3개로 수행된다.

| suite | np | n1 | n2 | n3 | cxx_mpi_modes | notes |
|---|---:|---:|---:|---:|---|---|
| single_gpu_reference | 1 | 64 | 64 | 2048 | device | local_tdma_no_mpi_small |
| single_gpu_reference | 1 | 128 | 128 | 4096 | device | local_tdma_no_mpi_medium |
| strong_scaling | 2 | 128 | 128 | 4096 | device | fixed_global_medium |
| strong_scaling | 2 | 256 | 256 | 4096 | device | fixed_global_nsys_rich |
| strong_scaling | 4 | 128 | 128 | 4096 | device | fixed_global_medium |
| strong_scaling | 4 | 256 | 256 | 4096 | device | fixed_global_nsys_rich |
| strong_scaling | 8 | 128 | 128 | 4096 | device | fixed_global_medium |
| strong_scaling | 8 | 256 | 256 | 4096 | device | fixed_global_nsys_rich |
| weak_nrow_scaling | 2 | 128 | 128 | 2048 | device | fixed_nsys_local_nrow_1024 |
| weak_nrow_scaling | 4 | 128 | 128 | 4096 | device | fixed_nsys_local_nrow_1024 |
| weak_nrow_scaling | 8 | 128 | 128 | 8192 | device | fixed_nsys_local_nrow_1024 |
| weak_nsys_scaling | 2 | 128 | 128 | 2048 | device | n2_scaled_local_work_constant |
| weak_nsys_scaling | 4 | 128 | 256 | 2048 | device | n2_scaled_local_work_constant |
| weak_nsys_scaling | 8 | 128 | 512 | 2048 | device | n2_scaled_local_work_constant |
| nsys_sensitivity | 2 | 64 | 64 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 2 | 128 | 128 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 2 | 128 | 256 | 4096 | device | n3_fixed_vary_nsys |
| nrow_sensitivity | 2 | 128 | 128 | 2048 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 2 | 128 | 128 | 4096 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 2 | 128 | 128 | 8192 | device | nsys_fixed_vary_n3 |
| nsys_sensitivity | 4 | 64 | 64 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 4 | 128 | 128 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 4 | 128 | 256 | 4096 | device | n3_fixed_vary_nsys |
| nrow_sensitivity | 4 | 128 | 128 | 2048 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 4 | 128 | 128 | 4096 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 4 | 128 | 128 | 8192 | device | nsys_fixed_vary_n3 |
| nsys_sensitivity | 8 | 64 | 64 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 8 | 128 | 128 | 4096 | device | n3_fixed_vary_nsys |
| nsys_sensitivity | 8 | 128 | 256 | 4096 | device | n3_fixed_vary_nsys |
| nrow_sensitivity | 8 | 128 | 128 | 2048 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 8 | 128 | 128 | 4096 | device | nsys_fixed_vary_n3 |
| nrow_sensitivity | 8 | 128 | 128 | 8192 | device | nsys_fixed_vary_n3 |
| mpi_mode_compare | 2 | 128 | 128 | 4096 | device host | cxx_device_vs_host_for_same_case |
| mpi_mode_compare | 4 | 128 | 128 | 4096 | device host | cxx_device_vs_host_for_same_case |
| mpi_mode_compare | 8 | 128 | 128 | 4096 | device host | cxx_device_vs_host_for_same_case |

## 실행 방법

서버에서 full Study를 한 번에 실행한다.

```bash
cd PaScaL_TDMAcuda/Study
./run_full_study.sh
```

`run_full_study.sh`는 이 문서의 unique base case 22개와 host mode 비교 3개를 순서대로 `STUDY_PRESET=custom` 호출로 실행한다. 각 case마다 `CUDA_VISIBLE_DEVICES`를 직접 지정하므로, 전체 실행 앞에 전역 `CUDA_VISIBLE_DEVICES`를 줄 필요가 없다.

실행 전에 실제 계산 없이 custom 호출 목록만 확인하고 싶으면:

```bash
cd PaScaL_TDMAcuda/Study
DRY_RUN=1 ./run_full_study.sh
```

결과 파일:

```text
tdma_total_profile_*.csv
tdma_correctness_*.csv
tdma_full_case_list_*.csv
tdma_full_study_*.log
tdma_full_study_*_case_files/tdma_environment_*.txt
tdma_full_study_*_case_files/tdma_case_manifest_*.csv
```
