# PaScaL_TDMAcuda Study 설계 리포트

작성일: 2026-07-02  
상태: 새 Study case matrix 기준 결과 수집 전 계획서
대상 코드:

- `Fortran_Original`: 원본 CUDA Fortran + MPI TDMA 구현
- `CUDA_CXX_Port`: CUDA C++ + MPI 포팅 구현
- `Study`: 두 구현을 같은 조건에서 비교하기 위한 correctness/performance 실험 공간

## 1. 핵심 목적

이 Study의 목적은 단순히 CUDA C++ 포트가 실행된다는 것을 보이는 것이 아닙니다. 목표는 다음 질문에 답할 수 있는 데이터와 해석 구조를 만드는 것입니다.

> CUDA Fortran 원본 PaScaL_TDMAcuda의 distributed TDMA 알고리즘과 MPI 통신 구조가 CUDA C++ 포트에서도 올바르게 보존되었는가? 그리고 H200 GPU 환경에서 계산, 통신, packing, scaling 특성은 어떻게 나타나는가?

따라서 Study는 다음 여섯 축으로 구성합니다.

1. Correctness
2. Compute vs Communication Breakdown
3. Scaling
   - Strong Scaling
   - Weak Scaling
   - `np=2` baseline
4. `nsys = n1*n2` 영향과 `nrow = n3/rank` 영향 분리
5. MPI Mode 비교
6. Reproducibility

이 구조가 중요한 이유는, 포트폴리오 관점에서 “C++로 옮겼다”보다 “기존 HPC 코드를 이해하고, 알고리즘 의미를 보존했으며, 병목을 계측 가능한 형태로 설명할 수 있다”가 더 강한 증거이기 때문입니다.

## 2. Study 질문

### 2.1 Correctness

질문:

> CUDA C++ 포트가 원본 CUDA Fortran 구현과 같은 TDMA 해를 만드는가?

현재 Study 문제는 global z 방향의 simple second-difference tridiagonal system입니다. 모든 independent system이 같은 계수와 RHS 구조를 가지며, global 양 끝단에 boundary forcing이 들어갑니다. 기대 해는 모든 위치에서 `1.0`입니다.

확인 지표:

- `max_abs_error_to_expected`
- `solution_sum`
- `solution_l2`
- `solution_linf`
- `sample_z0`
- `sample_zmid`
- `sample_zlast`

기본 판정:

- Fortran과 CUDA C++ 각각의 `max_abs_error_to_expected`가 충분히 작아야 합니다.
- 같은 case에서 Fortran/CUDA C++의 solution signature가 같은 수준이어야 합니다.
- `sample_z0`, `sample_zmid`, `sample_zlast`가 기대값 `1.0`과 일치해야 합니다.

현재 correctness 비교는 전체 field dump가 아니라 solution signature 방식입니다. 초기 포트폴리오 Study에는 이 방식이 적절합니다. 만약 signature가 어긋나면 다음 단계에서 full-field difference 또는 residual check를 추가합니다.

### 2.2 Compute vs Communication Breakdown

질문:

> TDMA solve 시간 중 계산, MPI 통신, packing/unpacking의 비중은 어떻게 나뉘는가?

기본 total 지표:

- `total_s_max`: rank 중 가장 느린 rank 기준 시간
- `total_s_avg`: rank 평균 시간

분산 MPI solver의 실제 wall-clock time은 가장 느린 rank에 의해 결정되므로, 주요 해석은 `total_s_max`를 기준으로 합니다. `total_s_avg`는 rank imbalance나 편차를 확인하는 보조 지표입니다.

phase 지표:

- `local_compute_s_max`: local modified TDMA 또는 single-rank TDMA 계산
- `pack_forward_s_max`: reduced coefficient forward packing
- `mpi_forward_s_max`: reduced coefficient forward all-to-all
- `unpack_forward_s_max`: transformed/reduced system assembly unpacking
- `reduced_compute_s_max`: transformed/reduced TDMA solve
- `pack_backward_s_max`: reduced solution packing
- `mpi_backward_s_max`: reduced solution backward all-to-all
- `unpack_backward_s_max`: reduced solution unpacking
- `update_compute_s_max`: local full solution update

aggregate 지표:

```text
compute_s_max       = local_compute + reduced_compute + update_compute
communication_s_max = mpi_forward + mpi_backward
packing_s_max       = pack_forward + unpack_forward + pack_backward + unpack_backward
```

이 breakdown이 있어야 Fortran 대비 CUDA C++의 차이를 다음처럼 분리해서 말할 수 있습니다.

- 순수 계산 차이인지
- MPI communication 차이인지
- device buffer packing/unpacking 차이인지
- 첫 iteration warm-up 차이인지
- CUDA-aware MPI device path와 host fallback 차이인지

### 2.3 Scaling

질문:

> GPU/MPI rank 수를 늘릴 때 total time, phase 비중, throughput, efficiency는 어떻게 변하는가?

중요한 결정:

- distributed scaling baseline은 `np=2`입니다.
- `np=1`은 single-GPU local TDMA reference로 남기지만, distributed TDMA scaling의 주 baseline으로 쓰지 않습니다.
- scaling 중심 rank 수는 `np=2,4,8`입니다.

기본 rank 설정:

```text
BASELINE_NP=2
SCALING_NP_LIST="2 4 8"
```

일반적인 strong scaling 공식은 `T1/Tp`이지만, 이 Study의 핵심 비교는 2 GPU baseline 기준입니다. 따라서 분석 표에는 다음 2 GPU baseline metric을 우선 둡니다.

```text
T_base = T_2
p_rel = p / 2

speedup_2base(p)    = T_2 / T_p
efficiency_2base(p) = T_2 / ((p / 2) * T_p)
throughput(p)       = (n1 * n2 * n3) / total_s_max
```

보조 지표로 `np=1` reference가 있는 case에서는 다음도 계산할 수 있습니다.

```text
speedup_1gpu_ref(p)    = T_1 / T_p
efficiency_1gpu_ref(p) = T_1 / (p * T_p)
```

단, 최종 리포트의 scaling 해석은 `np=2` baseline 기준을 우선합니다.

### 2.4 Strong Scaling

질문:

> global problem size를 고정하고 GPU/rank 수를 늘릴 때 solve time이 얼마나 줄어드는가?

Strong scaling case는 fixed global size로 실행합니다.

```text
study_suite=strong_scaling
np = 2, 4, 8
sizes:
  128,128,4096
  256,256,4096
```

해석:

- 같은 `n1,n2,n3`에서 `np=2 -> 4 -> 8`로 갈 때 `total_s_max`가 얼마나 줄어드는지 봅니다.
- `compute_s_max`는 줄어드는 방향이 자연스럽습니다.
- `communication_s_max`와 `packing_s_max`는 rank 수 증가와 함께 상대 비중이 커질 수 있습니다.
- `np=8`에서 speedup이 낮으면 reduced system communication, all-to-all latency, packing overhead가 local compute 감소를 상쇄했는지 확인합니다.

### 2.5 Weak Scaling

질문:

> local work를 유지하면서 GPU/rank 수와 global problem size를 함께 키울 때 성능이 유지되는가?

TDMA에서는 weak scaling을 하나로만 보면 해석이 흐려집니다. `nsys = n1*n2`와 `nrow = n3/rank`가 서로 다른 의미를 갖기 때문입니다. 따라서 weak scaling을 두 방향으로 나눕니다.

#### 2.5.1 Weak Scaling A: `nrow` 경로

목적:

> `nsys`를 고정하고 `n3`를 rank 수에 비례시켜 local `nrow`를 유지할 때, global line length 증가가 어떤 영향을 주는지 본다.

case:

```text
study_suite=weak_nrow_scaling
np=2: 128,128,2048   # local nrow = 1024
np=4: 128,128,4096   # local nrow = 1024
np=8: 128,128,8192   # local nrow = 1024
```

해석:

- `nsys`는 일정합니다.
- 각 rank의 local row length는 유지됩니다.
- global z-line이 길어지고 reduced distributed TDMA 구조가 커집니다.
- rank 수 증가에 따른 communication/reduced-system overhead를 보기 좋습니다.

#### 2.5.2 Weak Scaling B: `nsys` 경로

목적:

> `n3`를 고정하고 `nsys=n1*n2`를 rank 수에 비례시켜 local work를 유지할 때, independent system 수 증가가 어떤 영향을 주는지 본다.

case:

```text
study_suite=weak_nsys_scaling
np=2: 128,128,2048
np=4: 128,256,2048
np=8: 128,512,2048
```

해석:

- `n3`는 일정합니다.
- `nsys`가 rank 수에 맞춰 증가합니다.
- 각 rank가 처리하는 `nsys*nrow` work가 대략 유지됩니다.
- system-level parallelism이 커질 때 GPU occupancy와 memory access 특성이 어떻게 바뀌는지 볼 수 있습니다.

### 2.6 `nsys` Sensitivity

질문:

> `n3`를 고정하고 `n1*n2`만 키우면 compute/communication 비중이 어떻게 바뀌는가?

case:

```text
study_suite=nsys_sensitivity
np = 2, 8
sizes:
  64,64,4096
  128,128,4096
  128,256,4096
```

해석:

- `n3`가 같으므로 TDMA line length 조건은 유지됩니다.
- `nsys`가 증가하면 independent TDMA system 수가 증가합니다.
- GPU 입장에서는 병렬 작업 수가 늘어나는 효과가 있습니다.
- compute 비중, packing volume, throughput 변화가 주요 관찰 대상입니다.

### 2.7 `nrow` Sensitivity

질문:

> `n1*n2`를 고정하고 `n3`만 키우면 local row length와 distributed TDMA 통신 구조가 어떻게 영향을 받는가?

case:

```text
study_suite=nrow_sensitivity
np = 2, 8
sizes:
  128,128,2048
  128,128,4096
  128,128,8192
```

해석:

- `nsys`가 같으므로 independent system 수는 유지됩니다.
- `n3` 증가로 local row length와 global line length가 증가합니다.
- local compute와 update cost가 증가할 가능성이 큽니다.
- reduced system 및 communication phase가 어느 정도 증가하는지 확인합니다.

### 2.8 MPI Mode 비교

질문:

> CUDA C++ 포트에서 CUDA-aware MPI device-buffer path와 host-staging fallback은 어떤 차이를 보이는가?

case:

```text
study_suite=mpi_mode_compare
np = 2, 4, 8
size = 128,128,4096
modes = device, host
```

해석:

- `device`: MPI가 device buffer를 직접 처리하는 목표 경로
- `host`: host staging fallback

Fortran 원본은 CUDA-aware MPI device path 전제를 갖는 구현으로 봅니다. MPI mode 비교는 CUDA C++ 포트에서만 수행합니다.

확인할 것:

- `device`가 `host`보다 빠른가?
- 차이가 주로 `mpi_forward_s_max`/`mpi_backward_s_max`에 나타나는가?
- host fallback이 correctness는 유지하는가?
- 서버 MPI 설정에서 device path가 안정적으로 동작하는가?

### 2.9 Reproducibility

질문:

> 나중에 같은 결과를 재현하거나 설명할 수 있도록 환경과 case 의도가 충분히 기록되었는가?

필수 산출물:

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
tdma_correctness_YYMMDD_HHMMSS.csv
tdma_environment_YYMMDD_HHMMSS.txt
tdma_case_manifest_YYMMDD_HHMMSS.csv
```

`tdma_environment_*.txt`에는 다음이 들어가야 합니다.

- 실행 서버
- GPU 상태
- CUDA/NVHPC/MPI version
- git revision
- 실행 preset과 주요 환경변수
- `CUDA_VISIBLE_DEVICES`
- 출력 파일명

`tdma_case_manifest_*.csv`에는 다음이 들어가야 합니다.

- 각 case의 `study_suite`
- `case_id`
- `nranks,n1,n2,n3`
- baseline rank
- scaling kind
- C++ MPI mode list
- 왜 이 case가 필요한지에 대한 notes

case manifest가 있어야 결과 분석 때 strong scaling, weak scaling, `nsys` sensitivity, `nrow` sensitivity, MPI mode comparison을 자동으로 분리할 수 있습니다.

## 3. 실행 계획

### 3.1 기본 portfolio 실행

서버에서 먼저 dry-run으로 case manifest와 실행 명령을 확인합니다.

```bash
cd PaScaL_TDMAcuda/Study
DRY_RUN=1 ./run_study_sweep.sh
```

case 수와 예상 실행 시간이 괜찮으면 실제 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_study_sweep.sh
```

기본 preset은 다음입니다.

```text
STUDY_PRESET=portfolio
BASELINE_NP=2
SCALING_NP_LIST="2 4 8"
ITERATIONS=10
CXX_DEFAULT_MPI_MODES="device"
MPI_MODE_LIST="device host"
RUN_FORTRAN=1
RUN_CXX=1
```

### 3.2 Quick 실행

긴 실행 전에 script와 output schema만 확인하려면 quick preset을 사용합니다.

```bash
STUDY_PRESET=quick ./run_study_sweep.sh
```

### 3.3 Custom 실행

특정 case만 다시 돌리고 싶으면 custom preset을 사용합니다.

```bash
STUDY_PRESET=custom \
NP_LIST="2 4 8" \
SIZE_LIST="128,128,4096" \
./run_study_sweep.sh
```

### 3.4 Portfolio case matrix

현재 `portfolio` preset은 다음 suite를 생성합니다.

```text
single_gpu_reference: 2 manifest cases
strong_scaling:       6 manifest cases
weak_nrow_scaling:    3 manifest cases
weak_nsys_scaling:    3 manifest cases
nsys_sensitivity:     6 manifest cases
nrow_sensitivity:     6 manifest cases
mpi_mode_compare:     3 manifest cases
```

중복되는 실제 execution case는 `run_study_sweep.sh`에서 `sort -u`로 한 번만 실행합니다. 예를 들어 `128,128,4096`의 `np=2,4,8` device case는 strong scaling, sensitivity, MPI mode compare에서 반복 등장할 수 있지만 실제 실행은 한 번으로 deduplicate됩니다. 다만 manifest에는 이 case가 여러 Study 질문에 연결된다는 정보가 남습니다.

## 4. 출력 파일과 schema

한 번의 sweep은 같은 timestamp를 가진 네 파일을 생성합니다.

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
tdma_correctness_YYMMDD_HHMMSS.csv
tdma_environment_YYMMDD_HHMMSS.txt
tdma_case_manifest_YYMMDD_HHMMSS.csv
```

### 4.1 Timing CSV

파일:

```text
tdma_total_profile_*.csv
```

역할:

- 모든 iteration의 timing raw data 저장
- Fortran/CUDA C++ 비교
- phase breakdown
- scaling metric 계산

핵심 columns:

```text
solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg,local_compute_s_max,pack_forward_s_max,mpi_forward_s_max,unpack_forward_s_max,reduced_compute_s_max,pack_backward_s_max,mpi_backward_s_max,unpack_backward_s_max,update_compute_s_max,compute_s_max,communication_s_max,packing_s_max
```

### 4.2 Correctness CSV

파일:

```text
tdma_correctness_*.csv
```

역할:

- 각 execution case의 `iter=0` solution signature 저장
- Fortran/CUDA C++ correctness 비교
- MPI mode별 correctness 확인

핵심 columns:

```text
solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,mpi_mode,solution_sum,solution_l2,solution_linf,sample_z0,sample_zmid,sample_zlast,expected_value,max_abs_error_to_expected
```

### 4.3 Environment TXT

파일:

```text
tdma_environment_*.txt
```

역할:

- 서버 환경 기록
- GPU, CUDA, MPI, compiler, git revision 기록
- 실행 preset과 output path 기록

### 4.4 Case Manifest CSV

파일:

```text
tdma_case_manifest_*.csv
```

역할:

- 각 case가 어떤 Study 질문에 속하는지 기록
- 분석 script가 suite별로 결과를 나눌 수 있게 함

columns:

```text
study_suite,case_id,nranks,n1,n2,n3,baseline_nranks,scaling_kind,cxx_mpi_modes,notes
```

## 5. 분석 규칙

### 5.1 Iteration 처리

각 case는 `ITERATIONS=10`으로 실행합니다.

기본 해석:

- `iter=0`: correctness와 first-run warm-up 관찰
- `iter=1..9`: 안정화 timing 분석

첫 실행에서 MPI communication 시간이 크게 튈 수 있으므로, timing 분석은 raw row를 모두 보존한 뒤 `iter>=1` 기준 mean, median, min, max를 계산합니다.

### 5.2 Correctness 분석

분석 표:

```text
study_suite | implementation | mpi_mode | nranks | n1 | n2 | n3 | max_abs_error_to_expected | signature_match
```

판정:

- `max_abs_error_to_expected`가 충분히 작으면 pass
- 같은 case에서 Fortran/CUDA C++의 signature가 같은 수준이면 pass
- `host` mode도 `device` mode와 같은 correctness를 보여야 함

### 5.3 Performance 분석

기본 표:

```text
study_suite | implementation | mpi_mode | nranks | n1 | n2 | n3 | total_s_max_mean_iter_ge_1 | total_s_avg_mean_iter_ge_1 | compute_s_max | communication_s_max | packing_s_max
```

Fortran vs CUDA C++ 비교:

```text
cxx_over_fortran_total = total_s_max_cxx / total_s_max_fortran
```

MPI mode 비교:

```text
host_over_device_total = total_s_max_host / total_s_max_device
```

### 5.4 Scaling 분석

Strong scaling:

```text
same n1,n2,n3
compare np=2,4,8
baseline = np=2
```

Weak `nrow` scaling:

```text
same n1,n2
n3 proportional to rank count
local nrow approximately constant
```

Weak `nsys` scaling:

```text
same n3
n1*n2 proportional to rank count
local work approximately constant
```

Metric:

```text
speedup_2base(p)    = T_2 / T_p
efficiency_2base(p) = T_2 / ((p / 2) * T_p)
throughput(p)       = (n1 * n2 * n3) / total_s_max
```

보조 metric:

```text
speedup_1gpu_ref(p)    = T_1 / T_p
efficiency_1gpu_ref(p) = T_1 / (p * T_p)
```

### 5.5 `nsys` vs `nrow` 해석

TDMA에서 `n1,n2,n3`는 같은 의미의 size가 아닙니다.

- `nsys = n1*n2`: independent TDMA line 개수
- `nrow = n3/rank`: rank당 TDMA line 길이

따라서 다음 두 질문을 분리해서 해석해야 합니다.

1. `nsys`가 커지면 GPU가 더 많은 independent systems를 처리하므로 parallelism과 occupancy가 좋아지는가?
2. `nrow`가 커지면 각 TDMA line의 serial work와 update cost가 어떻게 증가하는가?

이 차이가 이 Study에서 가장 좋은 이야기거리 중 하나입니다. 단순히 “size가 커졌다”가 아니라, TDMA 알고리즘에서 어떤 차원의 증가가 어떤 phase를 키우는지 설명할 수 있기 때문입니다.

## 6. 결과 수령 후 채울 항목

### 6.1 실행 환경

결과 파일:

```text
tdma_environment_YYMMDD_HHMMSS.txt
```

채울 내용:

- GPU 모델:
- GPU 개수:
- driver version:
- CUDA runtime/toolkit version:
- `nvcc --version`:
- MPI implementation:
- compiler wrapper:
- git revision:
- `CUDA_VISIBLE_DEVICES`:
- `STUDY_PRESET`:
- `BASELINE_NP`:
- `SCALING_NP_LIST`:
- `ITERATIONS`:
- `MPI_MODE_LIST`:

### 6.2 Correctness 결과

결과 파일:

```text
tdma_correctness_YYMMDD_HHMMSS.csv
```

채울 표:

```text
suite | nranks | n1 | n2 | n3 | mode | Fortran max error | C++ max error | pass/fail
```

채울 해석:

- 모든 case에서 기대 해 `1.0`에 도달했는가?
- Fortran과 CUDA C++ signature가 일치하는가?
- `device`와 `host` mode 사이 correctness 차이는 없는가?

### 6.3 Compute vs Communication 결과

결과 파일:

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
```

채울 표:

```text
suite | implementation | mode | nranks | n1 | n2 | n3 | total | compute % | communication % | packing %
```

채울 해석:

- rank 수 증가에 따라 communication/packing 비중이 커지는가?
- Fortran과 CUDA C++의 phase profile이 같은가?
- 특정 phase에서 C++ overhead가 보이는가?

### 6.4 Strong Scaling 결과

채울 표:

```text
n1 | n2 | n3 | implementation | mode | np | total | speedup_2base | efficiency_2base | throughput
```

채울 해석:

- fixed global size에서 `np=2 -> 4 -> 8`로 갈 때 speedup은 어떤가?
- efficiency가 떨어진다면 communication/packing phase 증가와 연결되는가?
- Fortran과 CUDA C++ scaling curve가 같은 모양인가?

### 6.5 Weak Scaling 결과

Weak `nrow` scaling 표:

```text
np | n1 | n2 | n3 | local_nrow | total | efficiency_2base | throughput
```

Weak `nsys` scaling 표:

```text
np | n1 | n2 | n3 | nsys | total | efficiency_2base | throughput
```

채울 해석:

- `nrow` 방향 weak scaling에서 communication/reduced-system overhead가 커지는가?
- `nsys` 방향 weak scaling에서 GPU parallelism 증가가 성능 유지에 도움이 되는가?
- 두 weak scaling path가 서로 다른 phase profile을 보이는가?

### 6.6 MPI Mode 결과

채울 표:

```text
np | n1 | n2 | n3 | total_device | total_host | host/device | mpi_forward_device | mpi_forward_host | mpi_backward_device | mpi_backward_host
```

채울 해석:

- device-buffer path가 host fallback보다 빠른가?
- 차이가 forward/backward MPI 중 어디에서 나는가?
- host fallback은 correctness를 유지하는가?

### 6.7 Reproducibility 확인

확인할 것:

- timing CSV, correctness CSV, environment txt, case manifest가 같은 timestamp로 존재하는가?
- environment txt에 git revision이 있는가?
- case manifest가 result CSV와 join 가능한가?
- 실행 당시 working tree 상태가 기록되어 있는가?

## 7. 결과 해석 시 주의할 점

### 7.1 `np=1`과 `np=2`를 섞어서 말하지 않는다

`np=1`은 local TDMA reference입니다. distributed TDMA의 all-to-all, reduced system, update 흐름이 본격적으로 나타나는 기준은 `np>=2`입니다. 따라서 scaling baseline은 `np=2`로 둡니다.

### 7.2 `nsys` 증가와 `nrow` 증가를 구분한다

`n1,n2,n3`를 모두 단순 problem size로 묶으면 해석이 약해집니다. TDMA에서는 `nsys` 증가와 `nrow` 증가가 서로 다른 병목을 만듭니다.

### 7.3 `iter=0`은 timing 평균에 섞지 않는다

첫 iteration은 CUDA/MPI warm-up, communicator/device-buffer path 초기화, cache 효과가 섞일 수 있습니다. 특히 MPI communication이 크게 튈 수 있으므로, 안정화 timing은 `iter>=1`로 봅니다.

### 7.4 Fortran vs C++ 차이는 phase로 분해해서 말한다

총 시간이 다르더라도 바로 언어 차이라고 말하지 않습니다. 먼저 compute, communication, packing 중 어디에서 차이가 나는지 봐야 합니다.

## 8. 예상되는 최종 메시지

결과가 정상적으로 나오면 최종 리포트의 핵심 메시지는 다음 형태가 됩니다.

> CUDA C++ 포트는 원본 CUDA Fortran PaScaL_TDMAcuda의 distributed TDMA solve flow를 보존하며, 동일한 TDMA 문제에서 같은 solution signature를 생성했다. H200 환경에서 수집한 Study dataset은 total time을 compute, MPI communication, packing/unpacking으로 분해하고, `np=2` baseline의 strong/weak scaling과 `nsys`/`nrow` sensitivity를 분리해 분석할 수 있게 한다.

결과가 일부 기대와 다르더라도 리포트 가치는 유지됩니다.

> Correctness는 만족하지만 특정 phase 또는 MPI mode에서 overhead가 관찰되었다. 이 overhead는 total time 하나가 아니라 phase timing과 case manifest를 통해 특정 scaling path 또는 communication path로 좁혀졌다.

즉, 이 Study의 목표는 단순히 좋은 성능 숫자를 얻는 것이 아니라, CUDA C++ 포팅 결과를 HPC 개발자 관점에서 설명 가능한 증거로 만드는 것입니다.

## 9. 다음 작업

1. 서버에서 `DRY_RUN=1 ./run_study_sweep.sh`로 case manifest를 확인한다.
2. case 수와 예상 실행 시간이 괜찮으면 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_study_sweep.sh`를 실행한다.
3. 결과 파일 4종을 로컬로 가져온다.
4. `tdma_case_manifest_*.csv`와 `tdma_total_profile_*.csv`를 join해 suite별 분석 표를 만든다.
5. 이 문서의 6장 결과 섹션을 채운다.
6. 결과 해석을 확인한 뒤 영어 리포트로 변환한다.
