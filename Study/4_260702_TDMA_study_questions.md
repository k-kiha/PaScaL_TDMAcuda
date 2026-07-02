# PaScaL_TDMAcuda Study 설계 리포트

작성일: 2026-07-02  
상태: 결과 수집 전 초안  
대상 코드:

- `Fortran_Original`: 원본 CUDA Fortran + MPI TDMA 구현
- `CUDA_CXX_Port`: CUDA C++ + MPI 포팅 구현
- `Study`: 두 구현을 같은 조건에서 비교하기 위한 correctness/performance 실험 공간

## 1. 핵심 결론

이 Study의 중심 질문은 다음입니다.

> CUDA Fortran으로 작성된 원본 PaScaL_TDMAcuda의 전역 TDMA 알고리즘과 MPI 통신 구조를 CUDA C++ 포팅 구현이 올바르게 보존했는가, 그리고 H200 GPU 환경에서 계산/통신/packing 단계별 성능 특성이 어떻게 나타나는가?

따라서 이 Study는 단순히 “C++ 코드가 돌아간다”를 보이는 것이 아니라, 다음 세 가지를 함께 보여야 합니다.

1. 원본 Fortran과 CUDA C++ 포트가 같은 수치 문제에서 같은 해를 만든다.
2. 전체 시간뿐 아니라 계산, MPI 통신, packing/unpacking 시간이 어떻게 나뉘는지 설명할 수 있다.
3. GPU 수와 문제 크기가 바뀔 때 병렬 TDMA의 성능 병목이 어디로 이동하는지 해석할 수 있다.

이 방향은 NVIDIA DevTech/HPC 지원 목적에도 맞습니다. 포트폴리오 관점에서 중요한 증거는 “CUDA C++를 썼다”가 아니라, 기존 HPC 코드를 이해하고, 알고리즘 의미를 보존하며, GPU/MPI 병목을 계측 가능한 형태로 재구성했다는 점입니다.

## 2. Study의 중심 질문

### 2.1 Correctness 질문

첫 번째 질문은 다음입니다.

> CUDA C++ 포트가 원본 CUDA Fortran 구현과 같은 TDMA 해를 만드는가?

현재 Study 문제는 global z 방향의 단순 second-difference tridiagonal system입니다. 모든 independent system이 같은 계수와 RHS 구조를 가지며, global 양 끝단에 boundary forcing이 들어갑니다. 이 문제의 기대 해는 모든 위치에서 `1.0`입니다.

따라서 correctness는 다음 기준으로 먼저 확인합니다.

- Fortran과 CUDA C++ 각각의 `max_abs_error_to_expected`가 충분히 작다.
- 같은 case에서 `solution_sum`, `solution_l2`, `solution_linf`가 Fortran/CUDA C++ 사이에 일치한다.
- `sample_z0`, `sample_zmid`, `sample_zlast`가 기대값 `1.0`과 일치한다.

현재 correctness 비교는 전체 solution dump가 아니라 solution signature 방식입니다. 포트폴리오 초안 단계에서는 이 방식이 적절합니다. 결과가 이상하면 다음 단계로 전체 field difference 또는 residual check를 추가합니다.

### 2.2 전체 성능 질문

두 번째 질문은 다음입니다.

> 같은 문제 크기와 같은 MPI rank 수에서 CUDA C++ 포트의 total solve time은 원본 Fortran과 비교해 어느 정도인가?

여기서 total time은 각 rank 중 가장 느린 rank 기준인 `total_s_max`를 우선 봅니다. 분산 MPI solver의 실제 wall-clock 시간은 평균 rank 시간이 아니라 가장 느린 rank에 의해 결정되기 때문입니다.

`total_s_avg`는 load balance 또는 rank별 편차를 보조적으로 보는 용도입니다. `total_s_max`와 `total_s_avg`의 차이가 크면 특정 rank 또는 특정 GPU/통신 경로가 병목일 수 있습니다.

### 2.3 Phase breakdown 질문

세 번째 질문은 다음입니다.

> TDMA solve 시간 중 계산, MPI 통신, packing/unpacking의 비중은 어떻게 나뉘는가?

현재 Study는 다음 phase를 기록합니다.

- `local_compute_s_max`: 각 rank의 local modified TDMA 또는 단일 rank TDMA 계산
- `pack_forward_s_max`: reduced coefficient를 통신 버퍼로 packing
- `mpi_forward_s_max`: reduced coefficient forward all-to-all 통신
- `unpack_forward_s_max`: global reduced system 조립을 위한 unpacking
- `reduced_compute_s_max`: transformed/reduced TDMA solve
- `pack_backward_s_max`: reduced solution packing
- `mpi_backward_s_max`: reduced solution backward all-to-all 통신
- `unpack_backward_s_max`: reduced solution unpacking
- `update_compute_s_max`: local full solution update

그리고 분석 편의를 위해 다음 aggregate도 봅니다.

- `compute_s_max = local_compute + reduced_compute + update_compute`
- `communication_s_max = mpi_forward + mpi_backward`
- `packing_s_max = pack_forward + unpack_forward + pack_backward + unpack_backward`

이 breakdown이 있어야 “C++가 빠르다/느리다”에서 끝나지 않고, 어떤 단계가 차이를 만드는지 설명할 수 있습니다.

### 2.4 Scaling 질문

네 번째 질문은 다음입니다.

> GPU/MPI rank 수를 늘릴 때 total time과 phase 비중은 어떻게 변하는가?

현재 우선 실험 rank 수는 다음을 기본으로 둡니다.

```text
NP_LIST="1 2 4"
```

해석 기준은 다음입니다.

- `np=1`: MPI all-to-all이 없는 local TDMA 기준 성능
- `np=2,4`: reduced system 통신과 update가 포함된 distributed TDMA 성능
- `np` 증가 시 `local_compute_s_max`는 줄어들 가능성이 있지만, `communication_s_max`와 `packing_s_max`는 증가하거나 상대 비중이 커질 수 있음
- `np` 증가로 total time이 줄지 않거나 오히려 증가하면, 통신/packing overhead가 local compute 절감보다 크다는 뜻일 수 있음

### 2.5 Problem-size 질문

다섯 번째 질문은 다음입니다.

> `n1,n2,n3`가 바뀔 때 계산과 통신의 상대적 병목은 어떻게 변하는가?

현재 우선 실험 size는 다음입니다.

```text
SIZE_LIST="64,64,2048 128,128,2048 128,128,4096"
```

해석 관점은 다음입니다.

- `n1*n2 = nsys`: independent TDMA system 수
- `n3`: 각 TDMA line의 global 길이
- `n1,n2` 증가: system-level parallelism과 전체 작업량 증가
- `n3` 증가: local line 길이와 reduced system 관련 작업 증가

같은 `n3`에서 `n1,n2`가 커질 때 compute 비중이 커지는지, 같은 `n1,n2`에서 `n3`가 커질 때 local compute와 communication 비중이 어떻게 바뀌는지를 보는 것이 중요합니다.

### 2.6 CUDA-aware MPI 질문

여섯 번째 질문은 다음입니다.

> CUDA-aware MPI device-buffer path가 Study의 기본 경로로 정상 동작하며, 필요한 경우 host-staging fallback과 비교 가능한가?

현재 CUDA C++ Study는 `MPI_MODE=device`가 기본입니다. 이는 device buffer를 MPI 통신에 직접 넘기는 경로입니다. 이 경로가 목표 성능 경로입니다.

`MPI_MODE=host`는 fallback입니다. device mode가 실패하거나 서버 MPI 설정상 CUDA-aware path가 불안정할 때만 비교 대상으로 사용합니다.

결과를 받을 때 `mpi_mode` column을 반드시 확인해야 합니다. Fortran 원본은 CUDA-aware MPI 전제를 갖는 device path로 해석하고, CUDA C++는 CSV에 실제 mode를 기록합니다.

## 3. 데이터 수집 설계

### 3.1 실행 단위

Study 실행은 다음 스크립트를 기준으로 합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NP_LIST="1 2 4" \
SIZE_LIST="64,64,2048 128,128,2048 128,128,4096" \
ITERATIONS=10 \
./run_study_sweep.sh
```

각 case는 다음 key로 구분합니다.

```text
implementation, nranks, n1, n2, n3, nsys, nrow_min, nrow_max, mpi_mode
```

### 3.2 출력 파일

한 번의 sweep은 같은 timestamp를 가진 세 파일을 생성합니다.

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
tdma_correctness_YYMMDD_HHMMSS.csv
tdma_environment_YYMMDD_HHMMSS.txt
```

각 파일의 역할은 다음과 같습니다.

- `tdma_total_profile_*.csv`: 모든 iteration의 timing raw data
- `tdma_correctness_*.csv`: 각 case의 첫 solve 결과 signature
- `tdma_environment_*.txt`: GPU, CUDA, MPI, git revision, 실행 조건 기록

이 세 파일은 항상 한 묶음으로 보관해야 합니다. 성능 수치만 있고 서버 환경 정보가 없으면 포트폴리오 문서에서 재현성이 약해집니다.

### 3.3 Iteration 해석

각 case는 `ITERATIONS=10`으로 실행합니다.

현재 해석 규칙은 다음입니다.

- `iter=0`: 첫 solve 결과. correctness signature와 warm-up 관찰에 사용
- `iter=1..9`: 안정화된 timing 분석에 우선 사용

첫 실행에서 통신 시간이 비정상적으로 크게 나오는 현상이 관찰될 수 있습니다. 따라서 실행 중 평균을 내지 않고 모든 row를 보존합니다. 이후 분석 단계에서 `iter=0` 제외 평균, median, min, max를 계산합니다.

## 4. 결과 수령 후 채울 항목

아래 항목은 서버 결과를 받은 뒤 채웁니다.

### 4.1 실행 환경

결과 파일:

```text
tdma_environment_YYMMDD_HHMMSS.txt
```

기록할 내용:

- GPU 모델:
- GPU 개수:
- driver version:
- CUDA runtime/toolkit version:
- `nvcc --version`:
- MPI implementation:
- compiler wrapper:
- git revision:
- `CUDA_VISIBLE_DEVICES`:
- `NP_LIST`:
- `SIZE_LIST`:
- `ITERATIONS`:
- `MPI_MODE`:

### 4.2 Correctness 결과

결과 파일:

```text
tdma_correctness_YYMMDD_HHMMSS.csv
```

확인할 표:

```text
nranks | n1 | n2 | n3 | Fortran max error | C++ max error | signature match
```

채울 해석:

- 모든 case에서 Fortran과 CUDA C++가 기대 해 `1.0`에 도달했는가?
- `max_abs_error_to_expected`가 구현 간 동일한 수준인가?
- sample 값이 `z=0`, `z=n3/2`, `z=n3-1`에서 일관적인가?
- 차이가 있다면 어느 rank 수 또는 문제 크기에서 발생하는가?

### 4.3 Timing 결과

결과 파일:

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
```

기본 분석 표:

```text
implementation | nranks | n1 | n2 | n3 | total_s_max(iter>=1 mean) | compute_s_max | communication_s_max | packing_s_max
```

채울 해석:

- Fortran 대비 CUDA C++ total time 비율은 얼마인가?
- rank 수 증가에 따라 total time이 줄어드는가?
- communication/packing 비중이 rank 수 증가에 따라 커지는가?
- C++ 포트에서 특정 phase가 Fortran보다 유난히 큰가?

### 4.4 Warm-up 영향

확인할 내용:

- `iter=0`의 `total_s_max`가 `iter=1..9`보다 큰가?
- 큰 경우 어느 phase가 원인인가?
  - `mpi_forward_s_max`
  - `mpi_backward_s_max`
  - packing/unpacking
  - local/reduced compute
- warm-up을 제외한 값이 안정적으로 수렴하는가?

### 4.5 Scaling 해석

확인할 내용:

- `np=1 -> 2 -> 4`에서 total time 변화
- fixed problem size strong scaling 관점의 speedup
- communication 비중 증가 여부
- rank별 local row 범위 차이: `nrow_min`, `nrow_max`

### 4.6 Problem-size 해석

확인할 내용:

- `64,64,2048` 대비 `128,128,2048`:
  - `nsys` 증가에 따른 compute 증가와 GPU utilization 개선 가능성
- `128,128,2048` 대비 `128,128,4096`:
  - line length 증가에 따른 local compute와 communication 구조 변화

## 5. 결과를 해석할 때 주의할 점

### 5.1 이 Study는 최종 최적화 경쟁이 아니다

현재 목적은 CUDA C++ 포트가 원본 알고리즘을 보존했고, 성능 병목을 분석 가능한 형태로 드러낸다는 것을 보여주는 것입니다. 따라서 첫 보고서에서는 “절대 최적 성능”보다 다음을 우선합니다.

- 동일 문제에서의 correctness
- 같은 phase schema를 가진 fair comparison
- 반복 실행 raw data 보존
- 서버 환경과 실행 조건 기록

### 5.2 Fortran과 CUDA C++의 완전한 일대일 성능 비교는 조심해야 한다

Fortran CUDA와 CUDA C++는 compiler, kernel launch, device memory management, MPI buffer handling 방식이 다를 수 있습니다. 따라서 결과가 다르면 단순히 언어 차이라고 결론내리면 안 됩니다.

먼저 phase breakdown으로 차이를 분리해야 합니다.

- compute 차이인지
- MPI communication 차이인지
- pack/unpack 차이인지
- 첫 실행 warm-up 차이인지
- CUDA-aware MPI path 차이인지

### 5.3 `total_s_max`를 우선한다

분산 solver의 실제 진행 시간은 가장 느린 rank에 의해 결정됩니다. 평균값은 보조 지표로 사용하고, 주 해석은 `total_s_max`와 phase별 `*_s_max`를 기준으로 합니다.

### 5.4 결과가 이상할 때의 우선 점검 순서

1. `tdma_environment_*.txt`에서 실제 GPU, CUDA, MPI, git revision 확인
2. `mpi_mode`가 의도한 값인지 확인
3. correctness CSV에서 error가 먼저 깨졌는지 확인
4. `iter=0`만 큰지, `iter>=1`도 계속 큰지 확인
5. phase breakdown에서 MPI, packing, compute 중 어느 쪽이 커졌는지 확인
6. 같은 case를 한 번 더 실행해 재현성 확인

## 6. 예상되는 보고서 메시지

결과가 정상적으로 나오면 최종 리포트의 핵심 메시지는 다음 형태가 됩니다.

> CUDA C++ 포트는 원본 CUDA Fortran PaScaL_TDMAcuda의 distributed TDMA solve flow를 보존하며, 동일한 simple second-difference TDMA 문제에서 같은 solution signature를 생성했다. H200 환경에서 수집한 phase timing은 total solve time을 local compute, reduced solve, MPI all-to-all, pack/unpack, update 단계로 분해해 보여주며, rank 수와 problem size 변화에 따른 병목 이동을 해석할 수 있게 한다.

결과가 아직 기대만큼 좋지 않아도 리포트 가치는 있습니다. 그 경우 메시지는 다음으로 바뀝니다.

> 초기 CUDA C++ 포트는 correctness는 만족하지만, 특정 phase에서 Fortran 대비 overhead가 관찰되었다. 이 overhead는 total time 하나가 아니라 phase timing으로 분리되어 확인되었으며, 다음 최적화 대상은 해당 phase로 좁혀졌다.

즉, 결과가 좋든 나쁘든 Study의 목표는 “분석 가능한 증거”를 만드는 것입니다.

## 7. 다음 작업

서버 결과를 받은 뒤 다음 순서로 이 문서를 업데이트합니다.

1. `tdma_environment_*.txt` 내용을 4.1에 요약한다.
2. `tdma_correctness_*.csv`를 바탕으로 4.2 correctness 표를 채운다.
3. `tdma_total_profile_*.csv`에서 `iter>=1` 기준 통계를 계산한다.
4. rank scaling, problem-size scaling, phase breakdown 해석을 추가한다.
5. 결과 해석이 맞는지 확인한 뒤 영어 버전으로 번역한다.
