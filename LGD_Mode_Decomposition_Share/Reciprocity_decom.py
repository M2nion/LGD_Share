import math, os
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import meep as mp
import meep.adjoint as mpa
from meep.materials import Ag
import matplotlib.pyplot as plt

mp.verbosity(0)
resolution = 50      # px/µm

lambda0 = 0.5            # [um]
fcen    = 1.0 / lambda0  # [1/um]
width = 1
fwidth = fcen*width

nfreq   = 1
frequencies = [fcen]

dpml = 0.5
dair = 1.0
hrod = 0.5     

dsub = 1.0
dAg  = 0.25

sx = 2.121                          # +-3차가 +-45도 나옴
sx = 2.828                          # +-4차가 +-45도 나옴
sy = dpml + dair + hrod + dsub + dAg + dpml
cell_size = mp.Vector3(sx, sy)

wrod = sx

substrate = mp.Medium(index=2.0)
air       = mp.Medium(index=1.0)

Medium1 = mp.Medium(index = 2.0)
Medium2 = mp.Medium(index = 1.0)

pml_layers = [
    mp.PML(direction=mp.Y, thickness=dpml, side=mp.High),
    mp.PML(direction=mp.Y, thickness=dpml, side=mp.Low)
]

DESIGN_W  = wrod
DESIGN_H  = hrod
design_c  = mp.Vector3(0, 0.5*sy - dpml - dAg - dsub - 0.5*hrod)
design_sz = mp.Vector3(DESIGN_W, DESIGN_H)


design_res = resolution
NX = int(DESIGN_W * design_res) + 1

DESIGN_MODE = 'free'  # or 'free' or 'grating'

if DESIGN_MODE == 'free':
    NY = int(round(DESIGN_H * design_res)) + 1
else:
    # 해상도와 레이어-패딩 조건을 동시에 만족
    DESIRED_LAYERS = 5       # 원하는 레이어 수 (패딩 1픽셀씩 자동)
    NY_min_layers = 2 * DESIRED_LAYERS - 1        # 레이어 L개 + 패딩 L-1개
    NY_res       = int(round(DESIGN_H * design_res))
    NY = max(NY_res, NY_min_layers)

n_vars = NX * NY

# 최소 피쳐/이진화 설정
MIN_FEATURE = 0.09    # ~80 nm
eta_i = 0.55
beta  = 2
beta_scale = 2
num_beta_steps = 3
filter_radius = mpa.get_conic_radius_from_eta_e(MIN_FEATURE, eta_i)

# x = np.ones((n_vars,), dtype=float) * 0.5
np.random.seed(42)

x = np.random.rand(n_vars) * 0.2 + 0.4
# x = np.ones(n_vars)*0.5

design_vars = mp.MaterialGrid(
    mp.Vector3(NX, NY, 0),
    Medium1, Medium2,
    grid_type="U_MEAN",
)
design_region = mpa.DesignRegion(design_vars, volume=mp.Volume(center=design_c, size=design_sz))
design_vars.update_weights(x.reshape(NX, NY))

# ----------------------------
# geometry (디자인 포함)
# ----------------------------
geometry = [
    mp.Block(
        material=Ag,
        center=mp.Vector3(0, 0.5*sy - dpml - 0.5*dAg),
        size=mp.Vector3(mp.inf, dAg, mp.inf),
    ),
    # Substrate bulk (Ag 아래에 위치)
    mp.Block(
        material=substrate,
        center=mp.Vector3(0, 0.5*sy - dpml - dAg - 0.5*dsub),
        size=mp.Vector3(mp.inf, dsub, mp.inf),
    ),
    # Design block (텍스처 영역)
    mp.Block(
        material=design_vars,
        center=design_region.center,
        size=design_region.size,
    ),
]

bottom_pml_top_y = -0.5 * sy + dpml
src_center = mp.Vector3(0, bottom_pml_top_y + (3 / resolution))
src_size   = mp.Vector3(sx, 0, 0)

sources = [
    mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        component=mp.Ez,  # TM 편광의 경우
        center=mp.Vector3(0, src_center.y),
        size=mp.Vector3(sx, 0)
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    k_point = mp.Vector3(0,0,0),
    extra_materials=[Ag],
    default_material=mp.Medium(index=1),
)

# --- FourierFields 모니터 (y=0.7 선) ---
ff_center = mp.Vector3(0, 0.7, 0)
ff_size   = mp.Vector3(sx, 0, 0)
ff_vol    = mp.Volume(center=ff_center, size=ff_size)

ff_Ez = mpa.FourierFields(sim, volume=ff_vol, component=mp.Ez)
# -----------------------------------------
# 1) DFT 모니터 설치 (ff_Ez와 동일한 위치/길이)
# -----------------------------------------
dft_line = sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=ff_center, size=ff_size)

margin = 0.4  # [um] 디자인 하단 바로 아래로 살짝 내림

# 디자인 하단 y좌표
design_bottom_y = design_region.center.y - 0.5 * design_region.size.y

# 추가 모니터의 중심과 크기 (x는 주기 전체 길이)
ff_center_below = mp.Vector3(0, design_bottom_y - margin, 0)
ff_size_below   = mp.Vector3(sx, 0, 0)
ff_vol_below    = mp.Volume(center=ff_center_below, size=ff_size_below)

# FourierFields 모니터 (Ez)
ff_Ez_below = mpa.FourierFields(sim, volume=ff_vol_below, component=mp.Ez)

# 동일 위치/길이로 DFT 모니터도 추가 (주파수 fcen 한 점)
check_monitor = sim.add_dft_fields([mp.Ez], fcen, 0, 1,
                                    center=ff_center_below, size=ff_size_below)


# --- FourierFields 모니터 (y=0.7 선) ---
ff_center = mp.Vector3(0, 0.7, 0)
ff_size   = mp.Vector3(sx, 0, 0)
ff_vol    = mp.Volume(center=ff_center, size=ff_size)

ff_Ez = mpa.FourierFields(sim, volume=ff_vol, component=mp.Ez)

# --- k_x vector 추가 보정 ---
margin = 0.4  # [um] 디자인 하단 바로 아래로 살짝 내림

# 디자인 하단 y좌표
design_bottom_y = design_region.center.y - 0.5 * design_region.size.y

# 추가 모니터의 중심과 크기 (x는 주기 전체 길이)
ff_center_below = mp.Vector3(0, design_bottom_y - margin, 0)
ff_size_below   = mp.Vector3(sx, 0, 0)
ff_vol_below    = mp.Volume(center=ff_center_below, size=ff_size_below)

ff_check_Ez = mpa.FourierFields(sim, volume=ff_vol_below, component=mp.Ez)

def J_kx(Ez_line_top, Ez_line_bot):
    # (추가) 목표 비율: 0° : +45°  (필요시 값만 바꿔 쓰면 됨)
    T0, Tp = 0.6, 0.4   # 예: 0° 70%, +45° 30% (원하면 0.5,0.5 등으로)        
    # ---------- 상부(매질 내) ----------
    Ez_top  = npa.ravel(Ez_line_top)
    fft_top = npa.fft.fftshift(npa.fft.fft(Ez_top))
    kx_top  = npa.fft.fftshift(npa.fft.fftfreq(len(fft_top), d=1.0/resolution))

    degree0_top = npa.argmin(npa.abs(kx_top))
    degree_p45  = degree0_top + 4  # +45° kx vector

    P0_top = npa.abs(fft_top[degree0_top])**2
    Pp_top = npa.abs(fft_top[degree_p45])**2

    # (핵심) 두 성분만 내부 정규화 → 다른 각 성분은 전혀 개입하지 않음
    sum2_top = P0_top + Pp_top
    r0_top = P0_top / sum2_top
    rp_top = Pp_top / sum2_top

    # 비율 오차(MSE) → 작을수록 좋음
    mse_top = (r0_top - T0)**2 + (rp_top - Tp)**2
    # 점수는 1 - 정규화된 MSE
    score_top = 1.0 - mse_top / (T0**2 + Tp**2)

    # ---------- 하부(공기측) ----------
    Ez_bot  = npa.ravel(Ez_line_bot)
    fft_bot = npa.fft.fftshift(npa.fft.fft(Ez_bot))
    kx_bot  = npa.fft.fftshift(npa.fft.fftfreq(len(fft_bot), d=1.0/resolution))

    degree0_bot = npa.argmin(npa.abs(kx_bot))
    degree_p45_bot = degree0_bot + 4

    P0_bot = npa.abs(fft_bot[degree0_bot])**2
    Pp_bot = npa.abs(fft_bot[degree_p45_bot])**2

    sum2_bot = P0_bot + Pp_bot
    r0_bot = P0_bot / sum2_bot
    rp_bot = Pp_bot / sum2_bot

    mse_bot = (r0_bot - T0)**2 + (rp_bot - Tp)**2
    score_bot = 1.0 - mse_bot / (T0**2 + Tp**2)

    # ---------- 위/아래 결합 ----------
    score = 0.5 * (score_top + score_bot)

    # 최대화 → 음수 반환
    return -npa.real(score)

def J_kx(Ez_line_top, Ez_line_bot):
    T0, Tp = 0.6, 0.2
    M45_BIN = 4
    EPS = 1e-30

    def power_and_balance(Ez_line):
        Ez  = npa.ravel(Ez_line)
        F   = npa.fft.fftshift(npa.fft.fft(Ez))
        kx  = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))

        k0  = npa.argmin(npa.abs(kx))
        kP  = k0 + M45_BIN

        P = npa.abs(F)**2
        Psum = npa.sum(P) + EPS
        P = P / Psum                     # 총 전력 = 1로 정규화
        Ptot = npa.sum(P)                # ≈ 1
        P0   = P[k0]
        Pp   = P[kP]

        a0 = P0 / (T0 + EPS)
        ap = Pp / (Tp + EPS)
        balance = (2.0 * a0 * ap) / (a0 + ap + EPS)
        return Ptot, balance
    Ptot_top, bal_top = power_and_balance(Ez_line_top)
    Ptot_bot, bal_bot = power_and_balance(Ez_line_bot)

    pow_hm = 2.0 * Ptot_top * Ptot_bot / (Ptot_top + Ptot_bot + EPS)  # ≈ 1
    bal_hm = 2.0 * bal_top * bal_bot   / (bal_top + bal_bot + EPS)

    total = 2.0 * pow_hm * bal_hm / (pow_hm + bal_hm + EPS)
    return -npa.real(total)

def J_kx(Ez_line_top, Ez_line_bot):
    T0, Tp, Tm  = 0.6, 0.2, 0.2     # 원하는 비율
    M45_BIN = 4            # +45°가 중앙 bin에서 +4

    t_sum = T0 + Tp + Tm
    t0 = T0 / t_sum
    tp = Tp / t_sum

    def hm(x, y):
        return (2.0 * x * y) / (x + y)

    def line_score(Ez_line):
        Ez = npa.ravel(Ez_line)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))

        P = npa.abs(F)**2
        P = P / npa.sum(P)

        k0 = npa.argmin(npa.abs(kx))
        kP = k0 + M45_BIN

        P0 = P[k0]
        Pp = P[kP]

        a0 = P0 / t0
        ap = Pp / tp
        ratio_score = hm(a0, ap)

        S = P0 + Pp
        return hm(ratio_score, S)

    # top/bot 결합(조화평균)
    s_top = line_score(Ez_line_top)
    s_bot = line_score(Ez_line_bot)
    score = hm(s_top, s_bot)

    # 최소화 대상
    return -npa.real(score)
def J_kx(Ez_line_top, Ez_line_bot):
    # 타깃 비율(합=1에서 0.6 : 0.2 : 0.2) → 여기서는 0°, +45° 두 성분만 사용
    T0, Tp, Tm = 0.6, 0.2, 0.2
    M45_BIN    = 4  # +45°가 중앙 bin에서 +4

    # 0°, +45° 타깃 분율(두 성분 합 = 0.8) — 코드 구조 유지
    t_sum = T0 + Tp + Tm            # = 1.0
    t0 = T0 / t_sum                 # = 0.6
    tp = Tp / t_sum                 # = 0.2

    # 조화평균 유틸
    def hm(x, y):
        return (2.0 * x * y) / (x + y)

    # --- 통계학적 더블체크: Hellinger만 활성화 ---
    def hellinger_one_minus_2(p0, p1, t0, t1):
        # 1 - H^2,  H^2 = 0.5 * Σ (sqrt(p_i) - sqrt(t_i))^2
        H2 = 0.5 * ((npa.sqrt(p0) - npa.sqrt(t0))**2 + (npa.sqrt(p1) - npa.sqrt(t1))**2)
        return 1.0 - H2  # ∈ [0,1], 클수록 타깃과 유사

    # (참고) Chi-square는 주석 처리 (원하면 주석 해제)
    # def chi_square_2(p0, p1, t0, t1):
    #     # χ^2 = Σ ( (p_i - t_i)^2 / t_i ), t_i>0 이므로 0-division 없음
    #     return ((p0 - t0) * (p0 - t0)) / t0 + ((p1 - t1) * (p1 - t1)) / t1

    # 단일 라인의 점수
    def line_score(Ez_line):
        # 1) FFT 및 kx
        Ez = npa.ravel(Ez_line)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))

        # 2) 파워 정규화 (합=1)
        P = npa.abs(F)**2
        P = P / npa.sum(P)

        # 3) 관심 bin: 0°, +45°
        k0 = npa.argmin(npa.abs(kx))
        kP = k0 + M45_BIN

        P0 = P[k0]     # 0°
        Pp = P[kP]     # +45°
        S  = P0 + Pp   # 두 성분 합(이상적으로 0.8 근처)

        # (기존) 비율 매칭 → HM(P0/t0, Pp/tp)
        a0 = P0 / t0
        ap = Pp / tp
        ratio_score = hm(a0, ap)

        # (추가) 통계학적 더블체크(두 성분 분포가 타깃과 맞는지) — Hellinger만 사용
        hell_score = hellinger_one_minus_2(P0, Pp, t0, tp)   # 클수록 좋음

        # (참고) Chi-square는 비활성 (원하면 아래 두 줄로 교체 가능)
        # chi2      = chi_square_2(P0, Pp, t0, tp)           # 작을수록 좋음
        # hell_score = 1.0 / (1.0 + chi2)                    # 0~1로 변환

        # (기존 목표)와 (통계 더블체크)를 HM으로 결합
        base_score = hm(ratio_score, S)                      # 비율 + 질량
        line_sc    = hm(base_score, hell_score)              # 통계 체크까지 반영

        return line_sc

    # top/bot 결합(작은 쪽에 민감)
    s_top = line_score(Ez_line_top)
    s_bot = line_score(Ez_line_bot)
    score = hm(s_top, s_bot)

    # 최소화 대상
    return -npa.real(score)


def J_kx(Ez_line_top, Ez_line_bot):
    T0, Tp = 0.6, 0.2           # 비율은 참고값일 뿐, 본 버전에서는 미사용
    M45_BIN = 4                 # +45°가 중앙 bin에서 +4
    S_target = T0 + Tp          # = 0.8

    # 조화평균
    def hm(x, y):
        return (2.0 * x * y) / (x + y)

    def line_score(Ez_line):
        # 1) FFT → 정규화 파워(합=1)
        Ez = npa.ravel(Ez_line)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))
        P  = npa.abs(F)**2
        P  = P / npa.sum(P)

        # 2) 관심 bin: 0°, +45°
        k0 = int(npa.argmin(npa.abs(kx)))
        kP = int(npa.clip(k0 + M45_BIN, 0, P.size - 1))

        P0, Pp = P[k0], P[kP]
        S      = P0 + Pp

        # 3) 합을 0.8로 맞추는 종모양 보상 (S=0.8에서 1, 벗어나면 감소)
        S_score = 1.0 / (1.0 + (S - S_target)**2)
        return S_score

    # top/bot 결합
    s_top = line_score(Ez_line_top)
    s_bot = line_score(Ez_line_bot)
    score = hm(s_top, s_bot)

    # 최소화 대상
    return -npa.real(score)

def J_kx(Ez_line_top, Ez_line_bot):
    # 원하는 타깃 비율과 총량(=선택한 성분의 합) 설정
    # 예: 0° 성분 60%, +45° 성분 20% (좌우 대칭으로 -45°도 20%라서 총 100%)
    T0, Tp = 0.6, 0.2

    def HM(x, y):
        # Harmonic mean: 위/아래 라인 점수를 균형 있게 결합
        return 2 * x * y / (x + y)

    def line_score(field):
        # field ............. 1D 라인상의 Ez (complex 가능)
        # FFT로 kx-스펙트럼 계산
        Ez = npa.ravel(field)
        fft = npa.fft.fftshift(npa.fft.fft(Ez))
        # 'resolution'은 격자 해상도(픽셀/단위길이)라고 가정 (사용자 환경에서 정의되어 있어야 함)
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(fft), d=1.0 / resolution))

        # 관심 성분의 인덱스 선택
        k0 = npa.argmin(npa.abs(kx))   # 0° 성분 (정확히 kx=0에 가장 가까운 bin)
        k45 = k0 + 4                   # +45° 성분 (사용자 설정: 격자/길이에 맞춰 고정 오프셋)

        # 파워 스펙트럼 정규화 (전체합 = 1)
        P = npa.abs(fft) ** 2
        P = P / npa.sum(P)

        # 우리가 최적화할 두 개의 성분
        P0 = P[k0]     # 0° 성분
        Pp = P[k45]    # +45° 성분  (−45°는 매핑으로 대칭 ⇒ 동일하다고 가정)

        # 선택 성분의 총합과 타깃 총합 (−45° 포함해서 +45°를 2배)
        S       = P0 + 2.0 * Pp
        S_tgt   = T0 + 2.0 * Tp

        # ----------- 정확한 비율 강제(교차곱) + 합 일치 제약 ----------- #
        # 비율(분배) 제약:  (Tp * P0) == (T0 * Pp)
        L_ratio = (Tp * P0 - T0 * Pp) ** 2

        # 합(선택 성분 총량) 제약:  S == S_tgt
        L_sum   = (S - S_tgt) ** 2

        # 가중치 (필요하면 조절 가능; 기본 1,1이면 두 제약을 동일 비중으로 맞춤)
        w_ratio = 1.0
        w_sum   = 1.0

        # 총 손실 (작을수록 좋음)
        L_total = w_ratio * L_ratio + w_sum * L_sum

        # 점수화: 손실이 0에 가까울수록 1에 가까운 점수
        # score = 1.0 / (1.0 + L_total)
        # return score
        return L_total

    # 위/아래 라인 각각 점수 계산
    Ez_top = line_score(Ez_line_top)
    Ez_bot = line_score(Ez_line_bot)

    # 두 라인 점수의 조화평균으로 최종 점수 결합
    score = HM(Ez_top, Ez_bot)

    # 최적화 루틴이 '최소화'를 한다면 음수 부호를 붙여 최대화와 등가로 만듦
    return -npa.real(score)

def J_kx(Ez_line_top, Ez_line_bot):
    T0, Tp = 0.6, 0.2

    def line_terms(field):
        Ez = npa.ravel(field)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))

        # 0°, +45° 인덱스 (각도 기반)
        k0  = npa.argmin(npa.abs(kx))
        k45_target = (1.0/lambda0)*npa.sin(npa.pi/4.0)  # cycles/um
        k45 = npa.argmin(npa.abs(kx - k45_target))

        P = npa.abs(F)**2
        P = P / npa.sum(P)

        P0, Pp = P[k0], P[k45]
        return P0, Pp

    P0_top, Pp_top = line_terms(Ez_line_top)
    P0_bot, Pp_bot = line_terms(Ez_line_bot)

    # 비율 손실
    Lr_top = (Tp*P0_top - T0*Pp_top)**2
    Lr_bot = (Tp*P0_bot - T0*Pp_bot)**2

    # 합 손실 (선택 성분 총합도 일치)
    S_top, S_bot = (P0_top + 2*Pp_top), (P0_bot + 2*Pp_bot)
    S_tgt = (T0 + 2*Tp)
    Ls_top = (S_top - S_tgt)**2
    Ls_bot = (S_bot - S_tgt)**2

    # 최소화 대상 (작을수록 좋음)
    w_ratio, w_sum = 1.0, 1.0
    L = w_ratio*(Lr_top + Lr_bot) + w_sum*(Ls_top + Ls_bot)
    return npa.real(L)

###############################################################################
def J_kx(Ez_line_top, Ez_line_bot):

    # ---- 타깃(예: 0° 60%, +45° 20%, -45°는 대칭 가정으로 +45°와 같음) ----
    T0, Tp = 0.6, 0.2
    S_tgt = T0 + 2.0*Tp              # 선택 성분 총합 타깃 (=1.0)
    r_tgt = Tp / T0                   # 목표 비율 (=1/3)
    m_ratio = 0.03                    # 허용 밴드 폭 (±3%p around r_tgt)

    # ---- 부드러운 hinge (C^1 ReLU 근사) ----
    eps = 1e-12
    def s_relu(x):
        return 0.5 * (x + npa.sqrt(x*x + eps))

    # ---- 한 라인에서 P0, Pp(+45°) 추출 ----
    def line_terms(field):
        Ez = npa.ravel(field)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0 / resolution))  # cycles/um

        # 0° bin, +45° bin(각도 기반; 배경 n≈1 가정)
        k0  = npa.argmin(npa.abs(kx))
        k45_target = (1.0 / lambda0) * npa.sin(npa.pi / 4.0)                # 1/(√2 λ0)
        k45 = npa.argmin(npa.abs(kx - k45_target))

        # 정규화 파워 스펙트럼
        P = npa.abs(F)**2
        P = P / npa.sum(P)

        P0, Pp = P[k0], P[k45]
        return P0, Pp

    # ---- 위/아래 라인 성분 파워 ----
    P0_t, Pp_t = line_terms(Ez_line_top)
    P0_b, Pp_b = line_terms(Ez_line_bot)

    # ---- 라인별 손실 정의 ----
    def line_loss(P0, Pp):
        # (1) 비율(교차곱) 손실: Tp*P0 == T0*Pp  → 0일 때 정확 분배
        L_ratio = (Tp * P0 - T0 * Pp)**2

        # (2) 선택 성분 총합 손실: P0 + 2*Pp == S_tgt
        S = P0 + 2.0 * Pp
        L_sum = (S - S_tgt)**2

        # (3) 뒤집힘 방지 밴드: r_min ≤ Pp/P0 ≤ r_max (미분가능 hinge)
        ratio = Pp / (P0 + eps)
        r_min = r_tgt - m_ratio
        r_max = r_tgt + m_ratio
        L_band = s_relu(r_min - ratio)**2 + s_relu(ratio - r_max)**2

        # 가중치 (필요시 조절)
        w_ratio = 1.0
        w_sum   = 1.0
        w_band  = 1.0

        return w_ratio*L_ratio + w_sum*L_sum + w_band*L_band
    L = npa.real(line_loss(P0_t, Pp_t) + line_loss(P0_b, Pp_b))
    return 1 / (1+L)
###############################################################################

################################ 이것도 굿굿 ################################
def J_kx(Ez_top, Ez_bot):
    # 타깃 분포
    T0, Tp = 0.6, 0.2
    q_star = npa.array([Tp, T0, Tp])  # 이미 합=1
    w_d, w_s = 0.7, 0.3               # 분배:총합 가중(합=1), 필요시 조정
    eps = 1e-12

    def P0_Pp(Ez):
        Ez = npa.ravel(Ez)
        F  = npa.fft.fftshift(npa.fft.fft(Ez))
        kx = npa.fft.fftshift(npa.fft.fftfreq(len(F), d=1.0/resolution))
        k0  = npa.argmin(npa.abs(kx))                                 # 0°
        k45 = npa.argmin(npa.abs(kx - (1.0/lambda0)*npa.sin(npa.pi/4)))  # +45°
        P   = npa.abs(F)**2
        P   = P / (npa.sum(P) + eps)                                   # 전력 정규화
        return P[k0], P[k45]

    def js_divergence(p, q):
        m = 0.5*(p + q)
        # KL(p‖m) + KL(q‖m): 0*log(0/.) = 0 취급을 위해 eps 더함
        kl_pm = npa.sum(p * (npa.log((p + eps)/(m + eps))))
        kl_qm = npa.sum(q * (npa.log((q + eps)/(m + eps))))
        return 0.5*(kl_pm + kl_qm)  # ∈ [0, ln 2]

    def line_loss(Ez):
        P0, Pp = P0_Pp(Ez)
        S  = P0 + 2.0*Pp                           # 선택 성분 총합 (절대 세기)
        qh = npa.array([Pp, P0, Pp]) / (S + eps)   # 관측 분포 [−45, 0, +45] (대칭 가정)
        L_dist = js_divergence(qh, q_star) / npa.log(2.0)   # ∈ [0,1]
        L_sum  = ((S - 1.0) / 2.0)**2                        # ∈ [0,1]
        return w_d*L_dist + w_s*L_sum

    # 두 라인 평균 손실 (0~1)
    L = 0.5*(line_loss(Ez_top) + line_loss(Ez_bot))
    return 1/(1+L)
#############################################################################

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J_kx],
    objective_arguments=[ff_Ez, ff_check_Ez],
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-3,
    minimum_run_time = 50/fcen
)

from autograd.extend import primitive, defvjp

@primitive
def _layer_division_with_padding_2d(X, L):
    NX, NY = X.shape
    L = int(L)
    if NY <= 1:
        return X
    if L <= 1:
        m = npa.mean(X, axis=1, keepdims=True)     # (NX,1)
        return npa.tile(m, (1, NY))                # (NX,NY) 전체 y-평탄화
    L_eff = min(L, (NY + 1) // 2)
    pads = L_eff - 1
    avail = NY - pads
    base, extra = divmod(avail, L_eff)
    sizes = [base + (1 if i < extra else 0) for i in range(L_eff)]
    cols, y0 = [], 0
    for i, s in enumerate(sizes):
        m = npa.mean(X[:, y0:y0+s], axis=1, keepdims=True)  # (NX,1)
        cols.append(npa.tile(m, (1, s)))                    # (NX,s)
        y0 += s
        if i < L_eff - 1:
            cols.append(npa.full((NX, 1), 0.5))             # pad 1픽셀
    return npa.concatenate(cols, axis=1)

def _ldp_vjp(g, X, L):
    NX, NY = X.shape
    L = int(L)
    if NY <= 1:
        return g
    if L <= 1:
        gsum = npa.sum(g, axis=1, keepdims=True)           # (NX,1)
        return npa.tile(gsum / NY, (1, NY))                # (NX,NY)
    L_eff = min(L, (NY + 1) // 2)
    pads = L_eff - 1
    avail = NY - pads
    base, extra = divmod(avail, L_eff)
    sizes = [base + (1 if i < extra else 0) for i in range(L_eff)]
    gout = npa.zeros_like(X)
    y_in = 0
    y_out = 0
    for i, s in enumerate(sizes):
        g_seg = g[:, y_out:y_out + s]                      # (NX,s)
        gsum = npa.sum(g_seg, axis=1, keepdims=True)       # (NX,1)
        gout = npa.concatenate(
            (gout[:, :y_in], npa.tile(gsum / s, (1, s)), gout[:, y_in + s:]),
            axis=1
        )
        y_in  += s
        y_out += s
        if i < L_eff - 1:
            y_out += 1                                     # pad grad=0
    return gout

defvjp(
    _layer_division_with_padding_2d,
    lambda ans, X, L: lambda g: _ldp_vjp(g, X, L),
    lambda ans, X, L: lambda g: None
)

def grating_mapping(x, eta, beta, desired_layers=None):
    X = npa.clip(x.reshape(NX, NY), 0.0, 1.0)              # (NX,NY)
    L = DESIRED_LAYERS if desired_layers is None else int(desired_layers)
    X = _layer_division_with_padding_2d(X, L)
    X = mpa.tanh_projection(X, beta=float(beta), eta=float(eta))
    X = 0.5 * (npa.flipud(X) + X)
    X = mpa.tanh_projection(X, beta=float(beta), eta=float(eta))
    return npa.clip(X, 0.0, 1.0).reshape(NX * NY)

# def free_mapping(x, eta, beta):
#     filt = mpa.conic_filter(x, filter_radius, DESIGN_W, DESIGN_H, design_res)   # (NY, NX)
#     projected_field = mpa.tanh_projection(filt, beta, eta)
#     projected_field = (
#         npa.flipud(projected_field) + projected_field
#     ) / 2

#     return projected_field.flatten()

def free_mapping(x, eta, beta):
    filt = mpa.conic_filter(x, filter_radius, DESIGN_W, DESIGN_H, design_res)   # (NY, NX)
    filt = (
        npa.flipud(filt) + filt
    ) / 2
    projected_field = mpa.tanh_projection(filt, beta, eta)

    return projected_field.flatten()

mapping = free_mapping if DESIGN_MODE == 'free' else grating_mapping

# ---- FoM + Grad (TJP) ----
def f_and_grad(v, eta, beta):
    rho = mapping(v, eta, beta)   # (n_vars,)

    f_raw, dJdrho = opt([rho], need_value=True, need_gradient=True)
    f_val = float(np.real(np.asarray(f_raw)).ravel()[0])

    dJdrho_flat = np.asarray(dJdrho).ravel()
    g_v = tensor_jacobian_product(mapping, 0)(v, eta, beta, dJdrho_flat)
    return f_val, np.asarray(g_v)

# ---- (최소) Adam Optimizer (Descent) ----
class AdamDescent:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, warmup=10, max_step=None):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None
        self.warmup = warmup
        self.max_step = max_step

    def step(self, x, grad):
        if self.m is None: self.m = np.zeros_like(x)
        if self.v is None: self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad*grad)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        lr_t = self.lr * (self.t/self.warmup) if self.t <= self.warmup else self.lr
        step = lr_t * mhat / (np.sqrt(vhat) + self.eps)
        if self.max_step is not None:
            step = np.clip(step, -self.max_step, self.max_step)
        # Descent: x <- x - step
        x_new = x - step
        # 디자인 변수 경계 투영
        return np.clip(x_new, 0.0, 1.0)

# ---- (최소) Adam Optimizer (Ascent) ----
class AdamAscent:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, warmup=10, max_step=None):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None
        self.warmup = warmup
        self.max_step = max_step

    def step(self, x, grad):
        """
        grad: 목적함수 J에 대한 ∂J/∂x (즉, J를 '최대화'하려는 방향의 기울기)
        Ascent 업데이트: x <- x + step
        """
        if self.m is None: self.m = np.zeros_like(x)
        if self.v is None: self.v = np.zeros_like(x)

        # 모멘트 업데이트
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)

        # 바이어스 보정
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)

        # 워밍업 스케줄
        lr_t = self.lr * (self.t / self.warmup) if self.t <= self.warmup else self.lr

        # 스텝 크기 계산
        step = lr_t * mhat / (np.sqrt(vhat) + self.eps)

        # 스텝 클리핑(선택)
        if self.max_step is not None:
            step = np.clip(step, -self.max_step, self.max_step)

        # ★ Ascent: x <- x + step (Descent는 x - step)
        x_new = x + step

        # 디자인 변수 경계 투영 [0,1]
        return np.clip(x_new, 0.0, 1.0)

# # 2. 최적화 과정에서 기록할 데이터 리스트
# # ------------------------------------------------------------------------------
beta_history = []
binarization_history = []
iteration_count = [0] # 리스트를 사용해 callback 함수 내에서 값 변경

# # 3. 시각화 및 저장을 위한 헬퍼(Helper) 함수
# ------------------------------------------------------------------------------
def binarize(weights, beta, eta):
    return (np.tanh(beta * eta) + np.tanh(beta * (weights - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )

def calculate_binarization_metric(weights):
    """이진화 정도를 측정하는 함수 (0.5에 가까울수록 1, 0 또는 1에 가까울수록 0)"""
    return np.mean(4 * weights * (1 - weights))

LABEL_SIZE = 16

def save_design_plot(d_vars, d_region, beta_val, eta_val, filepath,
                     title="Design", use_local_axes=True, normalize_axes=False,
                     m1_label="Medium2", m2_label="Medium1",
                     label_size=12, fig_width=7.0):

    # ▶ weights를 '있는 그대로' 가져오고, 1D일 때만 복원
    w = np.asarray(d_vars.weights)
    if w.ndim == 1:
        w = w.reshape(NX, NY)        # (x, y)
    elif w.ndim == 2:
        # (NX, NY) or (NY, NX)인지 확인 필요 없이 그대로 두고 플롯에서만 .T 사용
        pass
    else:
        raise ValueError("d_vars.weights의 차원이 1 또는 2가 아닙니다.")

    # 축 범위
    if use_local_axes:
        if normalize_axes:
            xmin, xmax = 0.0, 1.0
            ymin, ymax = 0.0, 1.0
            x_label = "x (normalized)"; y_label = "y (normalized)"
        else:
            xmin, xmax = 0.0, DESIGN_W
            ymin, ymax = 0.0, DESIGN_H
            x_label = "x (µm)"; y_label = "y (µm)"
    else:
        xmin = d_region.center.x - 0.5 * d_region.size.x
        xmax = d_region.center.x + 0.5 * d_region.size.x
        ymin = d_region.center.y - 0.5 * d_region.size.y
        ymax = d_region.center.y + 0.5 * d_region.size.y
        x_label = "x (µm)"; y_label = "y (µm)"

    dx = DESIGN_W / NX
    dy = DESIGN_H / NY
    extent_half_pixel = [xmin - dx/2, xmax - dx/2, ymin - dy/2, ymax - dy/2]

    aspect = DESIGN_H / DESIGN_W
    fig_w = float(fig_width)
    fig_h = max(2.5, fig_w * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150, constrained_layout=True)

    # ▶ 플롯에서만 (y,x)로 보이도록 전치
    im = ax.imshow(
        w.T,
        interpolation='none',
        cmap='Greys_r',
        extent=extent_half_pixel,
        origin='lower',
        vmin=0, vmax=1
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.95)
    cbar.set_label("")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels([m2_label, m1_label])
    cbar.ax.tick_params(labelsize=label_size)

    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(title, fontsize=label_size)
    ax.tick_params(axis='both', labelsize=label_size)

    try:
        ax.set_box_aspect(aspect)
    except Exception:
        ax.set_aspect('equal', adjustable='box')

    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# ############ Beta update ############
# def update_beta(evaluation_history, binarization_history, beta):
#     if len(evaluation_history) < 3:
#         return beta  # 결정을 내리기에 데이터가 충분하지 않음

#     # 최근 3번의 반복에서 기록된 값들을 가져옵니다.
#     f_prev2, f_prev1, f_curr = evaluation_history[-3:]
#     bin_prev2, bin_prev1, bin_curr = binarization_history[-3:]

#     # 목표 함수(FoM)의 상대적 변화량을 계산합니다.
#     fom_change1 = abs(f_curr - f_prev1) / (abs(f_prev1) + 1e-12)
#     fom_change2 = abs(f_prev1 - f_prev2) / (abs(f_prev2) + 1e-12)

#     # 이진화 정도의 상대적 변화량을 계산합니다.
#     bin_change1 = abs(bin_curr - bin_prev1) / (abs(bin_prev1) + 1e-12)
#     bin_change2 = abs(bin_prev1 - bin_prev2) / (abs(bin_prev2) + 1e-12)

#     # 수렴 기준을 정의합니다.
#     fom_converged = fom_change1 < 0.005 and fom_change2 < 0.005
#     bin_converged = bin_change1 < 0.002 and bin_change2 < 0.002

#     # 목표 함수와 이진화 정도가 모두 수렴했다면 beta를 증가시킵니다.
#     if fom_converged and bin_converged:
#         print("Convergence detected. Increasing beta.")
#         # 사용자 정의 업데이트 규칙
#         # new_beta = beta + np.tanh((beta - 0.5) * 0.02)
#         beta *= 1.4
#         new_beta = beta
#         return new_beta
    
#     return beta # 변경 없음

# def update_beta(evaluation_history, binarization_history, beta):
#     it = len(evaluation_history)
#     if it > 0 and (it % 100 == 0):
#         update_rate = 1.5
#         beta *= update_rate
#         print(f"[Schedule β] iter={it}: beta ×{update_rate} → {beta:.3f}")
#     return beta

def update_beta(evaluation_history, binarization_history, beta):
    it = len(evaluation_history)

    # [추가] 최초 호출 시 초기 β를 기억 (기존 변수명은 유지, 추가 변수는 함수 속성으로 관리)
    if not hasattr(update_beta, "_beta0"):
        update_beta._beta0 = float(beta)  # 초기 β 저장

    if it > 0 and (it % 10 == 0):
        update_rate = 1.5  # 기존 변수명 유지
        milestones = it // 100  # 100 스텝 단위 누적 횟수
        # beta = update_beta._beta0 * (update_rate ** milestones)
        beta = beta + 9*np.tanh((beta - 0.5) * 0.02)
        print(f"[Schedule β] iter={it}: beta ← beta0×({update_rate})^{milestones} → {beta:.6f}")

    return beta

def plot_history(beta_hist, binarization_hist, eval_hist, base_filepath):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    n_beta_bin = min(len(beta_hist), len(binarization_hist))
    if n_beta_bin == 0:
        print("[plot_history] 기록이 없어 플롯을 건너뜁니다.")
        return

    base_root, _ = os.path.splitext(base_filepath)
    out_dir = os.path.dirname(base_root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    iterations = np.arange(1, n_beta_bin + 1)

    # Beta History
    fig_beta, ax_beta = plt.subplots(figsize=(10, 10), dpi=120)
    color_beta = "tab:orange"
    ax_beta.plot(iterations, beta_hist[:n_beta_bin], linestyle='-', linewidth=2, color=color_beta)
    ax_beta.set_xlabel('Iteration', fontsize=LABEL_SIZE)
    ax_beta.set_ylabel('Beta', fontsize=LABEL_SIZE)
    ax_beta.set_title('Beta History', fontsize=LABEL_SIZE)
    ax_beta.tick_params(axis='both', labelsize=LABEL_SIZE)
    ax_beta.grid(True)
    fig_beta.tight_layout()
    fig_beta.savefig(base_root + "_beta.png")
    plt.close(fig_beta)
    print(f"[plot_history] Beta 기록 플롯 저장: {base_root}_beta.png")

    # Binarization History
    fig_bin, ax_bin = plt.subplots(figsize=(10, 10), dpi=120)
    color_bin = "tab:blue"
    ax_bin.plot(iterations, binarization_hist[:n_beta_bin], linestyle='-', linewidth=2, color=color_bin)
    ax_bin.set_xlabel('Iteration', fontsize=LABEL_SIZE)
    ax_bin.set_ylabel('Grayness [%]  (0 = binary, 100 = gray)', fontsize=LABEL_SIZE)
    ax_bin.set_title('Binarization History', fontsize=LABEL_SIZE)
    ax_bin.set_ylim(0, 100)
    ax_bin.tick_params(axis='both', labelsize=LABEL_SIZE)
    ax_bin.grid(True)
    fig_bin.tight_layout()
    fig_bin.savefig(base_root + "_binarization.png")
    plt.close(fig_bin)
    print(f"[plot_history] Binarization 기록 플롯 저장: {base_root}_binarization.png")

    # FoM History
    if len(eval_hist) > 0:
        n_all = min(n_beta_bin, len(eval_hist))
        iterations_f = np.arange(1, n_all + 1)

        fig_fom, ax_fom = plt.subplots(figsize=(10, 10), dpi=120)
        ax_fom.plot(iterations_f, eval_hist[:n_all], linestyle='-', linewidth=2)
        ax_fom.set_xlabel('Iteration', fontsize=LABEL_SIZE)
        ax_fom.set_ylabel('FoM', fontsize=LABEL_SIZE)
        ax_fom.set_title('Figure of Merit (FoM) History', fontsize=LABEL_SIZE)
        ax_fom.tick_params(axis='both', labelsize=LABEL_SIZE)
        ax_fom.grid(True)
        # ax_fom.set_ylim(-1.01,0.01)
        fig_fom.tight_layout()
        fig_fom.savefig(base_root + "_fom.png")
        plt.close(fig_fom)
        print(f"[plot_history] FoM 기록 플롯 저장: {base_root}_fom.png")
    else:
        print("[plot_history] FoM 기록이 없어 FoM 플롯을 생략합니다.")

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if DESIGN_MODE == 'free' :
    top_folder_name = f"{timestamp}_{DESIGN_MODE}"
else :
    top_folder_name = f"Layer_{DESIRED_LAYERS} {timestamp}_{DESIGN_MODE}"

os.makedirs(top_folder_name, exist_ok=True)

beta_bin_folder = os.path.join(top_folder_name, "Beta&Binarization")
design_folder   = os.path.join(top_folder_name, "Design")
os.makedirs(beta_bin_folder, exist_ok=True)
os.makedirs(design_folder,   exist_ok=True)

print(f"[INFO] 결과 저장 폴더: {top_folder_name}")

# ---------- 초기 상태 저장 (디자인 플롯) ----------
initial_design_path = os.path.join(design_folder, "initial_design.png")
save_design_plot(
    d_vars=design_vars,
    d_region=design_region,
    beta_val=beta,
    eta_val=eta_i,
    filepath=initial_design_path,
    title="Initial Design",
    use_local_axes=True,
    normalize_axes=False
)

print(f"[INFO] 초기 디자인 저장: {initial_design_path}")

# ---------- 최적화 준비 ----------
v = np.asarray(x).reshape(-1)  # (n_vars,)

ETA        = eta_i
BETA       = beta             # 초기/기본 β (미니멀 구성)
MAX_ITERS  = 300
optimizer  = AdamAscent(lr=0.03, beta1=0.8, beta2=0.999, eps=1e-8, warmup=10, max_step=0.02)

# 히스토리
f_history             = []
beta_history          = []
binarization_history  = []
evaluation_history    = []     # FoM 기록(= f_history와 동일하게도 쓸 수 있으나 별도 유지)

# --- 루프 들어가기 전: 누적/플래그 변수 추가 ---
grad_accum = np.zeros_like(v)   # β 변경 전까지의 그래디언트 누적 (elementwise)
steps_since_beta = 0            # 누적 스텝 수
pending_stab_grad = None        # 다음 반복에서 1회 안정화 스텝에 사용할 평균 그래디언트

# --- 루프 전에: 이전 gradient 버퍼 추가 ---
prev_grad = None  # 직전 반복의 ∂J/∂v 저장용

print("\n[최적화 시작]")
for it in range(1, MAX_ITERS + 1):
    fval, grad_curr = f_and_grad(v, ETA, BETA)

    if prev_grad is not None:
        grad_used = 0.5 * (grad_curr + prev_grad)
    else:
        grad_used = grad_curr

    v = optimizer.step(v, grad_used)

    f_history.append(float(fval))
    evaluation_history.append(float(fval))
    beta_history.append(float(BETA))

    rho_now = np.asarray(mapping(v, ETA, BETA))
    bin_metric = float(calculate_binarization_metric(rho_now)) * 100
    binarization_history.append(bin_metric)

    print(f"Iter {it:4d} | J = {fval:.8e} | Grayness = {bin_metric:.4f} | beta = {BETA:.3f}")

    prev_grad = grad_curr

    if bin_metric <= 5.0:
        BETA = float('inf')
        rho_now = np.asarray(mapping(v, ETA, BETA))
        print("[종료] Grayness ≤ 5 → β=∞로 고정, 최종 디자인 확정")
        break

    new_beta = float(update_beta(evaluation_history, binarization_history, BETA))
    if new_beta != BETA:
        print(f"[INFO] β 업데이트: {BETA:.3f} → {new_beta:.3f}")
    BETA = new_beta
print("[최적화 종료]")

# # ---------- 최종 weight 업데이트 (MaterialGrid) ----------
# rho_mapped  = np.asarray(mapping(v, ETA, BETA))      # shape: (NY*NX,)
# weights_sym = rho_mapped.reshape(NX, NY)                  # ★ (x, y)로 reshape 변경

# design_vars.update_weights(weights_sym.T)  # 내부가 (NY, NX)를 기대한다면 .T로 맞춤 (또는 내부 기대에 맞게 조정)

# final_design_path = os.path.join(design_folder, "final_design.png")

# dx = DESIGN_W / NX
# dy = DESIGN_H / NY
# extent_half_pixel = [-dx/2, DESIGN_W - dx/2, -dy/2, DESIGN_H - dy/2]

# fig, ax = plt.subplots(figsize=(10, 10 * DESIGN_H / DESIGN_W), dpi=150)
# im = ax.imshow(
#     weights_sym.T,                 # ★ 전치 유지 (요청 사항)
#     interpolation='none',
#     cmap='Greys_r',
#     extent=extent_half_pixel,
#     origin='lower',
#     vmin=0, vmax=1
# )
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label("weights", fontsize=LABEL_SIZE)
# cbar.ax.tick_params(labelsize=LABEL_SIZE)
# ax.set_xlabel("x (µm)", fontsize=LABEL_SIZE)
# ax.set_ylabel("y (µm)", fontsize=LABEL_SIZE)
# ax.set_title("Final Design", fontsize=LABEL_SIZE)
# ax.tick_params(axis='both', labelsize=LABEL_SIZE)
# ax.set_aspect('equal', adjustable='box')
# fig.tight_layout()
# plt.savefig(final_design_path)
# plt.close(fig)


# print(f"[INFO] 최종 디자인 저장(매핑 반영): {final_design_path}")

# ---------- 최종 weight 업데이트 (MaterialGrid) ----------
rho_mapped  = np.asarray(mapping(v, ETA, BETA))        # shape: (NY*NX,)
weights_sym = rho_mapped.reshape(NX, NY)               # (x, y)로 reshape

design_vars.update_weights(weights_sym)              # MaterialGrid expects (NY, NX)

final_design_path = os.path.join(design_folder, "final_design.png")

# save_design_plot 사용해 저장
save_design_plot(
    d_vars=design_vars,
    d_region=design_region,
    beta_val=BETA,
    eta_val=ETA,
    filepath=final_design_path,
    title="Final Design",
    use_local_axes=True,
    normalize_axes=False,
    # 필요 시 라벨 지정 (원하지 않으면 두 인자 제거 가능)
    m1_label="Medium2 (n=1.0)",
    m2_label="Medium1 (n=2.0)"
)

print(f"[INFO] 최종 디자인 저장(매핑 반영): {final_design_path}")

# 2) 가중치/히스토리 텍스트 저장
np.savetxt(os.path.join(top_folder_name, "final_weights.txt"), weights_sym)
print(f"[INFO] 최종 가중치/히스토리 저장 완료: {top_folder_name}")

# 3) 히스토리 플롯 (유틸 함수 사용)
history_base = os.path.join(beta_bin_folder, "history")
plot_history(beta_history, binarization_history, evaluation_history, history_base)
print(f"[INFO] 기록 플롯 저장 완료: {history_base}_*.png")
