#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# ---- 스타일 파라미터
def _default_rcparams():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1
    })

# figsize_DC_tall = (3.3, 3.3)
# figsize_SC_two_third = (4, 2.3) # single column

# ---- 핵심 함수
def plot_unbalance_with_annotations(x, y, *,
                                    figsize=(4, 2.3),
                                    smooth=True,
                                    smooth_strength=1e-4,   # 스무딩 강도 (필요 시 조정)
                                    x_oper=None,            # 운영속도(표시선)
                                    annotate_second=True,   # 두 번째 봉우리도 점만 표시
                                    savepath=None):
    """
    x: 속도(정규화/단위무관), y: 진폭
    smooth_strength: UnivariateSpline의 s 값 스케일(데이터 길이와 분산에 자동 가중)
    """
    _default_rcparams()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 스무딩 곡선
    if smooth and x.size >= 5:
        # 데이터 스케일에 맞춰 s 자동 스케일링
        s_auto = smooth_strength * x.size * np.var(y)
        spl = UnivariateSpline(x, y, s=s_auto, k=3)
        xs = np.linspace(x.min(), x.max(), max(400, 8*len(x)))
        ys = spl(xs)
    else:
        xs, ys = x, y

    # 피크 탐색 (스무딩 곡선 기준)
    pk_idx, _ = find_peaks(ys)
    if pk_idx.size == 0:
        raise RuntimeError("피크를 찾지 못했습니다. 스무딩 강도(smooth_strength)를 줄이거나 데이터를 확인하세요.")

    # 가장 큰 피크 = 주요 공진
    i_main = pk_idx[np.argmax(ys[pk_idx])]
    Nc1 = xs[i_main]             # 공진 중심
    peak_val = ys[i_main]
    half = 0.70710678 * peak_val # half-power

    # half-power 좌/우 교차점(N1, N2)
    def cross_left(ix_center):
        yy = ys[:ix_center+1]
        xx = xs[:ix_center+1]
        below = np.where(yy <= half)[0]
        if below.size == 0:
            return xx[0]
        i0 = below[-1]
        if i0 == ix_center:
            return xx[i0]
        # 선형 보간
        x0, y0 = xx[i0], yy[i0]
        x1, y1 = xx[i0+1], yy[i0+1]
        t = (half - y0) / (y1 - y0 + 1e-18)
        return x0 + t * (x1 - x0)

    def cross_right(ix_center):
        yy = ys[ix_center:]
        xx = xs[ix_center:]
        below = np.where(yy <= half)[0]
        if below.size == 0:
            return xs[-1]
        i1 = below[0]
        if i1 == 0:
            return xx[i1]
        x0, y0 = xx[i1-1], yy[i1-1]
        x1, y1 = xx[i1], yy[i1]
        t = (half - y0) / (y1 - y0 + 1e-18)
        return x0 + t * (x1 - x0)

    N1 = cross_left(i_main)
    N2 = cross_right(i_main)

    # 두 번째 봉우리(있으면 표시용 점)
    i_second = None
    if annotate_second and pk_idx.size >= 2:
        # 메인 피크 제외하고 가장 큰 것
        rest = pk_idx[pk_idx != i_main]
        if rest.size:
            i_second = rest[np.argmax(ys[rest])]

    # ---- 그리기
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, ys, '-', label='Vibration response')

    # 보조선: peak, 0.707*peak
    # ax.axhline(peak_val, linestyle=':', linewidth=1)
    # ax.axhline(half, linestyle='--', linewidth=1)
    
    ax.plot([0, Nc1], [peak_val, peak_val], linestyle=':', linewidth=1, color='k')
    ax.plot([0,N2], [half, half], linestyle='--', linewidth=1, color='k')

    # 세로선: N1, Nc1, N2
    # for xv, ls in [(N1, ':'), (Nc1, '--'), (N2, ':')]:
    #     ax.axvline(xv, linestyle=ls, linewidth=1, color='k')

    ax.plot([N1,N1], [0,0.707*peak_val], linestyle=':', linewidth=1, color='k') 
    ax.plot([N2,N2], [0,0.707*peak_val], linestyle=':', linewidth=1, color='k') 
    ax.axvline(Nc1, linestyle='--', linewidth=1, color='k')
    # ax.plot([N1,N1], [0,peak_val], linestyle=':', linewidth=1, color='k') 
    
    
    # 라벨 텍스트
    ax.text(Nc1*0.5, peak_val*1.02, r'peak', ha='center', va='bottom')
    ax.text(xs[0]+0.01, half*1.02, r'$0.707\,\times$ peak', va='bottom')
    ax.text(N1-0.01, ax.get_ylim()[0]+0.05, r'$N_{1}$', ha='right', va='bottom')
    ax.text(Nc1, ax.get_ylim()[0]+0.01, r'$N_{c1}$', ha='center', va='top')
    ax.text(N2+0.01, ax.get_ylim()[0]+0.05, r'$N_{2}$', ha='left', va='bottom')

    # SM 화살표 (N1 → Nc1)
    ax.annotate("", xy=(Nc1, peak_val*1.15), xytext=(x_oper, peak_val*1.15),
                arrowprops=dict(arrowstyle='<->', lw=1))
    ax.text((x_oper+Nc1)/2, peak_val*1.2, r'$\mathrm{SM}$', ha='center', va='bottom')

    # 운영속도 표시(옵션)
    if x_oper is not None:
        ax.axvline(x_oper, color='tab:red', linestyle='-', linewidth=1)
        # ax.text(x_oper, ax.get_ylim()[0], r'$N_{\mathrm{MCOS}}$', color='tab:red',
        #         ha='center', va='bottom')
        ax.text(x_oper, ax.get_ylim()[0]+0.01, r'$N_{\mathrm{Oper}}$', color='tab:red',
                ha='center', va='top')

    # 두 번째 봉우리 점 표시
    if i_second is not None:
        ax.plot(xs[i_second], ys[i_second], 'o', ms=3, color='k')
        ax.text(xs[i_second], ys[i_second], r'$N_{c n}$', ha='left', va='bottom')

    ax.set_xlabel('Rotor speed',labelpad=14)
    ax.set_ylabel('Vibration Response')
    ax.set_ylim(0,0.9)
    ax.set_xlim(0,0.95)
    # ax.grid(True, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=600, bbox_inches='tight')
    return dict(N1=N1, Nc1=Nc1, N2=N2, peak=peak_val, half=half)

# ---- 예시 실행 (질문에 준 데이터 사용)
x = np.array([
    0.0027,0.0442,0.0849,0.1238,0.1503,0.1724,0.1954,0.2104,0.2175,0.2246,
    0.237,0.2431,0.252,0.2538,0.2599,0.2688,0.2732,0.2838,0.2918,0.2989,
    0.3033,0.3068,0.3112,0.3139,0.3165,0.321,0.3236,0.3254,0.3298,0.3333,
    0.3386,0.3431,0.3484,0.3554,0.3705,0.3802,0.3899,0.405,0.4209,0.4341,
    0.4474,0.4624,0.4766,0.4916,0.5022,0.5128,0.534,0.5676,0.6039,0.6375,
    0.6711,0.6976,0.7233,0.7321,0.7507,0.763,0.7807,0.8019,0.8161,0.8338,
    0.847,0.8532,0.8647,0.8727,0.8842,0.8948,0.9089,0.9187,0.9346,0.9452,
    0.9523,0.9602,0.9673,0.9735,0.9814
])
y = np.array([
    0.018,0.0386,0.0694,0.0874,0.1093,0.1324,0.1645,0.2069,0.2391,0.2725,
    0.3175,0.356,0.3907,0.419,0.4743,0.5283,0.5591,0.5784,0.5861,0.5707,
    0.5411,0.5167,0.4897,0.4614,0.4293,0.4023,0.3792,0.3509,0.3033,0.2879,
    0.2712,0.2481,0.2237,0.2082,0.1864,0.1787,0.1697,0.162,0.1555,0.1504,
    0.1465,0.144,0.1414,0.1414,0.1401,0.1388,0.135,0.1311,0.126,0.126,
    0.1247,0.1311,0.1362,0.1401,0.1555,0.171,0.1902,0.2185,0.2404,0.2416,
    0.2314,0.2211,0.2044,0.1915,0.1851,0.1825,0.1838,0.1864,0.1902,0.1954,
    0.1992,0.2031,0.2082,0.2121,0.2159
])*1.1

# 예시 호출 (운영속도 미지정)
res = plot_unbalance_with_annotations(x, y, smooth=True, smooth_strength=3e-4,savepath='api_unb.png',x_oper=0.52)
print(res)