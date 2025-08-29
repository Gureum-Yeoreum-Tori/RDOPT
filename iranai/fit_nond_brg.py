#%%
import h5py, numpy as np
from typing import Dict, Any, List, Tuple
from scipy.interpolate import PchipInterpolator
from import_data import Bearing

COEFFS = ["Kxx","Kxy","Kyx","Kyy","Cxx","Cxy","Cyx","Cyy"]

class BearingNondModel:
    def __init__(self, filepath: str = "dataset/data/nondBrg.h5"):
        self.filepath = filepath
        self.data: Dict[str, Dict[str, Any]] = {}
        self._load_fit_from_rdcr()

    # ---- RDCRaw 전용 로더/피터 ----
    def _load_fit_from_rdcr(self):
        print(f"'{self.filepath}'에서 베어링 데이터를 메모리에 로딩합니다...")
        self.data = {}
        with h5py.File(self.filepath, "r") as f:
            for gname, grp in f.items():              # gname: 'brg_17' 등
                out: Dict[str, Any] = {}

                # 스칼라/배열 기타 메타 값은 그대로 적재(있으면)
                for k, ds in grp.items():
                    if isinstance(ds, h5py.Dataset) and k != "RDCRaw":
                        val = ds[()]
                        out[k] = np.array(val).item() if np.isscalar(val) or (isinstance(val, np.ndarray) and val.shape == ()) \
                                else np.asarray(val).squeeze()

                # RDCRaw 로드 → PCHIP 피팅
                if "RDCRaw" in grp:
                    raw = grp["RDCRaw"][()]
                    coeff_funcs = self._fit_pchip_from_rdcr(raw)   # dict: {'fKxx': PCHIP, ...}
                    out.update(coeff_funcs)

                self.data[gname] = out
        print("데이터 로딩 완료.")

    @staticmethod
    def _fit_pchip_from_rdcr(raw) -> Dict[str, Any]:
        """
        raw가 구조화배열이면 필드명 사용,
        2D 배열이면 열 인덱스 규칙 사용:
        [S, eps, phi, Kxx, Kxy, Kyx, Kyy, Cxx, Cxy, Cyx, Cyy]
        """
        # --- S와 각 계수 열 추출 ---
        if isinstance(raw, np.ndarray) and raw.dtype.names:   # structured array
            names = {n.lower(): n for n in raw.dtype.names}
            S = np.asarray(raw[names["s"]], dtype=float).ravel()
            cols = {c: np.asarray(raw[names[c.lower()]], dtype=float).ravel() for c in COEFFS}
        else:                                                  # 2D numeric table
            A = np.asarray(raw, dtype=float)
            S = A[:, 0]
            cols = {c: A[:, i] for c, i in zip(COEFFS, range(3, 3+len(COEFFS)))}

        # --- 정렬(+중복 제거) ---
        idx = np.argsort(S)
        S = S[idx]
        for k in cols: cols[k] = np.asarray(cols[k])[idx]

        # --- PCHIP 생성 ---
        fits = {}
        for c in COEFFS:
            fits["f"+c] = PchipInterpolator(S, cols[c], extrapolate=True)

        return fits

    # ---- 조회/사용 ----
    def get_bearing_by_id(self, bearing_id: int) -> Dict[str, Any]:
        key = f"brg_{bearing_id}"
        if key not in self.data:
            raise KeyError(f"{key} not found")
        return self.data[key]

    def get_dynamic_coefficients(self, bearing_id: int) -> Dict[str, Any]:
        brg = self.get_bearing_by_id(bearing_id)
        return {k: v for k, v in brg.items() if k.startswith("f")}
    
    def calculate_brg_rdc_batch(
        self, brgs: List[Bearing], params_batch: np.ndarray, w_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # params_batch: (n_pop, n_brg, 2) -> [brg_id(int), cr_ratio(10~30)]
        n_cases, n_brg = params_batch.shape[:2]
        n_w = w_vec.shape[0]
        K_out = np.zeros((n_cases, n_brg, 4, n_w))
        C_out = np.zeros((n_cases, n_brg, 4, n_w))
        cache: Dict[Tuple[int, float, int], Tuple[np.ndarray, np.ndarray]] = {}

        for j in range(n_brg):
            Db, mu, load = brgs[j].Db, brgs[j].mu, brgs[j].load
            ids = params_batch[:, j, 0].astype(int)
            crs = params_batch[:, j, 1].astype(float)

            for i in range(n_cases):
                bid, cr = int(ids[i]), float(crs[i])
                key = (bid, cr, j)
                brg = self.get_bearing_by_id(bid)

                Cr = Db * cr * 1e-4
                Mp = float(brg['Mp'])
                Cp = Cr / (1.0 - Mp)
                stiff = load / Cp
                damp  = stiff / w_vec

                if key not in cache:
                    LD = float(brg['LD']); L = LD * Db
                    funcs = self.get_dynamic_coefficients(bid)
                    Kmat = np.zeros((4, n_w)); Cmat = np.zeros((4, n_w))

                    # S = μ ω L D^3 / (8π W C^2)
                    S = (mu * w_vec * L * Db**3) / (8.0 * np.pi * load * Cp**2)

                    # PchipInterpolator/PPoly는 벡터 입력 가능 → 루프 없이 평가
                    Kmat[0, :] = funcs['fKxx'](S); Kmat[1, :] = funcs['fKxy'](S)
                    Kmat[2, :] = funcs['fKyx'](S); Kmat[3, :] = funcs['fKyy'](S)
                    Cmat[0, :] = funcs['fCxx'](S); Cmat[1, :] = funcs['fCxy'](S)
                    Cmat[2, :] = funcs['fCyx'](S); Cmat[3, :] = funcs['fCyy'](S)

                    cache[key] = (Kmat, Cmat)

                Kmat, Cmat = cache[key]
                print(S)
                print(Kmat)
                
                K_out[i, j] = Kmat * stiff
                C_out[i, j] = Cmat * damp

        return K_out, C_out


# def save_pchip_group(h5file, group_path, named_series):
#     # named_series: dict{name: (x, y)}
#     with h5py.File(h5file, "a") as f:
#         g = f.require_group(group_path)
#         for name, (x, y) in named_series.items():
#             p = PchipInterpolator(np.asarray(x), np.asarray(y), extrapolate=True)
#             gsub = g.require_group(name)
#             if "breaks" in gsub: del gsub["breaks"]
#             if "coefs"  in gsub: del gsub["coefs"]
#             gsub.create_dataset("breaks", data=p.x)
#             gsub.create_dataset("coefs",  data=p.c.T)

# def load_pchip_from_group(h5file, group_path, name):
#     with h5py.File(h5file, "r") as f:
#         g = f[f"{group_path}/{name}"]
#         brk = np.asarray(g["breaks"]); cof = np.asarray(g["coefs"])
#         return PPoly(c=cof.T, x=brk, extrapolate=True)

# # 예: "brg_12/fKxx", "brg_12/fCxx" ...
# save_pchip_group(
#     "brg_fits.h5",
#     "brg_12",
#     {
#         "fKxx": (w_vec, Kxx_vals),
#         "fCxx": (w_vec, Cxx_vals),

#     }
# )

# fKxx_ppoly = load_pchip_from_group("brg_fits.h5", "brg_12", "fKxx")
# yq = fKxx_ppoly(wq)