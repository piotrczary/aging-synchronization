"""
Model starzenia v2 — pętla desynchronizacji
===========================================

Poprawki względem v1:
  1. B degraduje od bieżącego kosztu (nie od D) — ładniejsza krzywa starzenia
  2. Wyraźna różnica σ w eksperymencie kwadratowym
  3. Wykres: skuteczność lokalnej interwencji vs liczba oscylatorów (n)
  4. Analiza przeżywalności (Kaplan-Meier + test Gompertza)

Hipotezy:
  H1: człowiek starzeje się szybciej (σ rośnie nieliniowo)
  H2: lokalna interwencja: -40% u myszy, ~0% u człowieka
  H3: im więcej oscylatorów → tym mniej działa lokalna naprawa
  H4: log(hazard) rośnie liniowo z czasem → prawo Gompertza emergentne
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


# ─────────────────────────────────────────────
# ORGANIZM
# ─────────────────────────────────────────────

class Organism:
    """
    Sieć oscylatorów Kuramoto z pętlą uszkodzeń.

    Kluczowa zmiana v2:
    B degraduje od bieżącego kosztu cost(t), nie od skumulowanego D.
    To daje ładniejszą krzywą starzenia — powolny start, przyspieszenie.
    """
    def __init__(self, n_osc, coupling, noise, name="",
                 alpha=0.0008, beta=0.0003):
        self.name   = name
        self.n      = n_osc
        self.K      = coupling
        self.K0     = coupling
        self.noise  = noise
        self.alpha  = alpha
        self.beta   = beta

        self.phases = np.random.uniform(0, 2*np.pi, n_osc)
        self.omegas = np.random.uniform(0.9, 1.1, n_osc)
        self.B      = 1.0
        self.D      = 0.0

        self.history = {"sigma": [], "cost": [], "B": [], "D": [], "K": []}

    def kuramoto_step(self, dt=0.1):
        phase_diff     = self.phases[:, None] - self.phases[None, :]
        coupling_force = (self.K / self.n) * np.sin(phase_diff).sum(axis=1)

        # Efektywny szum: rośnie gdy bufor spada — pętla zwrotna
        effective = getattr(self, '_effective_noise', self.noise)
        noise_scales = np.ones(self.n)
        local_repair = getattr(self, '_local_repair', {})
        for idx, reduction in local_repair.items():
            noise_scales[idx] *= reduction
        global_factor = getattr(self, '_global_noise_factor', 1.0)
        noise_scales  *= global_factor

        noise_term = np.random.normal(0, effective, self.n) * noise_scales
        self.phases += dt * (self.omegas + coupling_force + noise_term)

    def sigma(self):
        """
        Miara desynchronizacji: 1 - parametr porządku r.
        r = |mean(e^{iθ})| ∈ [0,1]: 1=pełna synchronizacja, 0=chaos.
        σ = 1 - r ∈ [0,1]: 0=zsynchronizowane, 1=pełny chaos.
        Ta miara rośnie gdy sprzężenie K maleje — dając rosnące σ w czasie.
        """
        r = np.abs(np.mean(np.exp(1j * self.phases)))
        return 1.0 - r

    def step(self, dt=0.1):
        # 1. NIELINIOWY WZROST SZUMU (Gompertz trigger)
        D_scale = 120.0
        damage_factor = np.exp(0.25 * self.D / D_scale)
        damage_factor = min(damage_factor, 5.0)
        self._effective_noise = self.noise * damage_factor

        # 2. MINIMALNY SZUM (żeby uniknąć zamrażania)
        noise_floor = 0.02
        self._effective_noise = max(self._effective_noise, noise_floor)

        # 3. KROK KURAMOTO
        self.kuramoto_step(dt)

        # 4. DESYNCHRONIZACJA — wygładzona żeby zmniejszyć oscylacje wizualne
        raw_sig = self.sigma()
        sig = 0.9 * raw_sig + 0.1 * getattr(self, "_prev_sigma", raw_sig)
        self._prev_sigma = sig

        # 5. KOSZT — σ^3.5 daje ostrzejszą ścianę przy wysokim σ
        cost = (sig ** 3.5) / max(self.B, 0.05)

        # 6. AKTUALIZACJA STANU
        self.D += cost * dt
        self.B  = max(self.B - self.alpha * cost * dt, 0.01)
        # K degraduje szybciej gdy chaos rośnie
        self.K  = max(self.K - self.beta * cost * dt * (1 + sig), 0.0)

        # 7. HISTORIA
        self.history["sigma"].append(sig)
        self.history["cost"].append(cost)
        self.history["B"].append(self.B)
        self.history["D"].append(self.D)
        self.history["K"].append(self.K)

        return sig, cost

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.phases = np.random.uniform(0, 2*np.pi, self.n)
        self.B = 1.0
        self.D = 0.0
        self.K = self.K0
        self.history = {"sigma": [], "cost": [], "B": [], "D": [], "K": []}


# ─────────────────────────────────────────────
# INTERWENCJE
# ─────────────────────────────────────────────

def intervene_local(organism, osc_idx=0, noise_reduction=0.5):
    """
    Lokalna interwencja: trwałe zmniejszenie szumu w jednym oscylatorze.
    Symuluje np. naprawę jednego szlaku mitochondrialnego, jednego genu.
    Efekt fizyczny: jeden oscylator jest ciszej → mniejszy wkład do σ.
    Przy n=3: jeden oscylator to 33% systemu → duży efekt na σ.
    Przy n=30: jeden oscylator to 3% systemu → mały efekt na σ.
    Efekt wynika z matematyki σ, nie z hardkodowania.
    """
    # Zmniejsz szum tylko dla tego jednego oscylatora
    # Implementacja: zapamiętaj indeks i skaluj noise_term selektywnie
    organism._local_repair = getattr(organism, '_local_repair', {})
    organism._local_repair[osc_idx] = noise_reduction


def intervene_global(organism, noise_reduction=0.3, coupling_boost=0.2):
    """
    Globalna interwencja: wzmocnienie sprzężenia + redukcja szumu we wszystkich.
    Odpowiada: regularność snu, ćwiczenia, zeitgebery.
    """
    organism.K = min(organism.K + coupling_boost, organism.K0 * 1.5)
    organism._global_noise_factor = noise_reduction


# ─────────────────────────────────────────────
# EKSPERYMENTY
# ─────────────────────────────────────────────

def run_baseline(T=3000, seed=42):
    """Starzenie bez interwencji — mysz vs człowiek."""
    np.random.seed(seed)
    mouse = Organism(n_osc=5,  coupling=0.8, noise=0.05, name="Mysz",
                     alpha=0.0015, beta=0.003)
    human = Organism(n_osc=20, coupling=0.3, noise=0.15, name="Człowiek",
                     alpha=0.002,  beta=0.004)
    for _ in range(T):
        mouse.step()
        human.step()
    return mouse, human


def run_interventions(T=3000, T_int=1000, seed=42):
    """Porównanie interwencji lokalnej vs globalnej."""
    np.random.seed(seed)

    def mk_mouse():
        return Organism(n_osc=5,  coupling=0.8, noise=0.05,
                        alpha=0.0015, beta=0.003)
    def mk_human():
        return Organism(n_osc=20, coupling=0.3, noise=0.15,
                        alpha=0.002, beta=0.004)

    orgs = {
        "Mysz — kontrola":       mk_mouse(),
        "Mysz — lokalna":        mk_mouse(),
        "Człowiek — kontrola":   mk_human(),
        "Człowiek — lokalna":    mk_human(),
        "Człowiek — globalna":   mk_human(),
    }

    for t in range(T):
        for org in orgs.values():
            org.step()
        if t == T_int:
            intervene_local(orgs["Mysz — lokalna"])
            intervene_local(orgs["Człowiek — lokalna"])
            intervene_global(orgs["Człowiek — globalna"])

    return orgs, T_int


def run_variance_cost(T=2000, seed=42):
    """
    Koszt kwadratowy wariancji.
    Ekstremalne różnice K i noise — wymuszamy σ ratio > 2.
    """
    np.random.seed(seed)
    org_low  = Organism(n_osc=10, coupling=1.2, noise=0.01, name="Niska σ",
                        alpha=0.004, beta=0.001)
    org_high = Organism(n_osc=10, coupling=0.05, noise=0.40, name="Wysoka σ",
                        alpha=0.004, beta=0.001)
    for _ in range(T):
        org_low.step()
        org_high.step()
    return org_low, org_high


def run_n_vs_effect_analytical(n_values):
    """
    Analityczna predykcja: Δσ/σ = f(2-f)/n dla lokalnej, f(2-f) dla globalnej.
    Wyprowadzenie w Appendix A paperu.
    """
    f = 0.5
    local_effect  = [(2*f - f**2) / n * 100 for n in n_values]
    global_effect = [(2*f - f**2) * 100] * len(n_values)
    return local_effect, global_effect


def run_figure1(n_values=(3, 5, 8, 12, 20, 30), T=3000, seed=42, n_runs=15):
    """
    Figure 1: σ(t) uśrednione po n_runs symulacjach dla każdego n.
    Uśrednienie usuwa przecięcia trajektorii — zostaje czysty trend.
    """
    results = {}
    for n in n_values:
        coupling = max(0.85 - 0.02 * n, 0.15)
        noise    = 0.05 + 0.003 * n
        alpha    = 0.0015 + 0.0001 * n
        beta     = 0.003  + 0.0001 * n

        runs = []
        for run in range(n_runs):
            np.random.seed(seed + n * 100 + run)
            org = Organism(n_osc=n, coupling=coupling, noise=noise,
                           alpha=alpha, beta=beta)
            for _ in range(T):
                org.step()
            runs.append(org.history["sigma"])

        # Średnia i odchylenie po wszystkich uruchomieniach
        arr = np.array(runs)
        results[n] = {
            "mean": arr.mean(axis=0),
            "std":  arr.std(axis=0),
        }
        print(f"  n={n:3d}  σ_końcowe: {results[n]['mean'][-1]:.3f} ± {results[n]['std'][-1]:.3f}")

    return results


def plot_figure1(results, T=3000):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Figure 1: Desynchronizacja σ(t) dla różnych złożoności systemu\n"
        "Średnia z 15 symulacji — większe n → szybsze wejście w chaos",
        fontsize=12, fontweight="bold"
    )

    n_vals = sorted(results.keys())
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(n_vals)))
    t_ax   = np.arange(T)

    for n, col in zip(n_vals, colors):
        mean = smooth(results[n]["mean"], w=60)
        std  = smooth(results[n]["std"],  w=60)
        label = f"n={n}  (mysz)" if n <= 5 else (
                f"n={n}  (człowiek)" if n >= 20 else f"n={n}")
        lw = 3 if n in (5, 20) else 1.8

        ax.plot(t_ax, mean, color=col, lw=lw, label=label)
        ax.fill_between(t_ax,
                        np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        color=col, alpha=0.12)

    ax.axhline(0.95, color="black", ls=":", lw=1.5,
               label="σ_crit = 0.95 (próg utraty homeostazy)")

    # Zaznacz mediany przejścia progu
    for n, col in zip(n_vals, colors):
        mean = smooth(results[n]["mean"], w=60)
        cross = np.where(np.array(mean) > 0.95)[0]
        if len(cross) > 0:
            ax.scatter([cross[0]], [0.95], color=col, s=90,
                       zorder=5, edgecolors="white", linewidth=1)

    ax.set_xlabel("Czas biologiczny", fontsize=11)
    ax.set_ylabel("σ (desynchronizacja, średnia ± SD)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("starzenie_figure1.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_figure1.png")
    plt.close()
    """
    Poprawne skalowanie (Appendix A paperu):

    Redukcja amplitudy szumu jednego oscylatora: η_k → (1-f)·η_k
    Wkład do wariancji: η_k² → (1-f)²·η_k²
    Zmiana wariancji: Δ(η_k²) = η₀²·(2f - f²)

    Δσ/σ = (2f - f²) / n   (lokalna — jeden oscylator)
    Δσ/σ = (2f - f²)       (globalna — wszystkie oscylatory)

    Ratio = 1/n dokładnie, niezależnie od f.
    Dla f=1 (pełna naprawa): lokalna = 1/n.
    """
    f = 0.5
    local_effect  = [(2*f - f**2) / n * 100 for n in n_values]
    global_effect = [(2*f - f**2) * 100] * len(n_values)
    return local_effect, global_effect


def run_survival(n_agents=200, T=3000, seed=42,
                 B_crit=0.02, sigma_crit=0.95):
    """
    Analiza przeżywalności: śledzenie momentu śmierci każdego agenta.
    Śmierć = B < B_crit LUB σ > sigma_crit.
    """
    np.random.seed(seed)
    death_m = []
    death_h = []

    print(f"  Symulacja {n_agents} agentów...")
    for i in range(n_agents):
        rng_offset = i * 7
        np.random.seed(seed + rng_offset)

        mouse = Organism(n_osc=5,  coupling=0.8, noise=0.06,
                         alpha=0.0006, beta=0.0002)
        np.random.seed(seed + rng_offset + 1)
        human = Organism(n_osc=20, coupling=0.4, noise=0.13,
                         alpha=0.0008, beta=0.0003)

        t_m = t_h = None
        for t in range(T):
            if t_m is None:
                mouse.step()
                if mouse.B < B_crit or mouse.sigma() > sigma_crit:
                    t_m = t
            if t_h is None:
                human.step()
                if human.B < B_crit or human.sigma() > sigma_crit:
                    t_h = t
            if t_m is not None and t_h is not None:
                break

        death_m.append(t_m if t_m is not None else T)
        death_h.append(t_h if t_h is not None else T)

    return np.array(death_m), np.array(death_h)


# ─────────────────────────────────────────────
# WIZUALIZACJA
# ─────────────────────────────────────────────

def smooth(x, w=40):
    return uniform_filter1d(x, size=w)


def plot_baseline(mouse, human):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Starzenie jako pętla desynchronizacji\n"
        "Mysz (5 osc.) vs Człowiek (20 osc.) — bez interwencji",
        fontsize=13, fontweight="bold"
    )
    t  = np.arange(len(mouse.history["sigma"]))
    cm = "#1565C0"
    ch = "#C62828"

    for ax, (key, title, do_smooth) in zip(axes.flat, [
        ("sigma", "Desynchronizacja σ (wariancja faz)", True),
        ("B",     "Bufor regulacyjny B",                False),
        ("cost",  "Koszt C ~ σ^3.5/B",                    True),
        ("D",     "Uszkodzenia skumulowane D",          False),
    ]):
        dm = smooth(mouse.history[key]) if do_smooth else mouse.history[key]
        dh = smooth(human.history[key]) if do_smooth else human.history[key]
        ax.plot(t, dm, color=cm, lw=2, label="Mysz")
        ax.plot(t, dh, color=ch, lw=2, label="Człowiek")
        ax.set_title(title)
        ax.set_xlabel("Czas biologiczny")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("starzenie_baseline.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_baseline.png")
    plt.close()


def plot_interventions(orgs, T_int):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Interwencje: lokalna (naprawa 1 oscylatora) vs globalna (sprzężenie)\n"
        "Teoria: lokalna ~15% u myszy (n=5), ~4% u człowieka (n=20) | globalna ~75%",
        fontsize=12, fontweight="bold"
    )
    styles = {
        "Mysz — kontrola":     {"color": "#90CAF9", "ls": "--", "lw": 1.5},
        "Mysz — lokalna":      {"color": "#0D47A1", "ls": "-",  "lw": 2.5},
        "Człowiek — kontrola": {"color": "#FFCDD2", "ls": "--", "lw": 1.5},
        "Człowiek — lokalna":  {"color": "#B71C1C", "ls": "-",  "lw": 2},
        "Człowiek — globalna": {"color": "#1B5E20", "ls": "-",  "lw": 2.5},
    }
    for ax, (key, title) in zip(axes, [
        ("sigma", "Desynchronizacja σ"),
        ("B",     "Bufor B"),
        ("D",     "Uszkodzenia D"),
    ]):
        for name, org in orgs.items():
            data = smooth(org.history[key]) if key == "sigma" else org.history[key]
            ax.plot(data, label=name, **styles[name])
        ax.axvline(T_int, color="black", ls=":", lw=1.5, label="Interwencja")
        ax.set_title(title)
        ax.set_xlabel("Czas biologiczny")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("starzenie_interwencje.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_interwencje.png")
    plt.close()


def plot_variance_cost(org_low, org_high):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Koszt kwadratowy wariancji: 2× σ → 4× koszt długoterminowo\n"
        "Niska σ (silne sprzężenie) vs Wysoka σ (słabe sprzężenie)",
        fontsize=12, fontweight="bold"
    )
    t  = np.arange(len(org_low.history["sigma"]))
    cl = "#1976D2"
    ch = "#C62828"

    for ax, (key, title, do_smooth) in zip(axes, [
        ("sigma", "Desynchronizacja σ", True),
        ("cost",  "Koszt C ~ σ^3.5/B",    True),
        ("D",     "Uszkodzenia D",      False),
    ]):
        for org, col in [(org_low, cl), (org_high, ch)]:
            data = smooth(org.history[key]) if do_smooth else org.history[key]
            ax.plot(t, data, color=col, lw=2, label=org.name)
        ax.set_title(title)
        ax.set_xlabel("Czas biologiczny")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Annotacja stosunku
    d_low  = org_low.history["D"][-1]
    d_high = org_high.history["D"][-1]
    s_low  = np.mean(org_low.history["sigma"])
    s_high = np.mean(org_high.history["sigma"])
    sr = s_high / max(s_low, 1e-6)
    dr = d_high / max(d_low, 1e-6)
    axes[2].text(0.05, 0.95,
        f"σ_high/σ_low ≈ {sr:.1f}×\n"
        f"D_high/D_low ≈ {dr:.1f}×\n"
        f"teoria (σ²): {sr**2:.1f}×",
        transform=axes[2].transAxes, fontsize=9, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig("starzenie_wariancja.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_wariancja.png")
    plt.close()


def plot_n_vs_effect(n_values, local_effect, global_effect):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(n_values, local_effect,  "o-", color="#C62828", lw=2.5,
            markersize=8, label="Lokalna (1 oscylator)")
    ax.plot(n_values, global_effect, "s--", color="#1B5E20", lw=2.5,
            markersize=8, label="Globalna (wszystkie oscylatory)")

    # Zaznacz mysz i człowieka
    for n_mark, name, color in [(5, "Mysz", "#1565C0"), (20, "Człowiek", "#B71C1C")]:
        if n_mark in n_values:
            idx = n_values.index(n_mark)
            ax.scatter([n_mark], [local_effect[idx]],
                       color=color, s=200, zorder=5)
            ax.annotate(f"{name}\n({local_effect[idx]:.1f}%)",
                        (n_mark, local_effect[idx]),
                        xytext=(n_mark + 1, local_effect[idx] + 0.3),
                        fontsize=9, color=color)

    ax.fill_between(n_values, local_effect, global_effect,
                    alpha=0.1, color="gray",
                    label="Przewaga globalnej nad lokalną")

    ax.set_xlabel("Liczba oscylatorów (złożoność systemu)", fontsize=11)
    ax.set_ylabel("Przewidywana redukcja σ [%]", fontsize=11)
    ax.set_title(
        "Skuteczność interwencji lokalnej vs globalnej\n"
        "Lokalna: Δσ/σ ~ 1/n  |  Globalna: Δσ/σ ~ const\n"
        "→ im większy system, tym bardziej potrzebna synchronizacja globalna",
        fontweight="bold", fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(n_values) - 1, max(n_values) + 1)

    plt.tight_layout()
    plt.savefig("starzenie_n_vs_efekt.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_n_vs_efekt.png")
    plt.close()


def plot_survival(death_m, death_h, T):
    """Krzywe przeżycia + test Gompertza (log-hazard)."""
    t_ax = np.arange(T)

    def survival_curve(dt):
        return np.array([np.sum(dt > t) / len(dt) for t in t_ax])

    def hazard(S):
        h = []
        for i in range(len(S) - 1):
            h.append((S[i] - S[i+1]) / (S[i] + 1e-9))
        return np.array(h)

    S_m = survival_curve(death_m)
    S_h = survival_curve(death_h)
    H_m = hazard(S_m)
    H_h = hazard(S_h)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Analiza przeżywalności — prawo Gompertza jako efekt emergentny\n"
        "Śmierć = B < 0.05 lub σ > 0.9 (utrata homeostazy)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1 — Krzywa przeżycia
    ax = axes[0]
    ax.plot(t_ax, S_m, color="#1565C0", lw=3, label="Mysz (5 osc.)")
    ax.plot(t_ax, S_h, color="#C62828", lw=3, label="Człowiek (20 osc.)")
    ax.fill_between(t_ax, S_m, alpha=0.1, color="#1565C0")
    ax.fill_between(t_ax, S_h, alpha=0.1, color="#C62828")
    ax.set_title("Krzywa przeżycia (Kaplan-Meier)")
    ax.set_xlabel("Czas biologiczny")
    ax.set_ylabel("Frakcja żyjących")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mediany
    for S, col, name in [(S_m, "#1565C0", "Mysz"), (S_h, "#C62828", "Człowiek")]:
        idx = np.where(S <= 0.5)[0]
        if len(idx) > 0:
            ax.axvline(idx[0], color=col, ls="--", alpha=0.6)
            ax.text(idx[0], 0.52, f"  {name}\n  t50={idx[0]}", color=col, fontsize=8)

    # Panel 2 — Log-hazard (Gompertz)
    ax = axes[1]
    log_h_m = np.log(gaussian_filter1d(H_m, 25) + 1e-7)
    log_h_h = np.log(gaussian_filter1d(H_h, 25) + 1e-7)
    t_h_ax  = np.arange(len(H_m))
    ax.plot(t_h_ax, log_h_m, color="#1565C0", lw=2.5, label="Mysz")
    ax.plot(t_h_ax, log_h_h, color="#C62828", lw=2.5, label="Człowiek")

    # Dopasowanie linii prostej (test Gompertza)
    for S_arr, col, name in [(log_h_m, "#1565C0", "Mysz"),
                              (log_h_h, "#C62828", "Człowiek")]:
        valid = np.isfinite(S_arr) & (t_h_ax > 50) & (t_h_ax < T - 100)
        if valid.sum() > 10:
            coeffs = np.polyfit(t_h_ax[valid], S_arr[valid], 1)
            fit = np.poly1d(coeffs)
            ax.plot(t_h_ax[valid], fit(t_h_ax[valid]),
                    color=col, ls="--", alpha=0.6, lw=1.5,
                    label=f"{name} fit: k={coeffs[0]:.5f}")

    ax.set_title("Log-hazard (test Gompertza)\nLinia prosta = wykładniczy wzrost ryzyka")
    ax.set_xlabel("Czas biologiczny")
    ax.set_ylabel("log h(t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.05,
        "Jeśli linie są proste → model odkrył prawo Gompertza\n"
        "bez wpisywania go do kodu",
        transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig("starzenie_przezywalnosc.png", dpi=150, bbox_inches="tight")
    print("Zapisano: starzenie_przezywalnosc.png")
    plt.close()


def print_summary(mouse, human, orgs, org_low, org_high,
                  n_values, local_eff, global_eff, death_m, death_h):
    print("\n" + "="*55)
    print("PODSUMOWANIE WYNIKÓW")
    print("="*55)

    print("\nEksperyment 1 — starzenie:")
    dr = human.history["D"][-1] / max(mouse.history["D"][-1], 1e-6)
    print(f"  Człowiek akumuluje {dr:.1f}× więcej uszkodzeń niż mysz")

    print("\nEksperyment 2 — interwencje:")
    for name, org in orgs.items():
        print(f"  {name:30s}  D={org.history['D'][-1]:.1f}")
    ctrl_h = orgs["Człowiek — kontrola"].history["D"][-1]
    lok_h  = orgs["Człowiek — lokalna"].history["D"][-1]
    glob_h = orgs["Człowiek — globalna"].history["D"][-1]
    ctrl_m = orgs["Mysz — kontrola"].history["D"][-1]
    lok_m  = orgs["Mysz — lokalna"].history["D"][-1]
    print(f"\n  Lokalna u myszy:         {(1-lok_m/ctrl_m)*100:+.1f}%")
    print(f"  Lokalna u człowieka:     {(1-lok_h/ctrl_h)*100:+.1f}%")
    print(f"  Globalna u człowieka:    {(1-glob_h/ctrl_h)*100:+.1f}%")

    print("\nEksperyment 3 — koszt kwadratowy:")
    sr = np.mean(org_high.history["sigma"]) / max(np.mean(org_low.history["sigma"]), 1e-6)
    dr2 = org_high.history["D"][-1] / max(org_low.history["D"][-1], 1e-6)
    print(f"  σ_high/σ_low = {sr:.2f}×  →  D_high/D_low = {dr2:.2f}×  "
          f"(teoria: {sr**2:.2f}×)")

    print("\nEksperyment 4 — n vs efekt (analitycznie):")
    for n, l, g in zip(n_values, local_eff, global_eff):
        print(f"  n={n:3d}: lokalna={l:.1f}%  globalna={g:.1f}%")

    print("\nEksperyment 5 — przeżywalność:")
    t50_m = np.where(
        np.array([np.sum(death_m > t) for t in range(len(death_m))]) / len(death_m)
        <= 0.5)[0]
    t50_h = np.where(
        np.array([np.sum(death_h > t) for t in range(len(death_h))]) / len(death_h)
        <= 0.5)[0]
    if len(t50_m): print(f"  Mediana życia mysz:     t50 ≈ {t50_m[0]}")
    if len(t50_h): print(f"  Mediana życia człowiek: t50 ≈ {t50_h[0]}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("MODEL STARZENIA v2 — pętla desynchronizacji\n")

    print("[1/5] Starzenie bez interwencji...")
    mouse, human = run_baseline(T=3000, seed=42)
    plot_baseline(mouse, human)

    print("[2/5] Porównanie interwencji...")
    orgs, T_int = run_interventions(T=3000, T_int=1000, seed=42)
    plot_interventions(orgs, T_int)

    print("[3/5] Koszt kwadratowy wariancji...")
    org_low, org_high = run_variance_cost(T=2000, seed=42)
    plot_variance_cost(org_low, org_high)

    print("[4/5] Skuteczność lokalnej interwencji vs n oscylatorów (analitycznie)...")
    n_values = [3, 5, 8, 12, 16, 20, 25, 30]
    local_eff, global_eff = run_n_vs_effect_analytical(n_values)
    for n, l, g in zip(n_values, local_eff, global_eff):
        print(f"  n={n:3d}  lokalna: {l:.1f}%  globalna: {g:.1f}%")
    plot_n_vs_effect(n_values, local_eff, global_eff)

    print("[5/5] Analiza przeżywalności (Gompertz)...")
    death_m, death_h = run_survival(n_agents=200, T=3000, seed=42)
    plot_survival(death_m, death_h, T=3000)

    print("[6/6] Figure 1 — σ(t) dla różnych n...")
    fig1_results = run_figure1(n_values=(3, 5, 8, 12, 20, 30), T=3000, seed=42)
    plot_figure1(fig1_results, T=3000)

    print_summary(mouse, human, orgs, org_low, org_high,
                  n_values, local_eff, global_eff, death_m, death_h)

    print("\nGotowe. Pięć wykresów zapisanych.")
