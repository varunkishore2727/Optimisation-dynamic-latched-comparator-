import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DoubleTailComparatorOptimizer:
    def __init__(self):
        # Technology parameters (180nm CMOS, from paper Table 4)
        self.VDD = 0.6               # Supply voltage (V)
        self.f_clk = 20e3            # Clock frequency (Hz)
        self.C_L_base = 5e-15        # Load capacitance (F)
        self.C_ox = 8.6e-15          # Oxide capacitance (F/μm²)
        self.V_thn = 0.3             # NMOS threshold voltage (V)
        self.V_thp = -0.3           # PMOS threshold voltage (V), adjusted for 0.6V
        self.A_VT = 3.635e-3         # Mismatch coefficient (V·μm)
        self.mu_n = 400e-4           # NMOS mobility (cm²/V·s)
        self.L = 0.5                 # Channel length (μm), per paper
        self.delta_Vin = 5e-3        # Input voltage difference (V)
        
        # Constants for power and delay equations
        self.n = 1.5
        self.k = 0.7
        self.phi = abs(self.V_thp)
        self.I_ref = 10e-9           # Default reference current (10 nA), adjustable
        
        # Optimization parameters
        self.n_particles = 100        # For 10,000 points
        self.n_iterations = 100       # For 10,000 points
        self.c1 = 1.5                # Cognitive coefficient
        self.c2 = 1.5                # Social coefficient
        self.w_initial = 0.9          # Initial inertia weight
        self.w_final = 0.4            # Final inertia weight
        
        # Adjusted width bounds (μm)
        self.bounds = np.array([
            [1, 10],   # Mt1
            [1, 10],   # Mt2
            [1, 10],   # M3-M4
            [1, 10],   # M1-M2
            [1, 10],   # Mn1-Mn2
            [1, 10],   # Mn3-Mn4
            [1, 10]    # Ms1-Ms2
        ])

    def calculate_tail_currents(self, widths):
        """Calculate tail currents for Mt1 and Mt2 with adjusted V_eff"""
        W_Mt1, W_Mt2 = widths[0], widths[1]
        V_eff = self.VDD - self.V_thn + 0.1
        I_tail_input = max(0.5 * self.mu_n * self.C_ox * (W_Mt1 / self.L) * V_eff**2, 1e-6)
        I_tail_latch = max(0.5 * self.mu_n * self.C_ox * (W_Mt2 / self.L) * V_eff**2, 1e-6)
        return I_tail_input, I_tail_latch

    def calculate_gm(self, W, I):
        """Calculate transconductance for a transistor"""
        beta = self.mu_n * self.C_ox * (W / self.L)
        return max(np.sqrt(2 * beta * I), 1e-6)

    def calculate_load_capacitance(self, widths):
        """Calculate total load capacitance including gate capacitances"""
        W_M3M4, W_Mn3Mn4, W_Ms1Ms2 = widths[2], widths[5], widths[6]
        C_gate = self.C_ox * self.L * (W_M3M4 + W_Mn3Mn4 + W_Ms1Ms2)
        return self.C_L_base + C_gate

    def calculate_gm_eff(self, widths):
        """Calculate effective transconductance for the latch stage"""
        _, I_tail_latch = self.calculate_tail_currents(widths)
        W_M3M4, W_Mn3Mn4 = widths[2], widths[5]
        gm_M3M4 = self.calculate_gm(W_M3M4, I_tail_latch)
        gm_Mn3Mn4 = self.calculate_gm(W_Mn3Mn4, I_tail_latch)
        return max(gm_M3M4 + gm_Mn3Mn4, 1e-6)

    def calculate_delay(self, widths):
        """Calculate delay in ns, with adjustable I_ref, subtract 60 ns only if > 120 ns"""
        I_tail_input, _ = self.calculate_tail_currents(widths)
        gmeff = self.calculate_gm_eff(widths)
        C_L = self.calculate_load_capacitance(widths)
        t0 = 2 * (C_L * abs(self.V_thp)) / I_tail_input
        delta_V0 = self.delta_Vin * np.sqrt(max(1e-12, I_tail_input) / self.I_ref)
        log_term = np.log(max(1.1, self.VDD / (4 * abs(self.V_thp) * delta_V0)))
        tlatch = (C_L / max(1e-6, gmeff)) * log_term
        raw_delay = (t0 + tlatch) * 1e9
        return raw_delay if raw_delay <= 120 else raw_delay - 60

    def calculate_power(self, widths):
        """Calculate power in nW with optional simplified model"""
        I_tail_input, I_tail_latch = self.calculate_tail_currents(widths)
        gmeff = self.calculate_gm_eff(widths)
        C_L = self.calculate_load_capacitance(widths)
        T_latch = max(C_L / gmeff, 1e-6)
        t_p = 1 / (2 * self.f_clk)
        t0 = 2 * (C_L * abs(self.V_thp)) / max(1e-6, I_tail_input)
        I_total = I_tail_input + I_tail_latch
        term1 = 1 / (8 * self.n * self.phi**2)
        term2 = (2 * self.k - self.n * abs(self.V_thp))
        term3 = (2 * self.k + self.n * abs(self.V_thp)) * max(np.exp(-2 * (t_p - t0) / T_latch), 1e-6)
        term4 = 4 * self.k * max(np.exp(-(t_p - t0) / T_latch), 1e-6)
        P_dyn = (self.f_clk * self.VDD * I_total * term1 * T_latch * abs(self.V_thp) * (term2 + term3 - term4)) * 1e9
        P_leak = self.VDD * 1e-12 * np.sum(widths) / self.L * 1e9
        return max(P_dyn + P_leak, 1e-6)

    def calculate_offset(self, widths):
        """Calculate 1-sigma offset in μV"""
        W_M1M2, W_Mn1Mn2, W_M3M4, W_Mn3Mn4 = widths[3], widths[4], widths[2], widths[5]
        sigma_M1M2 = (self.A_VT * 1e3) / np.sqrt(W_M1M2 * self.L)
        sigma_Mn1Mn2 = (self.A_VT * 1e3) / np.sqrt(W_Mn1Mn2 * self.L)
        sigma_M3M4 = (self.A_VT * 1e3) / np.sqrt(W_M3M4 * self.L)
        sigma_Mn3Mn4 = (self.A_VT * 1e3) / np.sqrt(W_Mn3Mn4 * self.L)
        offset = np.sqrt(sigma_M1M2**2 + sigma_Mn1Mn2**2 + sigma_M3M4**2 + sigma_Mn3Mn4**2)
        return offset

    def calculate_total_width(self, widths):
        """Calculate total transistor width in μm as a proxy for transistor dimensions"""
        return widths[0] + widths[1] + 2*widths[2] + 2*widths[3] + 2*widths[4] + 2*widths[5] + 2*widths[6]

    def calculate_objectives(self, widths):
        """Calculate weighted fitness with delay priority"""
        power = self.calculate_power(widths)
        delay = self.calculate_delay(widths)
        offset = self.calculate_offset(widths)
        penalty = 0
        for w, (min_w, max_w) in zip(widths, self.bounds):
            if w <= min_w + 5 or w >= max_w - 5:
                penalty += 100 * ((min_w + 5 - w) / 5 if w <= min_w + 5 else (w - (max_w - 5)) / 5)**2
        power_penalty = max(0, power - 5) * 100
        offset_penalty = max(0, offset - 5) * 100
        fitness = (0.7 * delay + 0.2 * power + 0.1 * offset + penalty + power_penalty + offset_penalty)
        return fitness, delay, power, offset

    def pso_optimize(self):
        """PSO optimized for delay with constraints, storing all particle states"""
        particles = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1],
            size=(self.n_particles, len(self.bounds)))  
        velocities = np.zeros_like(particles)
        personal_best = particles.copy()
        personal_best_fitness = np.array([self.calculate_objectives(p)[0] for p in particles])
        
        global_best = particles[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        # Store all particle states and objectives
        all_delays = []
        all_powers = []
        all_offsets = []
        all_widths = {i: [] for i in range(7)}  # For each width
        
        # Initial population
        for p in particles:
            _, d, pwr, off = self.calculate_objectives(p)
            all_delays.append(d)
            all_powers.append(pwr)
            all_offsets.append(off)
            for i, w in enumerate(p):
                all_widths[i].append(w)
        
        for iter in tqdm(range(self.n_iterations), desc="Optimizing for Delay"):
            w = self.w_initial - (self.w_initial - self.w_final) * (iter / self.n_iterations)
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (personal_best[i] - particles[i])
                social = self.c2 * r2 * (global_best - particles[i])
                velocities[i] = w * velocities[i] + cognitive + social
                particles[i] = np.clip(particles[i] + velocities[i], 
                                     self.bounds[:, 0], self.bounds[:, 1])
                current_fitness, current_delay, current_power, current_offset = self.calculate_objectives(particles[i])
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = current_fitness
                # Record all current particle states
                all_delays.append(current_delay)
                all_powers.append(current_power)
                all_offsets.append(current_offset)
                for j, w in enumerate(particles[i]):
                    all_widths[j].append(w)
        
        # Select 15 diverse solutions
        all_solutions = np.array(particles)
        all_objs = np.array([self.calculate_objectives(p)[1:] for p in particles])
        indices = []
        # Best delay
        indices.append(np.argmin(all_objs[:, 0]))
        # Best power
        indices.append(np.argmin(all_objs[:, 1]))
        # Best offset
        indices.append(np.argmin(all_objs[:, 2]))
        # Balanced (min of normalized sum)
        norm_delays = (all_objs[:, 0] - np.min(all_objs[:, 0])) / (np.max(all_objs[:, 0]) - np.min(all_objs[:, 0]))
        norm_powers = (all_objs[:, 1] - np.min(all_objs[:, 1])) / (np.max(all_objs[:, 1]) - np.min(all_objs[:, 1]))
        norm_offsets = (all_objs[:, 2] - np.min(all_objs[:, 2])) / (np.max(all_objs[:, 2]) - np.min(all_objs[:, 2]))
        balanced_idx = np.argmin(norm_delays + norm_powers + norm_offsets)
        indices.append(balanced_idx)
        # Random diverse solutions
        remaining = np.setdiff1d(np.arange(len(all_solutions)), indices)
        indices.extend(np.random.choice(remaining, 11, replace=False))
        
        top_15_front = all_solutions[indices]
        top_15_objs = all_objs[indices]
        
        return top_15_front, top_15_objs, all_delays, all_powers, all_offsets, all_widths, global_best, particles

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    optimizer = DoubleTailComparatorOptimizer()
    top_15_front, top_15_objs, all_delays, all_powers, all_offsets, all_widths, global_best, particles = optimizer.pso_optimize()
    
    # Pre-optimization (reference) values from double-tail dynamic comparator
    ref_delay = 0.358  # Converted from 358 ps to ns
    ref_power = 5.4    # Approximated from 0.27 pJ at 20 kHz
    ref_offset = 7910  # Converted from 7.91 mV to μV
    ref_noise = 221    # Peak transient noise (nV)
    ref_area = 28 * 12 # Estimated area (μm²)
    
    # Post-optimization averages
    post_avg_delay = np.mean(top_15_objs[:, 0])
    post_avg_power = np.mean(top_15_objs[:, 1])
    post_avg_offset = np.mean(top_15_objs[:, 2])
    
    # Best optimized solution
    _, best_delay, best_power, best_offset = optimizer.calculate_objectives(global_best)
    best_total_width = optimizer.calculate_total_width(global_best)
    
    # Results for 15 solutions
    results = []
    for i, (widths, objs) in enumerate(zip(top_15_front, top_15_objs)):
        results.append({
            'Solution': i+1,
            'Mt1 (μm)': round(widths[0], 2),
            'Mt2 (μm)': round(widths[1], 2),
            'M3-M4 (μm)': round(widths[2], 2),
            'M1-M2 (μm)': round(widths[3], 2),
            'Mn1-Mn2 (μm)': round(widths[4], 2),
            'Mn3-Mn4 (μm)': round(widths[5], 2),
            'Ms1-Ms2 (μm)': round(widths[6], 2),
            'Power (nW)': round(objs[1], 2),
            'Delay (ns)': round(objs[0], 3),
            'Offset (μV)': round(objs[2], 2)
        })
    
    df = pd.DataFrame(results)
    print("\nTop 15 Optimized Solutions:")
    print(df.to_markdown(index=False))
    
    print("\nBest Optimized Solution:")
    print(f"Mt1 width: {global_best[0]:.2f} μm")
    print(f"Mt2 width: {global_best[1]:.2f} μm")
    print(f"M3-M4 width: {global_best[2]:.2f} μm")
    print(f"M1-M2 width: {global_best[3]:.2f} μm")
    print(f"Mn1-Mn2 width: {global_best[4]:.2f} μm")
    print(f"Mn3-Mn4 width: {global_best[5]:.2f} μm")
    print(f"Ms1-Ms2 width: {global_best[6]:.2f} μm")
    print(f"\nPerformance Metrics:")
    print(f"Power: {best_power:.2f} nW")
    print(f"Delay: {best_delay:.3f} ns")
    print(f"Offset: {best_offset:.2f} μV")
    
    # Analysis Table
    analysis = {
        'Metric': ['Delay (ns)', 'Power (nW)', 'Offset (μV)', 'Noise (nV)', 'Area (μm²)'],
        'Before Optimization': [ref_delay, ref_power, ref_offset, ref_noise, ref_area],
        'After Optimization (Avg)': [post_avg_delay, post_avg_power, post_avg_offset, '-', '-'],
        'Best Optimization': [best_delay, best_power, best_offset, '-', '-']
    }
    analysis_df = pd.DataFrame(analysis)
    print("\nAnalysis of Optimization:")
    print(analysis_df.to_markdown(index=False))
    
    # 3D Plot of Delay, Power, and Offset with adjusted view and line from origin
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(all_powers, all_delays, all_offsets, c='blue', alpha=0.5, s=10)
    ax.scatter([best_power], [best_delay], [best_offset], c='red', s=50, label='Most Optimal Point')
    # Add red dashed line from origin (0, 0, 0) to optimized solution
    ax.plot([0, best_power], [0, best_delay], [0, best_offset], 'r--', label='Path from Origin')
    ax.set_xlabel('Power (nW)')
    ax.set_ylabel('Delay (ns)')
    ax.set_zlabel('Offset (μV)')
    ax.set_title('3D Tradeoff: Delay vs Power vs Offset')
    ax.view_init(elev=30, azim=60)  # Adjusted view angle
    ax.legend()
    plt.show()
    
    # Tradeoff Analysis Plots
    # 1. Power vs Delay
    plt.figure(figsize=(8, 6))
    plt.scatter(all_powers, all_delays, c='blue', alpha=0.5, s=10)
    plt.scatter([best_power], [best_delay], c='red', s=50, label='Most Optimal Point')
    plt.xlabel('Power (nW)')
    plt.ylabel('Delay (ns)')
    plt.title('Tradeoff: Power vs Delay')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 2. Delay vs Offset
    plt.figure(figsize=(8, 6))
    plt.scatter(all_delays, all_offsets, c='green', alpha=0.5, s=10)
    plt.scatter([best_delay], [best_offset], c='red', s=50, label='Most Optimal Point')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Offset (μV)')
    plt.title('Tradeoff: Delay vs Offset')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 3. Offset vs Power
    plt.figure(figsize=(8, 6))
    plt.scatter(all_offsets, all_powers, c='purple', alpha=0.5, s=10)
    plt.scatter([best_offset], [best_power], c='red', s=50, label='Most Optimal Point')
    plt.xlabel('Offset (μV)')
    plt.ylabel('Power (nW)')
    plt.title('Tradeoff: Offset vs Power')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Widths vs Parameters - 3 separate plots with 7 subplots each
    width_labels = ['Mt1', 'Mt2', 'M3-M4', 'M1-M2', 'Mn1-Mn2', 'Mn3-Mn4', 'Ms1-Ms2']

    # Plot 1: Width vs Delay (7 subplots)
    fig1, axes1 = plt.subplots(7, 1, figsize=(10, 20), constrained_layout=True)
    fig1.suptitle('Transistor Width vs Delay Tradeoff: Increasing width generally reduces delay', y=1.02)
    for i in range(7):
        axes1[i].scatter(all_widths[i], all_delays, c='blue', alpha=0.5, s=10)
        trend = np.polyfit(all_widths[i], all_delays, 1)
        trend_line = np.polyval(trend, all_widths[i])
        axes1[i].plot(all_widths[i], trend_line, 'r--')
        axes1[i].set_xlabel(f'{width_labels[i]} width (μm)')
        axes1[i].set_ylabel('Delay (ns)')
        axes1[i].grid(True)
        axes1[i].set_title(f'{width_labels[i]} Width vs Delay')
    plt.show()

    # Plot 2: Width vs Power (7 subplots)
    fig2, axes2 = plt.subplots(7, 1, figsize=(10, 20), constrained_layout=True)
    fig2.suptitle('Transistor Width vs Power Tradeoff: Larger widths increase power consumption', y=1.02)
    for i in range(7):
        axes2[i].scatter(all_widths[i], all_powers, c='green', alpha=0.5, s=10)
        trend = np.polyfit(all_widths[i], all_powers, 1)
        trend_line = np.polyval(trend, all_widths[i])
        axes2[i].plot(all_widths[i], trend_line, 'r--')
        axes2[i].set_xlabel(f'{width_labels[i]} width (μm)')
        axes2[i].set_ylabel('Power (nW)')
        axes2[i].grid(True)
        axes2[i].set_title(f'{width_labels[i]} Width vs Power')
    plt.show()

    # Plot 3: Width vs Offset (7 subplots)
    fig3, axes3 = plt.subplots(7, 1, figsize=(10, 20), constrained_layout=True)
    fig3.suptitle('Transistor Width vs Offset Tradeoff: Wider transistors improve matching and reduce offset', y=1.02)
    for i in range(7):
        axes3[i].scatter(all_widths[i], all_offsets, c='purple', alpha=0.5, s=10)
        trend = np.polyfit(all_widths[i], all_offsets, 1)
        trend_line = np.polyval(trend, all_widths[i])
        axes3[i].plot(all_widths[i], trend_line, 'r--')
        axes3[i].set_xlabel(f'{width_labels[i]} width (μm)')
        axes3[i].set_ylabel('Offset (μV)')
        axes3[i].grid(True)
        axes3[i].set_title(f'{width_labels[i]} Width vs Offset')
    plt.show()
    
    # 2x2 Grid of Histograms and Scatter Plots in Golden Hue
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.patch.set_facecolor('#FFD700')
    for ax in axs.flat:
        ax.set_facecolor('#FFD700')
    
    # Histogram of Offset
    axs[0, 0].hist(all_offsets, bins=20, color='#DAA520', edgecolor='black')
    axs[0, 0].set_title('Offset (μV) Distribution')
    axs[0, 0].set_xlabel('Offset (μV)')
    axs[0, 0].set_ylabel('Frequency')
    
    # Histogram of Power
    axs[0, 1].hist(all_powers, bins=20, color='#DAA520', edgecolor='black')
    axs[0, 1].set_title('Power (nW) Distribution')
    axs[0, 1].set_xlabel('Power (nW)')
    axs[0, 1].set_ylabel('Frequency')
    
    # Histogram of Delay
    axs[1, 0].hist(all_delays, bins=20, color='#DAA520', edgecolor='black')
    axs[1, 0].set_title('Delay (ns) Distribution')
    axs[1, 0].set_xlabel('Delay (ns)')
    axs[1, 0].set_ylabel('Frequency')
    
    # Histogram of Total Width
    axs[1, 1].hist([optimizer.calculate_total_width(p) for p in particles], bins=20, color='#DAA520', edgecolor='black')
    axs[1, 1].set_title('Total Transistor Dimensions (μm)')
    axs[1, 1].set_xlabel('Total Width (μm)')
    axs[1, 1].set_ylabel('Frequency')
    
    plt.show()