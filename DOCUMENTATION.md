# سند فنی جامع - سیستم مدیریت انرژی ریزشبکه چندانرژی

## فهرست مطالب

1. [معماری سیستم](#معماری-سیستم)
2. [مدل‌سازی ریاضی](#مدل‌سازی-ریاضی)
3. [ماژول پیش‌بینی پیشرفته](#ماژول-پیش‌بینی-پیشرفته)
4. [عامل DRL آگاه به ریسک](#عامل-drl-آگاه-به-ریسک)
5. [روش‌های مرجع](#روش‌های-مرجع)
6. [معیارهای ارزیابی](#معیارهای-ارزیابی)
7. [پارامترهای تنظیم](#پارامترهای-تنظیم)

---

## معماری سیستم

### نمای کلی

سیستم MEMG شامل 4 لایه اصلی است:

```
لایه 4: ارزیابی و تحلیل
         ↑
لایه 3: تصمیم‌گیری (PPO Agent)
         ↑
لایه 2: پیش‌بینی (CNN-LSTM)
         ↑
لایه 1: فیزیک سیستم (MEMG Environment)
```

### اجزای فیزیکی ریزشبکه

#### 1. باس الکتریکی

**منابع تولید:**
- **PV (Photovoltaic)**: 200 kW ظرفیت نصب شده
  - ضریب ظرفیت متوسط: 20%
  - وابستگی به تابش خورشید و زمان روز
  - الگوی تولید: منحنی زنگوله‌ای در ساعات 6-18

- **WT (Wind Turbine)**: 150 kW ظرفیت
  - ضریب ظرفیت متوسط: 30%
  - تغییرپذیری بالا (intermittency)
  - منحنی توان غیرخطی

**ذخیره‌ساز:**
- **Battery**: 500 kWh ظرفیت
  - توان شارژ/دشارژ: ±100 kW
  - محدوده SOC: 10%-90%
  - راندمان شارژ: 95%
  - راندمان دشارژ: 95%
  - هزینه استهلاک: 0.05 $/kWh

#### 2. واحد CHP (Combined Heat and Power)

**مشخصات:**
- توان الکتریکی: 20-100 kW
- راندمان الکتریکی: 35%
- راندمان حرارتی: 45%
- راندمان کل: 80%
- نرخ رمپ: 30 kW/h
- نسبت حرارت به برق: 1.29:1

**مدل ریاضی:**
```
P_thermal_CHP = P_elec_CHP × (η_thermal / η_elec)
Q_gas_CHP = P_elec_CHP / η_elec
Cost_CHP = Q_gas × Price_gas + P_elec × Cost_maintenance
```

#### 3. بویلر گازسوز

**مشخصات:**
- توان حرارتی: 0-150 kW
- راندمان: 90%
- سوخت: گاز طبیعی

**مدل:**
```
Q_gas_boiler = P_thermal / η_boiler
Cost_boiler = Q_gas × Price_gas + P_thermal × Cost_maintenance
```

#### 4. اتصال شبکه

**محدودیت‌ها:**
- حداکثر خرید: 250 kW
- حداکثر فروش: 200 kW
- قیمت خرید پایه: 0.12 $/kWh
- قیمت فروش (FiT): 0.08 $/kWh
- ساعات پیک (ضریب 1.5): 17-21

---

## مدل‌سازی ریاضی

### فرمول‌بندی MDP

#### فضای حالت (State Space)

بردار حالت 10 بعدی:

```
s_t = [SOC_t, P̂_pv,t, P̂_wt,t, P̂_load_elec,t, P̂_load_th,t, 
       π_grid,t, h_t, d_t, P_chp,t-1, σ_forecast,t]
```

که:
- `SOC_t`: State of Charge باتری (نرمال شده)
- `P̂_*`: پیش‌بینی‌های توان (نرمال شده)
- `π_grid,t`: قیمت شبکه (نرمال شده)
- `h_t`: ساعت روز / 23
- `d_t`: روز هفته / 6
- `P_chp,t-1`: توان CHP قبلی (برای محدودیت رمپ)
- `σ_forecast,t`: عدم قطعیت پیش‌بینی

#### فضای عمل (Action Space)

بردار عمل 4 بعدی (نرمال شده):

```
a_t = [a_bat, a_chp, a_grid, a_boiler]
```

محدوده:
- `a_bat ∈ [-1, 1]`: منفی=شارژ، مثبت=دشارژ
- `a_chp ∈ [0, 1]`: نسبت به حداکثر توان
- `a_grid ∈ [-1, 1]`: منفی=صادرات، مثبت=واردات
- `a_boiler ∈ [0, 1]`: نسبت به حداکثر توان

**تبدیل به اعمال فیزیکی:**
```python
P_bat = a_bat × P_bat_max
P_chp = a_chp × P_chp_max
P_grid = a_grid × P_import_max  if a_grid > 0 else a_grid × P_export_max
P_boiler = a_boiler × P_boiler_max
```

#### تابع پاداش (Reward Function)

```
R_t = -(λ_cost × Cost_t + 
        λ_risk × CVaR_penalty_t + 
        λ_viol × Violation_penalty_t + 
        λ_smooth × Smoothness_penalty_t)
```

**اجزای هزینه:**
```
Cost_t = Cost_grid + Cost_CHP + Cost_boiler + Cost_bat_deg + Cost_bat_cycle
```

**جریمه نقض قیود:**
```
Violation_penalty = Σ (penalty_weight_i × violation_magnitude_i)
```

انواع نقض:
1. نقض محدوده SOC: 500 $/واحد
2. نقض توان باتری: 300 $/واحد
3. نقض محدودیت شبکه: 1000 $/واحد
4. نقض نرخ رمپ CHP: 200 $/واحد
5. عدم تعادل توان الکتریکی: 1000 $/واحد
6. عدم تعادل توان حرارتی: 800 $/واحد

#### قیود سیستم

**1. تعادل توان الکتریکی:**
```
P_pv,t + P_wt,t + P_chp,t + P_bat,t + P_grid,t = P_load_elec,t
```

**2. تعادل توان حرارتی:**
```
P_chp_thermal,t + P_boiler,t ≥ P_load_thermal,t
```

**3. دینامیک باتری:**
```
SOC_{t+1} = SOC_t + (ΔE_bat,t / E_bat_capacity)

ΔE_bat = { P_bat × η_charge × Δt      if P_bat < 0 (charging)
         { -P_bat / η_discharge × Δt  if P_bat > 0 (discharging)
```

**4. محدودیت SOC:**
```
SOC_min ≤ SOC_t ≤ SOC_max
0.1 ≤ SOC_t ≤ 0.9
```

**5. محدودیت رمپ CHP:**
```
|P_chp,t - P_chp,t-1| ≤ R_chp = 30 kW/h
```

**6. محدودیت عملیاتی CHP:**
```
if P_chp > 0: P_chp_min ≤ P_chp ≤ P_chp_max
              20 kW ≤ P_chp ≤ 100 kW
```

---

## ماژول پیش‌بینی پیشرفته

### معماری CNN-LSTM

```
Input (168 timesteps)
    ↓
[Wavelet Preprocessing]
    ↓
CNN Layers:
  Conv1D(1 → 64, kernel=3) + BatchNorm + ReLU + Dropout(0.2)
  Conv1D(64 → 128, kernel=3) + BatchNorm + ReLU + Dropout(0.2)
  Conv1D(128 → 64, kernel=3) + BatchNorm + ReLU + Dropout(0.2)
    ↓
LSTM Layers:
  LSTM(64 → 128, layers=2, dropout=0.2)
    ↓
[Multi-Head Attention (4 heads)]
    ↓
Fully Connected:
  Linear(128 → 64) + ReLU + Dropout(0.2)
  Linear(64 → 24)  # Output: 24-hour forecast
```

### پیش‌پردازش Wavelet

**تجزیه موجک:**
```python
coeffs = pywt.wavedec(signal, 'db4', level=3)
# coeffs[0]: Approximation (روند کلی)
# coeffs[1:]: Details (نوسانات)
```

**حذف نویز (Soft Thresholding):**
```
threshold = λ × std(detail_coeffs)
denoised = sign(x) × max(|x| - threshold, 0)
```

### برآورد عدم قطعیت (MC Dropout)

```python
# Generate N samples with dropout enabled
for i in range(30):
    sample_i = model(x)  # with dropout

# Uncertainty = standard deviation of samples
μ_forecast = mean(samples)
σ_forecast = std(samples)
```

### معیارهای ارزیابی پیش‌بینی

**1. MAE (Mean Absolute Error):**
```
MAE = (1/N) Σ |y_true - y_pred|
```

**2. RMSE (Root Mean Squared Error):**
```
RMSE = sqrt((1/N) Σ (y_true - y_pred)²)
```

**3. MAPE (Mean Absolute Percentage Error):**
```
MAPE = (100/N) Σ |y_true - y_pred| / y_true
```

**4. Coverage (پوشش فواصل اطمینان 95%):**
```
Coverage = (1/N) Σ 𝟙[y_true ∈ [μ - 1.96σ, μ + 1.96σ]]
```

---

## عامل DRL آگاه به ریسک

### معماری PPO

**Actor Network (Policy):**
```
State (10) → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(128) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(4) → [Tanh/Sigmoid] → Action
```

**Critic Network (Value):**
```
State (10) → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(128) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(1) → Value
```

### الگوریتم PPO

**1. جمع‌آوری تجربه:**
```
for t in episode:
    a_t, log_π(a_t|s_t), V(s_t) = π_θ(s_t)
    s_{t+1}, r_t ~ Env(s_t, a_t)
    Store (s_t, a_t, r_t, V(s_t), log_π(a_t|s_t))
```

**2. محاسبه GAE (Generalized Advantage Estimation):**
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

**3. بروزرسانی Policy (PPO Clipped):**
```
ratio_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
L^CLIP = min(ratio_t × A_t, 
             clip(ratio_t, 1-ε, 1+ε) × A_t)
```

**4. بروزرسانی Value:**
```
L^VF = (V_θ(s_t) - V_target)²
```

**5. تابع هدف کلی:**
```
L = L^CLIP - c₁ L^VF + c₂ H(π_θ)
```

که `H(π_θ)` آنتروپی برای تشویق exploration است.

### CVaR Risk-Aware Objective

**تعریف CVaR:**
```
CVaR_α(X) = E[X | X ≥ VaR_α(X)]
          = (1/(1-α)) ∫_{α}^{1} VaR_u(X) du
```

که:
- `α = 0.95`: سطح اطمینان
- `VaR_α`: Value at Risk (کوانتایل 95%)
- `CVaR_α`: میانگین 5% بدترین سناریوها

**محاسبه عملی:**
```python
threshold = percentile(returns, (1-α) × 100)
worst_returns = returns[returns ≤ threshold]
CVaR = mean(worst_returns)
```

**جریمه ریسک در Reward:**
```
risk_penalty = -CVaR × λ_risk_aversion
R_total = R_operational + risk_penalty
```

---

## روش‌های مرجع

### 1. DRL ساده (Simple DRL)

**تفاوت‌ها با روش پیشنهادی:**
- ❌ بدون پیش‌بینی پیشرفته (استفاده از actual values)
- ❌ بدون لایه ریسک (CVaR)
- ✅ معماری ساده‌تر: [128, 128]
- ✅ الگوریتم Policy Gradient ساده

### 2. بهینه‌سازی کلاسیک (Classical MILP)

**فرمول‌بندی:**
```
min Σ_t (Cost_grid,t + Cost_CHP,t + Cost_boiler,t)

subject to:
  Power balance constraints
  SOC dynamics
  Component limits
  Ramp constraints
```

**روش حل:**
- Linear Programming (LP)
- افق پیش‌بینی: 24 ساعت
- حل در هر timestep (MPC style)

**محدودیت‌ها:**
- فرض قطعی بودن پیش‌بینی‌ها
- زمان محاسبه بالا
- عدم یادگیری از تجربه

---

## معیارهای ارزیابی

### 1. معیارهای اقتصادی

```python
# هزینه کل
Total_Cost = Σ_t Cost_t

# هزینه متوسط
Mean_Cost = Total_Cost / T

# CVaR هزینه (ریسک)
CVaR_Cost = E[Cost | Cost ≥ VaR_0.95]
```

### 2. معیارهای قابلیت اطمینان

```python
# تعداد نقض قیود
Violation_Count = Σ_t 𝟙[any_constraint_violated]

# نرخ نقض
Violation_Rate = Violation_Count / T

# شدت متوسط نقض
Mean_Violation = Σ_t violation_magnitude_t / Violation_Count
```

### 3. معیارهای بهره‌برداری باتری

```python
# تعداد چرخه‌ها
Cycles = Σ_t |P_bat,t| / (2 × E_bat_capacity)

# نرم‌بودن عملکرد (smoothness)
Smoothness = 1 / (1 + mean(|P_bat,t - P_bat,t-1|))

# محدوده SOC استفاده شده
SOC_Range = max(SOC) - min(SOC)
```

### 4. معیارهای محیطی

```python
# نرخ بهره‌برداری از تجدیدپذیرها
Renewable_Utilization = Σ_t (P_pv + P_wt) / Σ_t P_load

# کاهش انتشار (فرضی)
Emission_Reduction = f(Renewable_Utilization)
```

---

## پارامترهای تنظیم

### راهنمای تنظیم پارامترها

#### 1. پارامترهای ریزشبکه

**ظرفیت باتری:**
- کوچک (250 kWh): انعطاف کمتر، نقض بیشتر
- متوسط (500 kWh): ✅ توصیه می‌شود
- بزرگ (1000 kWh): هزینه سرمایه‌گذاری بالا

**نسبت ظرفیت تجدیدپذیرها:**
```
Penetration_Ratio = (P_pv + P_wt) / P_load_avg

20-40%: محافظه‌کارانه
40-60%: ✅ متعادل
60-80%: چالش‌برانگیز، نیاز به باتری بزرگ
```

#### 2. پارامترهای PPO

**Learning Rate:**
```
Actor: 3e-4   # کمتر برای پایداری
Critic: 1e-3  # بیشتر برای همگرایی سریع
```

**Clip Epsilon:**
```
ε = 0.2   # استاندارد PPO
ε < 0.1   # محافظه‌کارانه‌تر
ε > 0.3   # exploration بیشتر
```

**GAE Lambda:**
```
λ = 0.95  # ✅ توازن bias-variance
λ → 1     # variance بالا
λ → 0     # bias بالا
```

#### 3. پارامترهای ریسک

**CVaR Alpha:**
```
α = 0.90  # محافظه‌کارانه‌تر (10% worst)
α = 0.95  # ✅ استاندارد (5% worst)
α = 0.99  # تمرکز روی extreme events
```

**Risk Aversion Weight:**
```
λ_risk = 0.1  # کم‌ریسک‌پذیر کم
λ_risk = 0.3  # ✅ متعادل
λ_risk = 0.5  # کم‌ریسک‌پذیر زیاد
```

---

## نکات پیاده‌سازی

### بهینه‌سازی عملکرد

1. **استفاده از GPU:**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

2. **Batch Processing:**
```python
batch_size = 32  # برای GPU
batch_size = 64  # برای GPU قوی
```

3. **حافظه:**
- Forecasting: ~2 GB RAM
- Training: ~4 GB GPU memory
- Evaluation: ~1 GB RAM

### رفع اشکال رایج

**مشکل: همگرایی نشدن PPO**
- کاهش learning rate
- افزایش buffer size
- کاهش clip epsilon

**مشکل: نقض زیاد قیود**
- افزایش وزن جریمه قیود
- کوچک کردن action space
- Pre-training با روش heuristic

**مشکل: پیش‌بینی ضعیف**
- افزایش lookback window
- تنظیم hyp parameters CNN-LSTM
- افزایش epoch های آموزش

---

## مراجع علمی

[1] Cao, D., et al. (2021). Deep Reinforcement Learning-Based Energy Storage Arbitrage. IEEE Trans. Smart Grid.

[2] Chow, Y., et al. (2018). Risk-Constrained Reinforcement Learning with Percentile Risk Criteria. JMLR.

[3] Mancarella, P. (2014). MES: An Overview of Concepts and Evaluation Models. Energy.

[4] Wang, H., et al. (2019). A Review of Deep Learning for Renewable Energy Forecasting. Energy Conversion and Management.

[5] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv.

---

**آخرین بروزرسانی**: دسامبر 2024
