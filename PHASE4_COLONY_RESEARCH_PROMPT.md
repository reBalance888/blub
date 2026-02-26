# Промпт для Claude Chat — Глубокий анализ Phase 4: Colony Dynamics + Pheromone Trails

Скопируй всё ниже в новый чат Claude.

---

Прими роль **мультидисциплинарного исследователя** объединяющего экспертизу в:
- **Эволюционной биологии** (eusociality, stigmergy, chemical communication у ракообразных, ant colony optimization, Bonabeau 1999, Camazine 2001)
- **Эволюционной лингвистике** (Lewis Signaling Games, Iterated Learning Model, креольские языки, грамматикализация, niche construction, Kirby 2015)
- **Multi-agent systems** (emergent communication: Lazaridou 2017, Kottur 2017, Mordatch 2018, Chaabouni 2020, swarm intelligence, stigmergic coordination)
- **Game theory & mechanism design** (стабильные равновесия, кооперативные стратегии, collective action problems, Ostrom 1990)

---

## Проект: BLUB Ocean — Phase 4 "Colony Dynamics"

Вечная симуляция 2D-океана где AI-агенты (лобстеры) развивают эмерджентный язык из 10 бессмысленных звуков. Язык рождается потому что координация = деньги. Phase 4 добавила химическую коммуникацию (феромонные следы) и коллективную организацию (колонии) — два новых слоя поверх существующей вокальной коммуникации.

**Реальные лобстеры** используют химическую коммуникацию (мочевые сигналы через нефропоры) для территориальности, доминантности и поиска пищи. Наши феромоны биологически обоснованы.

### Философия проекта
Хаос, но эволюционирующийся. Не метрики ради метрик, а живая экосистема. Язык — центральный, феромоны — дополнительный слой. Океан должен быть "живым" — пульсирующие следы, формирующиеся/распадающиеся кластеры, "дороги" между рифтами.

---

## АРХИТЕКТУРА (текущая, Phase 4)

### Мир
- 2D-сетка 100×100, адаптивный active zone (`size = sqrt(agents) × 7`), 33 агента → ~40×40
- **Рифты** — 3 типа: Gold (richness 8000, depletion 0.04), Silver (5000, 0.02), Copper (2000, 0.01)
- **Хищники** — спавнятся по плотности в зонах 10×10, confusion effect (kill_prob = 0.8/√group), social группа 3+ → дополнительное 0.8× снижение
- **Эпоха = 100 тиков** (~10 сек). Кредиты → BLUB-награды в конце
- **Turnover:** agent_lifetime = 3000 тиков → retirement + reconnect как naive

### Экономика
- Соло фарм у рифта = 0.1 кредитов/тик × group bonus
- Group reward: `log2(n+1) × 4` для n агентов (diminishing returns)
- Звуки: 2-sound message = 5 кредитов, 1-sound = 2 кредита
- Newcomer bonus: 2× первые 50 тиков
- Listener feedback: speaker получает 30% от reward слушателя (если слушатель у рифта в 10 тиков)

### 10 звуков
```
blub, glorp, skree, klak, mrrp, woosh, pop, zzzub, frrr, tink
```

### GaussianProductionPolicy (текущая — Phase 3d)
```python
# Ordinal policy: μ = sigmoid(W @ x), sound ~ Gaussian(μ*(n-1), σ=1.8)
# Per-position weights W[0] = [1.5, 0.5, 0.3, 0.0] (spatial)
#                      W[1] = [-1.5, -0.5, -0.3, 1.0] (social, flipped)
# REINFORCE gradient: push μ toward chosen sound if advantage > 0
# Topographic bonus: reward when close contexts → close sounds
```

### ContextDiscoverer (6 dims × 4 bins, dynamic refinement)
```
Dim 0: Расстояние до ближайшего рифта (Manhattan)
Dim 1: Richness % этого рифта (0-1)
Dim 2: Тип рифта (0=none, 1=copper, 2=silver, 3=gold)
Dim 3: Количество лобстеров рядом
Dim 4: Количество хищников рядом
Dim 5: Направление к рифту (0-7 компас, 8=нет)
```

### Cultural Cache (облique/horizontal transmission)
- При смерти/retirement агент deposit'ит GaussianPolicy weights + Comprehension counts
- Новый агент bootstrap'ит из cache (40% blend) + mentor от ближайшего опытного social (15% blend)
- Periodic deposits каждые 200 тиков (min age 300)

---

## PHASE 4: ЧТО ДОБАВЛЕНО

### 1. PheromoneMap (server/pheromone.py)

Sparse dict архитектура — только клетки с ненулевыми значениями.

**Два слоя:**
- **FOOD trails** — автоматически при получении rift reward. Интенсивность = reward × food_deposit_scale (0.02), cap 2.0
- **DANGER trails** — автоматически при гибели от хищника. Фиксированная интенсивность 3.0

**Физика каждый тик:**
```python
# Decay: intensity *= 0.95 (trail живёт ~60 тиков = 6 сек)
# Diffusion: 10% intensity → 4 соседа (по 2.5% каждому)
# Evaporation: intensity < 0.01 → удаление из dict
# Max intensity: 10.0
```

**Чтение:** агент получает `nearby_food_trails` и `nearby_danger_trails` в радиусе vision — список `{dx, dy, intensity}`.

### 2. ColonyManager (server/colony.py)

DBSCAN-like кластеризация:

**Lifecycle:**
1. **Detection** — каждый тик сканируем кластеры (seed → expand within radius 3, Manhattan dist)
2. **Candidacy** — кластер с ≥4 агентов = candidate
3. **Formation** — candidate стабилен ≥30 тиков → Colony (max 5 одновременно)
4. **Update** — colony matched to cluster по 50% member overlap, update center/members
5. **Dissolution** — alive members < formation_threshold/2 → dissolve

**Colony бонус:** members получают 1.2× rift rewards (20% bonus).

```python
@dataclass
class Colony:
    id: str
    center_x: float   # average position of members
    center_y: float
    members: set[str]  # lobster IDs
    formed_tick: int
    rift_id: str | None  # associated rift
    total_reward: float = 0.0
```

### 3. Social Agent — Pheromone-Following (agents/social_agent.py)

Movement priority chain (в порядке убывания приоритета):
```
1. Flee predator (immediate)
2. Heard rift sounds → move toward speaker (Bayesian comprehension)
3. Visible rift → move toward nearest
4. [NEW] Follow food pheromone gradient → move toward strongest food trail
5. [NEW] Avoid danger pheromone → move AWAY from strongest danger trail
6. Random walk (fallback)
```

Феромоны оставляются автоматически server-side (рефлекторное поведение, не через agent actions).

### 4. Viewer Rendering

- Food trails: зелёно-голубые клетки `rgba(0, 255, 180, intensity * 0.4)`
- Danger trails: красные клетки `rgba(255, 50, 50, intensity * 0.3)`
- Colony indicators: пунктирные золотые круги с labels
- Процедурные лобстеры: тело, клешни (анимация при speak), усики, глаза, хвост, ноги
- Мёртвые лобстеры перевёрнуты, полупрозрачные
- Colony members: золотая звёздочка сверху

### 5. Новые метрики
```
colony_count, avg_colony_size, food_trail_cells, danger_trail_cells
```

---

## РЕАЛЬНЫЕ РЕЗУЛЬТАТЫ: 23 ЭПОХИ ПОСЛЕ PHASE 4

Первый прогон (run 1, 23 эпохи):

| Epoch | TopSim | PosDis | MI    | Vocab | CSR  | Colonies | AvgSize | Food | Danger |
|-------|--------|--------|-------|-------|------|----------|---------|------|--------|
| 4     | 0.088  | 0.283  | 3.824 | 0     | 0.61 | 0        | 0       | 18   | 45     |
| 6     | 0.008  | 0.220  | 3.751 | 0     | 0.32 | 3        | 4.3     | 27   | 103    |
| 8     | 0.016  | 0.117  | 3.516 | 2     | 0.33 | 2        | 4.0     | 8    | 35     |
| 11    | 0.014  | 0.317  | 3.325 | 0     | 0.39 | 1        | —       | 27   | 87     |
| 14    | 0.002  | 0.276  | 3.139 | 0     | 0.36 | 1        | —       | 17   | 51     |
| 17    | 0.005  | 0.169  | 2.946 | 1     | 0.37 | 2        | —       | 21   | 58     |
| 20    | 0.003  | 0.210  | 2.783 | 0     | 0.36 | 2        | —       | 24   | 113    |
| 23    | 0.012  | 0.208  | 2.591 | 0     | 0.36 | 2        | —       | 34   | 54     |

Второй прогон (run 2, 9 эпох, свежие агенты):

| Epoch | TopSim | PosDis | MI    | Vocab | CSR  | Colonies | Food | Danger |
|-------|--------|--------|-------|-------|------|----------|------|--------|
| 5     | 0.020  | 0.148  | 3.401 | 1     | 0.35 | 0        | 6    | 54     |
| 7     | 0.071  | 0.319  | 3.498 | 0     | 0.24 | 0        | 0    | 7      |
| 9     | 0.035  | 0.146  | 3.260 | 1     | 0.30 | 3        | 7    | 71     |

---

## НАБЛЮДЕНИЯ И ПРОБЛЕМЫ

### Что работает:
1. **Колонии формируются и живут** — стабильно 1-3 колонии, avg size 4-5
2. **Феромонные следы работают** — food 8-89 клеток, danger 7-120 клеток
3. **Danger trails значительно больше food** — 50-120 vs 6-34. Это биологически реалистично (аверсивные стимулы сильнее)
4. **PosDis стабильно в зоне 0.15-0.32** — не деградирует
5. **CSR стабилизировался ~0.35** — не падает как раньше (было 99%→57%)
6. **Язык не деградирует** — MI медленно снижается (3.8→2.6) но PosDis держится

### Проблемы:
1. **TopSim всё ещё ~0.01** — topographic similarity практически нулевая. Похожие ситуации НЕ кодируются похожими звуками
2. **MI монотонно падает** — с 3.8 до 2.6 за 23 эпохи (потеря информативности)
3. **Vocabulary = 0-2** — почти нет стабильных sound→context ассоциаций с >60% consistency
4. **Food trails слабые** — reward × 0.02 даёт маленькие deposit'ы, food trails редко превышают 20-30 клеток
5. **Colony bonus малозаметен** — 1.2× не создаёт достаточного стимула для clustering поведения
6. **Нет "дорог" между рифтами** — food trails не формируют connected paths, только пятна у рифтов
7. **Danger trails доминируют** — 3.0 фиксированный deposit >> food deposits. Карта больше "красная" чем "зелёная"

### Метрические парадоксы:
- **PosDis высокий но Vocab=0** — звуки кодируют позиции диффузно, но нет СТАБИЛЬНЫХ ассоциаций
- **MI падает но CSR стабилен** — агенты реагируют на звуки (CSR 35%) но информация в звуках падает
- **Colonies=2-3 но нет colony-specific language** — колонии не развивают свои "диалекты"

---

## ЗАДАНИЕ

### Часть 1: Диагностика Phase 4

Для каждой из следующих проблем нужен глубокий анализ:

#### Проблема 1: TopSim ≈ 0 (отсутствие topographic structure)

GaussianProductionPolicy ДОЛЖНА давать topographic mapping (nearby μ → nearby sounds), но TopSim всё ещё ~0.01.

**Вопросы:**
- GaussianPolicy с W=[1.5, 0.5, 0.3, 0.0] и σ=1.8 — достаточно ли этого для TopSim? Или σ слишком большой (слишком много entropy)?
- REINFORCE gradient для Gaussian policy: правильно ли считается? `grad_mu = (target_mu - mu) * mu * (1-mu)` через sigmoid derivative
- Topographic bonus (reward за close ctx → close sounds) работает? Или его magnitude (2.0 reward) тонет в основном reward (~50-100)?
- Может TopSim просто невозможен с 10 звуками × 2 позиции = 100 комбинаций для ~200+ контекстов?
- Есть ли papers где TopSim > 0.3 получается с ordinal/Gaussian policy?

#### Проблема 2: MI монотонно падает (2.6 и ниже)

MI(signal; context) начинается высоко (~3.8) и медленно деградирует.

**Вопросы:**
- Cultural Cache blend 40% + mentor 15% — слишком сильное smoothing? Новые агенты получают "средний" язык вместо "лучшего"?
- Weight decay 0.9995 (в GaussianPolicy.decay_all) — может слишком агрессивный для W с 4 параметрами?
- Может MI падает потому что growing long_observations (rolling 5000) accumulates stale data?
- Или это естественное равновесие — MI stabilizes когда все агенты знают примерно одинаковое?
- Может нужно пересмотреть метрику: MI на коротком окне (1 эпоха) vs MI на длинном (3 эпохи)?

#### Проблема 3: Vocabulary = 0 (нет стабильных ассоциаций)

Vocabulary считается как: sound_seq → один context с >60% consistency, min 3 observations.

**Вопросы:**
- С GaussianPolicy звуки распределяются по Gaussian — может стохастичность слишком высокая для 60% порога?
- 10 звуков × 2 позиции: если μ ≈ 0.5 (centered), то Gaussian(σ=1.8) размазывает вероятность по 5-6 звукам
- Может порог 60% слишком высок? Или min observations = 3 слишком низко?
- Может нужен другой metric — не exact sequence match, а "top-1 most probable sound per position for this context"?
- Как vocabulary metric работает в литературе (Lazaridou, Chaabouni)?

#### Проблема 4: Food trails слабые, нет "дорог"

Food deposit = reward × 0.02, capped at 2.0. С typical reward ~20-50, deposit = 0.4-1.0. After decay 0.95/tick, trail lives ~60 ticks. Но trails не формируют connected paths.

**Вопросы:**
- Нужен ли deposit ПО ПУТИ к рифту (не только AT rift)? Реальные муравьи оставляют след по дороге
- Может food_deposit_scale = 0.02 слишком мал? Поднять до 0.05-0.1?
- Или нужен другой механизм: агент который ИДЁТ по food trail тоже deposit'ит (positive feedback)?
- Diffusion rate 0.1 — достаточно? Или trails затухают раньше чем диффундируют?
- Как ant colony optimization (Dorigo 1996) решает проблему trail bootstrapping?

#### Проблема 5: Colony-language interaction отсутствует

Колонии формируются, но не влияют на язык. Нет colony-specific "диалектов", нет colony identity.

**Вопросы:**
- Можно ли дать colony members шанс учить friends' language быстрее (horizontal transmission boost внутри colony)?
- Colony-specific sound: первый звук каждого colony member = colony identifier?
- Или colony должна влиять на context: добавить dim7 = colony_id в ContextDiscoverer?
- Может colony reward bonus (1.2×) нужно увеличить до 1.5-2.0× чтобы создать реальный incentive?
- В литературе по linguistic communities: как spatial clustering влияет на dialect formation?

---

### Часть 2: Эволюция системы

#### Вопрос A: Иерархия коммуникационных каналов

Сейчас у нас два канала: вокальный (10 звуков) и химический (феромоны). Как они должны ВЗАИМОДЕЙСТВОВАТЬ?

- Vocal = дальний + быстрый + дорогой (кредиты)
- Chemical = локальный + медленный + бесплатный (автоматический)
- Должны ли агенты ВЫБИРАТЬ между каналами? Или феромоны всегда автоматические?
- Может vocal нужен для referral ("иди ТУДА") а chemical для trail marking ("ТУТ было хорошо")?
- Есть ли multimodal communication systems в nature что подскажут правильную архитектуру?

#### Вопрос B: Expanding Map

В текущей системе active zone = fixed для данного количества агентов. Что если карта РАСШИРЯЕТСЯ по мере разведки?

- Scouts (агенты уходящие за пределы) → open new zones → new rifts?
- Или карта расширяется автоматически по epoch count?
- Как expanding territory создаёт communication pressure (далёкие scouts должны сообщать что нашли)?
- Риск: расширение может dilute density → меньше group rewards → меньше language pressure

#### Вопрос C: Predator-Pheromone Dynamics

Danger pheromone deposit = 3.0 (fixed) при каждом kill. Это создаёт "карту страха".

- Должны ли агенты учиться ИГНОРИРОВАТЬ старые danger trails? (decay 0.95 решает частично, но 60 тиков — долго)
- Может хищники должны ПРИВЛЕКАТЬСЯ danger trails (scavenger behavior)?
- Или наоборот: predators avoid high-danger areas (территория уже "обработана")?
- Баланс: danger avoidance vs missing rift opportunities (рифт рядом с danger zone)

#### Вопрос D: Emergent Roles

В ant colonies есть дифференциация: scouts, foragers, soldiers. Может ли что-то подобное ЭМЕРДЖЕНТНО возникнуть?

- Все агенты одинаковые (social), но ПОВЕДЕНИЕ может дифференцироваться через reinforcement?
- Агент который случайно много бродит (explorer) → находит рифты → получает reward → reinforced exploration
- Агент который стоит у рифта (defender) → colony bonus → reinforced staying
- Нужен ли explicit mechanism или достаточно emergent differentiation через разные starting weights?

---

### Часть 3: Конкретные рекомендации

Для каждого из 5 проблем + 4 вопросов эволюции:

#### Формат:

**Диагноз** (что именно ломается и почему, с привязкой к теории/papers)

**Рекомендованное решение** (одно основное, с обоснованием)

**Параметры** (конкретные числа)

**Псевдокод** (Python-like, без ML-фреймворков, реализуемый в текущей архитектуре)

**Метрики успеха** (target values и за сколько эпох)

**Риски** (что может сломаться)

---

### Часть 4: Итоговый план

#### Порядок реализации
Что делать первым, вторым, третьим. Зависимости. Что параллелизуется.

#### Предсказания
Если реализовать все рекомендации — какие метрики через 50 эпох:
- TopSim: ?
- PosDis: ?
- MI: ?
- Vocabulary: ?
- CSR: ?
- Colony count/size: ?
- Trail coverage: ?

Обоснуй цифры.

#### Открытые вопросы
Что ещё заметил? Какие дополнительные эксперименты нужны? Какие метрики не хватает?

#### Красные флаги
Что может фатально сломать систему если не учесть?
