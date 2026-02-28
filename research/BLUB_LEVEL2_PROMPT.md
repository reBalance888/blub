# BLUB Ocean — Level 2: SOC Rifts + Colony Memory

## Контекст

Текущее состояние после всех фиксов (baseline v1, 411 эпох):

```
TopSim = 0.10 устойчиво, 0.20 пик     ✅ 
Vocab  = 50                             ✅
social_MI = 2.9                         ✅
CSR    = 0.37                           ⏳ нужно 0.50
CIC    = 0.005                          ⏳ нужно 0.05
PosDis = 0.22                           ⏳ нужно 0.30
```

Язык работает — агенты используют разные звуки для разных контекстов, TopSim подтверждает композиционную структуру. Но CSR и CIC не растут дальше — нужны новые механики, а не тюнинг параметров.

## Цель Level 2

Поднять CSR с 0.37 до 0.50+ и CIC с 0.005 до 0.02+ через две новые механики:
1. **SOC Rift Eruptions** — создать punctuated equilibrium, давление на адаптацию языка
2. **Colony Memory** — культурный кэш на уровне колонии, не индивида

## Пять священных принципов (НЕ НАРУШАТЬ)

1. Никогда не хардкодить роли, конвенции или язык
2. Никаких нейросетей
3. Множественные давления одновременно
4. Никакой глобальной информации — только локальная
5. Не оптимизировать одну метрику

---

## Механика 1: SOC Rift Eruptions

### Зачем

Сейчас рифты — статичные точки ресурсов. Агенты находят рифт, встают рядом, зарабатывают. Нет причины СРОЧНО сообщать о находке — рифт никуда не денется. Поэтому CIC низкий: сообщения не меняют поведение, потому что спешить некуда.

SOC (самоорганизованная критичность) добавляет непредсказуемые "извержения" рифтов — кратковременные вспышки ×3-5 reward. Агент, который ПЕРВЫМ обнаружил извержение и СООБЩИЛ другим, получает огромное преимущество. Это создаёт давление на информативную коммуникацию.

### Реализация

**В ocean.py — новое свойство рифтов:**

```python
# Каждый рифт имеет pressure counter
rift.pressure = 0.0

# Каждый тик: pressure растёт медленно
rift.pressure += 0.01  # 1 единица за 100 тиков

# Когда pressure превышает threshold — ERUPTION
if rift.pressure >= 1.0:
    rift.erupting = True
    rift.eruption_ticks_remaining = 20  # длится 20 тиков
    rift.eruption_multiplier = 3.0      # reward ×3
    rift.pressure = 0.0                  # сброс

    # Каскад: 30% шанс триггернуть соседний рифт (в радиусе 10)
    for other_rift in nearby_rifts(rift, radius=10):
        if random.random() < 0.30:
            other_rift.pressure += 0.5  # толчок, не гарантированное извержение
```

**Modifiers:**
- Рифт который фармится (агенты рядом) накапливает pressure быстрее: `+= 0.01 * (1 + num_agents_nearby * 0.2)`
- Золотые рифты извергаются реже но мощнее (threshold 1.5, multiplier 5.0)
- Медные — чаще но слабее (threshold 0.7, multiplier 2.0)

**Извержение создаёт:**
1. Reward ×3-5 на 20 тиков — "золотая лихорадка"
2. Привлечение хищников — shark spawn +1 в радиусе 8 от извергающегося рифта через 10 тиков
3. Визуальный эффект во viewer (пульсирующее свечение) — для зрелищности

**Связь с языком:**
- Агент, обнаруживший извержение, имеет новый context situation = "eruption" (или максимальный urgency)
- Если он СООБЩИЛ и слушатели ПРИШЛИ к рифту за 20 тиков извержения — огромный reward для всех
- Если не сообщил или сообщил непонятно — извержение закончится до прихода остальных
- Это создаёт ПРЯМУЮ связь: качество сообщения → координация → reward → CIC растёт

**Config:**

```yaml
rifts:
  eruption:
    enabled: true
    pressure_per_tick: 0.01
    pressure_per_agent: 0.2       # дополнительное давление от фарма
    threshold_gold: 1.5
    threshold_silver: 1.0
    threshold_copper: 0.7
    multiplier_gold: 5.0
    multiplier_silver: 3.0
    multiplier_copper: 2.0
    duration_ticks: 20
    cascade_radius: 10
    cascade_chance: 0.30
    cascade_pressure_boost: 0.5
    predator_attract_delay: 10    # тиков до появления хищника
    predator_attract_radius: 8
```

**Как проверить что работает:**
- Среднее число извержений за эпоху: ~2-4 (при 8 рифтах)
- CIC должен вырасти — сообщение об извержении = высокая ценность
- CSR должен вырасти — новый context type (eruption) = новые конвенции

---

## Механика 2: Colony Memory

### Зачем

Сейчас cultural cache — один глобальный. Все умершие агенты сбрасывают веса в один котёл. Новорождённый наследует средние веса всех. Проблема: если колонии A и B развили разные диалекты, их веса усредняются → каша → MI деградация.

Colony Memory = каждая колония имеет свой cultural cache. Агент, умирающий в колонии, депонирует в кэш своей колонии. Новорождённый в колонии наследует кэш этой колонии. Это создаёт:
- Диалекты (разные колонии → разные конвенции)
- Более стабильную передачу знаний (нет смешивания)
- Путь к F_ST > 0.2 (необходимо для Babel Reef)

### Реализация

**В ocean.py — colony-level cache:**

```python
# Каждая колония имеет свой cultural cache
# Структура: colony.cultural_cache = {
#   "weights": averaged_W_of_colony_members,
#   "depositor_count": int,
#   "avg_lifetime_credits": float  # для weighted deposit
# }

# При формировании колонии (DBSCAN):
colony.cultural_cache = None  # пустой кэш новой колонии

# При смерти агента В КОЛОНИИ:
def deposit_to_colony_cache(agent, colony):
    weight = agent.lifetime_credits  # rich agents contribute more
    if colony.cultural_cache is None:
        colony.cultural_cache = {
            "weights": copy(agent.policy.W),
            "total_weight": weight,
            "count": 1
        }
    else:
        # Weighted running average
        total = colony.cultural_cache["total_weight"] + weight
        alpha = weight / total
        colony.cultural_cache["weights"] = (
            colony.cultural_cache["weights"] * (1 - alpha) +
            agent.policy.W * alpha
        )
        colony.cultural_cache["total_weight"] = total
        colony.cultural_cache["count"] += 1

# При рождении агента В КОЛОНИИ:
def inherit_from_colony_cache(newborn, colony):
    if colony.cultural_cache is not None:
        newborn.policy.import_knowledge(colony.cultural_cache["weights"], frac=0.60)
    else:
        # Fallback: global cache
        newborn.policy.import_knowledge(global_cache, frac=0.40)

# Epoch decay на colony cache:
for colony in colonies:
    if colony.cultural_cache:
        colony.cultural_cache["total_weight"] *= 0.98  # slowly forget
```

**В agents/social_agent.py — агент должен знать свою колонию:**

```python
# В think():
my_colony_id = state.get("colony_id", None)
# При deposit_to_cultural_cache — отправлять colony_id
```

**В ocean.py — передавать colony_id агенту:**

```python
# В get_agent_state():
colony_id = None
for colony in self.colonies:
    if agent_id in colony.members:
        colony_id = colony.id
        break
state["colony_id"] = colony_id
```

**Что происходит при распаде колонии:**
- Colony cache сохраняется ещё 500 тиков после распада (dormant)
- Если колония переформируется в том же месте — наследует dormant cache
- Если не переформируется — cache удаляется, depositors не теряются (уже в global)

**Взаимодействие global и colony cache:**
- Агент умирает В колонии → deposit в colony cache (primary) + global cache (secondary, weight × 0.3)
- Агент умирает ВНЕ колонии → deposit только в global cache
- Новорождённый В колонии → inherit 60% от colony cache
- Новорождённый ВНЕ колонии → inherit 40% от global cache (как раньше)

**Config:**

```yaml
cultural_cache:
  colony_cache_enabled: true
  colony_inheritance_frac: 0.60     # из colony cache
  global_inheritance_frac: 0.40     # из global cache (fallback)
  global_deposit_weight: 0.3        # colony agents also deposit 30% to global
  colony_cache_decay: 0.98          # per epoch
  dormant_cache_ticks: 500          # как долго хранить cache мёртвой колонии
```

**Как проверить что работает:**
- F_ST > 0 через 100 эпох (диалекты начинают различаться)
- MI деградация замедляется (colony cache ≠ global каша)
- CSR растёт внутри колоний (лучшая передача конвенций)

---

## Порядок реализации

### Шаг 1: Прочитай текущий код (НЕ МЕНЯЙ)

Прочитай целиком:
1. `server/ocean.py` — rift logic, colony logic, cultural cache, get_agent_state()
2. `agents/social_agent.py` — deposit_to_cultural_cache(), inherit_from_cultural_cache(), think()
3. `config.yaml` — все текущие параметры

Составь список:
- Где именно в tick loop находится rift reward calculation
- Где именно формируются колонии (DBSCAN)
- Где именно вызывается deposit/inherit cultural cache
- Какие поля уже есть в get_agent_state()
- Какие situation/target/urgency значения доступны для factored context

### Шаг 2: Реализуй SOC Eruptions (сначала)

Это проще и не зависит от colony memory:
1. Добавь pressure к каждому рифту
2. Добавь eruption logic в tick loop (после rift rewards? до?)
3. Добавь eruption multiplier к rift reward calculation
4. Добавь predator attraction при извержении
5. Добавь eruption info в get_agent_state() (для агента: is_rift_erupting)
6. Добавь конфиг параметры
7. Добавь WebSocket broadcast для viewer (erupting rifts glow)

### Шаг 3: Реализуй Colony Memory (после SOC)

1. Добавь cultural_cache к Colony class
2. Модифицируй deposit logic: colony vs global
3. Модифицируй inherit logic: colony vs global
4. Добавь colony_id в get_agent_state()
5. Добавь dormant cache logic
6. Добавь конфиг параметры

### Шаг 4: Тест

Запусти 100 эпох, проверь:
- Извержения происходят? (~2-4 за эпоху)
- Colony caches заполняются? (лог: colony X cache size = N deposits)
- CIC растёт? (цель: >0.01)
- CSR растёт? (цель: >0.40)
- MI деградация замедлилась?

---

## ВАЖНО

- НЕ ЛОМАЙ существующий код. Все новые фичи — через config kill switches (enabled: true/false)
- Сначала прочитай код, потом план, потом реализация
- Покажи мне план ПЕРЕД кодом — я хочу review
- Eruption CONTEXT: если factored context имеет situation/target/urgency — eruption может быть новым значением situation или urgency, БЕЗ добавления нового measurement в context space (это нарушит GaussianPolicy). Используй существующие значения: например eruption = situation "food" + urgency "high" (потому что извержение = суперресурс + срочность)
