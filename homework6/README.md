# Домашнее задание 6: Мониторинг ML моделей

Система мониторинга для ML моделей с использованием Prometheus, Grafana и уведомлениями в Telegram.

## Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Triton Server │    │  ML Metrics     │    │   Prometheus    │
│                 │◄───┤  Service        │───►│                 │
│ (ML Models)     │    │                 │    │ (Metrics Store) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Telegram Bot  │◄───┤  Alertmanager   │◄────────────┘
│                 │    │                 │    
│ (Notifications) │    │ (Alert Rules)   │    
└─────────────────┘    └─────────────────┘    
                                  ▲
                       ┌─────────────────┐
                       │    Grafana      │
                       │                 │
                       │ (Visualization) │
                       └─────────────────┘
```

## Компоненты

1. **Triton Inference Server** - ML модели для инференса
2. **ML Metrics Service** - Кастомный сервис для сбора метрик качества и производительности
3. **Prometheus** - Сбор и хранение метрик
4. **Grafana** - Визуализация метрик
5. **Alertmanager** - Управление алертами
6. **Telegram Bot** - Уведомления в Telegram

## Метрики

### Качественные метрики
- `ml_model_accuracy` - Точность модели
- `ml_model_precision` - Precision
- `ml_model_recall` - Recall
- `ml_model_f1_score` - F1 Score

### Метрики производительности
- `ml_model_avg_latency_ms` - Средняя задержка
- `ml_model_p95_latency_ms` - P95 задержка
- `ml_model_throughput_rps` - Пропускная способность
- `ml_model_total_requests` - Общее количество запросов
- `ml_model_failed_requests` - Количество неудачных запросов

## Настройка Telegram бота

### 1. Создание бота
1. Найдите [@BotFather](https://t.me/botfather) в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям для создания бота
4. Сохраните токен бота

### 2. Получение Chat ID
1. Добавьте бота в группу или напишите ему личное сообщение
2. Отправьте GET запрос: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Найдите `chat.id` в ответе

### 3. Настройка переменных окружения
Отредактируйте `docker-compose.yaml`:
```yaml
telegram-bot:
  environment:
    - TELEGRAM_BOT_TOKEN=YOUR_ACTUAL_BOT_TOKEN
    - TELEGRAM_CHAT_ID=YOUR_ACTUAL_CHAT_ID
```

## Запуск системы

### 1. Предварительные требования
```bash
# Docker и Docker Compose должны быть установлены
docker --version
docker-compose --version
```

### 2. Запуск всех сервисов
```bash
# Клонируйте репозиторий и перейдите в папку homework6
cd homework6

# Запустите все сервисы
docker-compose up --build -d
```

### 3. Проверка запуска
```bash
# Проверьте статус контейнеров
docker-compose ps

# Посмотрите логи
docker-compose logs triton
docker-compose logs ml_metrics_service
docker-compose logs telegram-bot
```

### 4. Доступ к сервисам
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Triton**: http://localhost:8000 (HTTP), localhost:8001 (gRPC)
- **ML Metrics**: http://localhost:8080/metrics

## Тестирование системы

### 1. Установка зависимостей для тестирования
```bash
pip install tritonclient[all] numpy requests
```

### 2. Быстрый тест
```bash
python test_alerts.py --action quick-test
```

### 3. Тест Telegram бота
```bash
python test_alerts.py --action test-bot
```

### 4. Генерация нагрузки
```bash
# Нагрузочный тест на 5 минут
python test_alerts.py --action load --duration 5
```

### 5. Симуляция деградации модели
```bash
# Симуляция проблем с конкретной моделью
python test_alerts.py --action degrade --model titanic_logistic_regression
```

## Настройка алертов

### Правила алертов (в `prometheus/alert_rules.yml`)

1. **ModelAccuracyLow** - Точность модели < 75%
2. **ModelLatencyHigh** - Средняя задержка > 100ms
3. **ModelF1ScoreLow** - F1 Score < 70%
4. **ModelErrorRateHigh** - Процент ошибок > 5%
5. **TritonServerDown** - Triton сервер недоступен

### Пороги алертов
```yaml
# Критический уровень (critical)
- F1 Score < 70%
- Error rate > 5%
- Triton server down

# Предупреждение (warning)
- Accuracy < 75%
- Latency > 100ms
- High server load
```

## Графики в Grafana

Автоматически создается дашборд "ML Models Monitoring" с панелями:

1. **Model Accuracy** - График точности моделей с порогом 75%
2. **Model Latency** - Задержка с порогом 100ms
3. **Model Throughput** - Пропускная способность в RPS
4. **Model F1 Score** - F1 метрика с порогом 70%

### Импорт дашборда
Дашборд автоматически импортируется при запуске. Если нужно импортировать вручную:
1. Откройте Grafana (http://localhost:3000)
2. Войдите (admin/admin123)
3. Перейдите в Import
4. Загрузите файл `grafana/dashboards/ml_models_dashboard.json`

## Структура проекта

```
homework6/
├── docker-compose.yaml              # Основная конфигурация
├── triton/
│   └── Dockerfile                   # Triton server
├── models/                          # ML модели
│   ├── titanic_logistic_regression/
│   └── titanic_random_forest/
├── prometheus/
│   ├── prometheus.yml               # Конфигурация Prometheus
│   └── alert_rules.yml              # Правила алертов
├── alertmanager/
│   └── alertmanager.yml             # Конфигурация алертов
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   └── dashboards/
│   └── dashboards/
│       └── ml_models_dashboard.json # Дашборд ML моделей
├── ml_metrics/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── metrics_service.py           # Сервис метрик
├── telegram_bot/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── telegram_bot.py              # Telegram бот
├── test_alerts.py                   # Скрипт тестирования
└── README.md
```

## Примеры алертов в Telegram

### Алерт о низкой точности
```
🤖 FIRING Alert

Alert: Model accuracy is too low
Model: titanic_logistic_regression
Severity: warning
Description: Model titanic_logistic_regression accuracy is 72% which is below 75% threshold
Time: 2025-06-09 16:30:15
```

### Алерт о высокой задержке
```
🤖 FIRING Alert

Alert: Model latency is too high
Model: titanic_random_forest
Severity: warning
Description: Model titanic_random_forest average latency is 150ms which is above 100ms threshold
Time: 2025-06-09 16:35:20
```

## Устранение неполадок

### 1. Triton не запускается
```bash
# Проверьте логи
docker-compose logs triton

# Проверьте модели
ls -la models/
```

### 2. Prometheus не собирает метрики
```bash
# Проверьте targets в Prometheus UI
# http://localhost:9090/targets

# Проверьте доступность сервисов
curl http://localhost:8080/metrics
curl http://localhost:8002/metrics
```

### 3. Telegram бот не отправляет сообщения
```bash
# Проверьте логи бота
docker-compose logs telegram-bot

# Проверьте переменные окружения
# Убедитесь, что токен и chat_id правильные

# Тест бота
curl -X POST http://localhost:5000/test
```

### 4. Grafana не показывает данные
```bash
# Проверьте подключение к Prometheus
# В Grafana: Configuration -> Data Sources -> Prometheus

# Убедитесь, что URL: http://prometheus:9090
```

## Остановка системы

```bash
# Остановить все сервисы
docker-compose down

# Остановить и удалить volumes (все данные будут потеряны)
docker-compose down -v
```

## Мониторинг производительности

Система автоматически обновляет метрики каждые 30 секунд:
- Тестирует качество на валидационных данных
- Измеряет производительность через тестовые запросы
- Экспортирует метрики в Prometheus
- Срабатывает алерты при превышении порогов

## Расширение системы

### Добавление новой модели
1. Скопируйте модель в папку `models/`
2. Добавьте имя модели в `ml_metrics/metrics_service.py`
3. Перезапустите сервис: `docker-compose restart ml_metrics_service`

### Добавление новых метрик
1. Определите метрику в `ml_metrics/metrics_service.py`
2. Добавьте логику сбора в `MLMetricsCollector`
3. Создайте правило алерта в `prometheus/alert_rules.yml`
4. Добавьте панель в Grafana дашборд 