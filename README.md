# UAV Path Planning with Reinforcement Learning

Проект для обучения беспилотного летательного аппарата (БПЛА) планированию пути с использованием алгоритмов глубокого обучения с подкреплением.

## Возможности

- **Обучение PPO агента** для навигации БПЛА в 3D пространстве
- **Динамические препятствия** и сложные сценарии полета
- **TensorBoard интеграция** для мониторинга обучения
- **Docker поддержка** для легкого развертывания

## Требования

- Docker и Docker Compose
- Или Python 3.11+ с зависимостями из `requirements.txt`

## Запуск с Docker

### 1. Сборка контейнера

`docker compose build`

### 2. Запуск контейнера

`docker compose up -d uav-training`

### 3. Подключение к контейнеру

`docker exec -it uav-path-planning bash`

### 4. Обучение модели

`python train_ppo.py`

### 5. Мониторинг обучения (TensorBoard)

В отдельном терминале

`docker-compose up -d tensorboard`

Откройте браузер: http://localhost:6006

### 6. Оценка обученной модели

`
python evaluate_ppo.py --model checkpoints/racing_ppo_uav_final.zip
`

## Локальная установка

`
pip install -r requirements.txt
`

## 📁 Структура проекта

```
uav-path-planning/
├── train_ppo.py          # Основной скрипт обучения
├── evaluate_ppo.py       # Скрипт оценки модели
├── envs/
│   └── base_3d_env.py           # Среда для обучения БПЛА
├── checkpoints/                 # Сохраненные модели
├── logs/                        # TensorBoard логи
├── Dockerfile                   # Docker конфигурация
├── docker-compose.yaml          # Docker Compose конфигурация
└── requirements.txt             # Python зависимости
```


## 📊 Мониторинг и анализ

### TensorBoard метрики:
- **Episode Reward** - награда за эпизод
- **Episode Length** - длительность эпизода
- **Success Rate** - процент успешных полетов
- **Policy Loss** - потери политики
- **Value Loss** - потери функции ценности
